import os
import pickle
import json
import tqdm
import cv2
import random
import torch
import numpy as np
import random
import math
import json
import time
from torch.utils.data import Dataset
from torchvision.transforms import functional as func_transforms
from torchvision import transforms
import torchvision
import torch.nn.functional as F
from PIL import Image, ImageFile, ImageFilter
from dataset.constant import *
from utils.geo_utils import get_relative_pose
from utils.process_utils import process_images, get_cameras, get_rays_from_pose, get_cameras_curriculum, ImageMaskAug
from utils import data_utils
from scipy.optimize import minimize
ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
MVIMageNet captures videos and camera parameters are computed using COLMAP
There is no canonical camera-to-world distance
We precompute the object scale and normalize the camera poses accordingly
'''

class MVIMageNet_SV(Dataset):
    def __init__(self, config, split='train', length=1, multiple_data=False, data_name='',
                 root='/data-local/hanwen/dataset/mvimgnet'):
        self.config = config
        self.split = split
        self.root = root
        self.multiple_data = multiple_data
        self.data_name = data_name
        assert split in ['train', 'val', 'test']
        self.length = length

        self.use_consistency = config.train.use_consistency
        self.num_frame_consistency = config.train.num_frame_consistency
        self.consistency_curriculum = config.dataset.sv_curriculum
        self.rerender_consistency_input = config.train.rerender_consistency_input

        self.img_size = config.dataset.img_size
        self.num_frame = 1
        self.render_size = config.model.render_resolution if split == 'train' else config.test.eval_resolution

        self.camera_distance = SCALE_OBJAVERSE if config.dataset.mv_data_name == 'objaverse' else SCALE_OBJAVERSE_BOTH

        self.render_views = config.dataset.sv_render_views
        self.render_views_sample = config.dataset.sv_render_views_sample

        self.data_split = {}
        self._load_dataset()
        self.seq_names = []
        if self.split == 'train':
            self.seq_names += self.data_split['train']
        else:
            self.seq_names += self.data_split['test']
            if self.split == 'val':
                self.seq_names = self.seq_names[::config.eval_vis_freq]

    def _load_dataset(self):
        data_root = './dataset/split_info'
        os.makedirs(data_root, exist_ok=True)
        data_split_file_path = os.path.join(data_root, 'mvimgnet.json')

        if not os.path.exists(data_split_file_path):
            self._split_data(data_split_file_path)

        with open(data_split_file_path, 'r') as f:
            data_split_file = json.load(f)

        print('MVImageNet (SV) dataset instances: train {}, test {}'.format(len(data_split_file['train']),
                                                                         len(data_split_file['test'])))
        self.data_split.update(data_split_file)


    def _split_data(self, data_split_file_path):
        '''
        Select data that have both images and masks
        '''
        all_instances_valid = []

        all_dirs = os.listdir(os.path.join(self.root, 'image'))
        for dir in all_dirs:
            all_instances = os.listdir(os.path.join(self.root, 'image', dir))
            for instance in all_instances:
                cur_instance = os.path.join(dir, instance)

                imgs_path = os.path.join(self.root, 'image', dir, instance, 'images')
                sparse_path = os.path.join(self.root, 'image', dir, instance, 'sparse/0')
                mask_path = imgs_path.replace('images', '').replace('mvimgnet/image', 'mvimgnet/mask')
                if (os.path.exists(imgs_path)) and (os.path.exists(sparse_path)) and (os.path.exists(mask_path)):
                    imgs = [it for it in os.listdir(imgs_path) if 'bg_removed' not in it]
                    masks = [it for it in os.listdir(mask_path) if it.endswith('.png')]
                    if len(imgs) == len(masks):
                        if set(os.listdir(sparse_path)) == set(['cameras.bin', 'images.bin', 'points3D.bin', 'project.ini']):
                            all_instances_valid.append(cur_instance)
        random.shuffle(all_instances_valid)
        all_info = {'train': all_instances_valid[:-1000], 'test': all_instances_valid[-1000:]}

        with open(data_split_file_path, 'w') as f:
            json.dump(all_info, f)
    

    def __len__(self):
        return len(self.seq_names)
    

    def __getitem__(self, idx):
        # seq_name = random.choice(self.seq_names)
        seq_name = self.seq_names[idx]                                  # e.g. 0/0501111e
        seq_path = os.path.join(self.root, 'image', seq_name)

        # get image names
        seq_path_imgs = os.path.join(seq_path, 'images')
        rgb_files = os.listdir(seq_path_imgs)
        rgb_files = [it for it in rgb_files if 'bg_removed' not in it]
        rgb_files_withIndices = [(it, int(it.replace('.jpg', ''))) for it in rgb_files]  # (name in xxx.png, index)
        rgb_files_withIndices = sorted(rgb_files_withIndices, key=lambda x: x[1])
        rgb_files = [it[0] for it in rgb_files_withIndices]
        len_seq = len(rgb_files)
        if self.split == 'train':
            chosen_index = random.sample(range(len_seq), self.num_frame)
        else:
            chosen_index = list(range(self.num_frame))

        # get image and mask
        chosen_rgb_files = [rgb_files[it] for it in chosen_index]        
        imgs, masks = [], []
        for rgb_file in chosen_rgb_files:
            img, mask = self._load_frame(seq_path_imgs, rgb_file)
            img = torch.tensor(img)
            mask = torch.tensor(mask)
            imgs.append(img)
            masks.append(mask)
        imgs = torch.stack(imgs)                # [t,c,h,w], images without cropping, in range [0,1]
        masks = torch.stack(masks).float()      # [t,1,h,w], with value 0/1

        # crop and process the image, mask and intrinsics
        try:
            bboxes = self._process_crop(masks, seq_path_imgs, chosen_rgb_files, expand_rate=1.7)  # [t,4], in XYXY format
        except:
            bboxes = torch.zeros(self.num_frame, 4)
            print(seq_name)
        imgs_crop, masks_crop = self._process_images(bboxes, imgs, masks)   # crop and resize
        imgs_crop = self._mask_background(imgs_crop, masks_crop)            # turn background white
        if self.config.train.normalize_img:
            imgs_crop = self._normalize_img(imgs_crop)

        fov = 0.691150367
        fx, fy = 0.5 / math.tan(0.5 * fov), 0.5 / math.tan(0.5 * fov)
        Ks = torch.tensor([[fx * self.render_size, 0., 0.5 * self.render_size],
                          [0., fy * self.render_size, 0.5 * self.render_size],
                          [0., 0., 1.]]).float().unsqueeze(0).repeat(self.num_frame,1,1)
        c2w = get_cameras(self.render_views, 0, self.camera_distance, sampling=self.render_views_sample)
        # if self.consistency_curriculum != 'none':
        #     c2w_curriculum = self._get_camerad_curriculum()
        #     c2w[-self.num_frame_consistency:] = c2w_curriculum
        rays_o, rays_d = get_rays_from_pose(c2w, focal=Ks[:,0,0], size=self.render_size)

        input, input_mask = imgs_crop[0].float(), masks_crop[0].float()   # [c,h,w], [1,h,w]
        input = input * input_mask + (1 - input_mask) * 0.5

        sample = {
                'input_image': input.unsqueeze(0),       # [1,c,h,w]
                'rays_o': rays_o.float(),                # [n,h,w,3], only used in training
                'rays_d': rays_d.float(),                # [n,h,w,3], only used in training
                'render_images': imgs_crop.float(),      # [n=1,c,h,w]
                'render_masks': masks_crop.float(),      # [n=1,1,h,w]
                'Ks': Ks,                                # [n,3,3]
                'c2w': c2w,                              # [n,4,4]
                'seq_name': seq_name
            }
        
        if self.use_consistency:
            c2w_consistency = self._get_cameras_consistency(c2w)
            rays_o_consistency, rays_d_consistency = get_rays_from_pose(c2w_consistency, focal=Ks[:,0,0], size=self.render_size)
            sample['rays_o_consistency'] = rays_o_consistency
            sample['rays_d_consistency'] = rays_d_consistency
        
        if self.rerender_consistency_input and self.use_consistency:
            rays_o_hres, rays_d_hres = get_rays_from_pose(c2w, 
                                                          focal=Ks[:,0,0] / self.render_size * self.img_size, 
                                                          size=self.img_size)
            sample['rays_o_hres'] = rays_o_hres
            sample['rays_d_hres'] = rays_d_hres

        return sample


    def _load_frame(self, img_root, img_name):
        # load image
        img_path = os.path.join(img_root, img_name)
        assert os.path.exists(img_path)
        img_pil = Image.open(img_path)
        w, h = img_pil.size

        # load mask
        mask_path = img_path.replace('images', '').replace('mvimgnet/image', 'mvimgnet/mask') + '.png'
        assert os.path.exists(mask_path)
        mask_pil = Image.open(mask_path)
        mask_pil = mask_pil.resize((w, h), Image.NEAREST)

        # format
        img = np.asarray(img_pil).transpose((2,0,1)) / 255.0                        # [3,H,W]
        mask = np.asarray(mask_pil).squeeze()[:,:,np.newaxis].transpose((2,0,1))
        mask = (mask > 225)                                                         # [1,H,W]
        return img, mask
    

    def _process_crop(self, masks, seq_path_imgs, chosen_rgb_files, expand_rate=1.7):
        bboxes = []
        for (file_name, mask) in zip(chosen_rgb_files, masks):
            img_path = os.path.join(seq_path_imgs, file_name)
            mask_path = img_path.replace('images', '').replace('mvimgnet/image', 'mvimgnet/mask') + '.png'
            bbox_path = mask_path.replace('.jpg.png', '.txt')
            # if os.path.exists(bbox_path):
            #     bbox = np.loadtxt(bbox_path)
            # else:
            bbox = find_bounding_box(mask.numpy())
                # np.savetxt(bbox_path, bbox)
            # bbox are represented in XYXY format (x-width, Y-height)
            bbox = square_bbox(bbox)
            bbox_expand = expand_bbox(bbox, expand_rate)
            bboxes.append(torch.tensor(bbox_expand))
        bboxes = torch.stack(bboxes)
        return bboxes
    
    
    def _process_images(self, bboxes, imgs, masks):
        N, C, H, W = imgs.shape
        assert imgs.shape[-2:] == masks.shape[-2:]
        bkgd_color = 1.0 if self.config.dataset.white_bkg else 0.0
        processed_imgs = torch.zeros((N, C, self.img_size, self.img_size), dtype=imgs.dtype) + bkgd_color
        processed_masks = torch.zeros((N, 1, self.img_size, self.img_size), dtype=masks.dtype)
        
        for i in range(N):
            try:
                bbox = bboxes[i]
                x1, y1, x2, y2 = bbox
                valid_x1, valid_y1 = max(int(x1), 0), max(int(y1), 0)
                valid_x2, valid_y2 = min(int(x2), W), min(int(y2), H)
                crop_img = imgs[i, :, valid_y1:valid_y2, valid_x1:valid_x2]
                crop_mask = masks[i, :, valid_y1:valid_y2, valid_x1:valid_x2]

                target_img = torch.full((C, y2-y1, x2-x1), 255, dtype=imgs.dtype)
                target_mask = torch.zeros((1, y2-y1, x2-x1), dtype=masks.dtype)

                target_start_x, target_start_y = max(-int(x1), 0), max(-int(y1), 0)
                target_img = paste_crop(crop_img, target_img, target_start_x, target_start_y)
                target_mask = paste_crop(crop_mask, target_mask, target_start_x, target_start_y)

                processed_imgs[i] = F.interpolate(target_img.unsqueeze(0), 
                                                size=(self.img_size, self.img_size),
                                                mode='bilinear', align_corners=False)[0]
                processed_masks[i] = F.interpolate(target_mask.unsqueeze(0), 
                                                size=(self.img_size, self.img_size),
                                                mode='nearest')[0]
            except:
                continue
        return processed_imgs, processed_masks
    

    def _mask_background(self, imgs, masks):
        masks = masks.expand_as(imgs)
        white_bg = torch.ones_like(imgs).to(imgs)
        masked_imgs = torch.where(masks>0, imgs, white_bg)
        return masked_imgs


    def _normalize_img(self, imgs):
        normalization = transforms.Compose([
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250])),
            ])
        return normalization(imgs)
    

    def _get_cameras_consistency(self, c2w):
        # c2w in shape [n,4,4]
        # c2w[0] is the input view pose (identity rotation)
        c2w_input, c2w_last = c2w[:1], c2w[-self.num_frame_consistency:]
        c2w_invert = c2w_input @ torch.inverse(c2w_last) @ c2w_input
        return c2w_invert
    


def distance_to_rays(point, rays_start, rays_dir):
    """
    Calculate the sum of squared distances from a point to all rays, given rays' starting points and directions.
    """
    point = np.array(point)
    vectors = rays_start - point
    distances_squared = np.sum(np.square(vectors - np.sum(vectors * rays_dir, axis=1)[:, np.newaxis] * rays_dir), axis=1)
    return np.sum(distances_squared)


def find_bounding_box(mask):
    '''
    mask in float of value 0 and 1 (foreground)
    '''
    mask = mask.squeeze()
    if np.sum(mask) == 0:
        h, w = mask.shape[-2:]
        return np.array([0,0,0,0])

    rows, cols = np.where(mask == 1)
    bbox = [np.min(cols), np.min(rows), np.max(cols), np.max(rows)]
    return np.array(bbox)


def square_bbox(bbox):
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    size = max(width, height)
    center = (bbox[:2] + bbox[2:]) / 2
    leftTop_new, rightBottom_new = center - size // 2, center + size // 2
    bbox_square = np.concatenate([leftTop_new, rightBottom_new]).astype(np.int32)
    return bbox_square


def expand_bbox(bbox, expand_rate=1.7):
    '''
    bbox is represented as XYXY format, X-width and Y-height, in shape [4,]
    '''
    leftTop, rightBottom = bbox[:2], bbox[2:]
    center = (leftTop + rightBottom) / 2.0
    size = bbox[2] - bbox[0]
    size_new = int(expand_rate * size)
    leftTop_new = center - size_new // 2
    rightBottom_new = center + size_new // 2
    bbox_expand = np.concatenate([leftTop_new, rightBottom_new]).astype(np.int32)
    return bbox_expand


def paste_crop(crop, target, start_x, start_y):
    H, W = crop.shape[-2:]
    target[:, start_y: start_y+H, start_x: start_x+W].copy_(crop)
    return target



if __name__ == '__main__':
    from omegaconf import OmegaConf
    from utils import exp_utils
    from torch.utils.data import DataLoader

    cfg = '/data-local/hanwen/wild_lrm_dev/config/mvimgnet/train.yaml'
    config = OmegaConf.load(cfg)
    config = exp_utils.to_easydict_recursively(config)

    data = MVIMageNet_SV(config, split='train')
    loader = DataLoader(data, batch_size=1, shuffle=False)

    for batch_idx, sample in enumerate(loader):
        print(batch_idx)
        #breakpoint()
    