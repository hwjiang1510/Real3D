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
from dataset.mvimagenet_sv import paste_crop
ImageFile.LOAD_TRUNCATED_IMAGES = True


class WILD_SV(Dataset):
    def __init__(self, config, split='train', length=1, multiple_data=False, data_name='',
                 root='/data-local/hanwen/dataset/singleview'):
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

        self.transform = ImageMaskAug() if config.dataset.sv_use_aug else None

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
        data_split_file_path = os.path.join(data_root, 'singleview.json')

        if not os.path.exists(data_split_file_path):
            self._split_data(data_split_file_path)

        with open(data_split_file_path, 'r') as f:
            data_split_file = json.load(f)

        print('Singleview (SV) dataset instances: train {}, test {}'.format(len(data_split_file['train']),
                                                                         len(data_split_file['test'])))
        self.data_split.update(data_split_file)


    def _split_data(self, data_split_file_path):
        '''
        Select data that have both images and masks
        '''
        all_instances_valid = []

        all_data = os.listdir(self.root)
        for dataset in all_data:
            all_instances = os.listdir(os.path.join(self.root, dataset))
            all_instances_valid += [os.path.join(dataset, it) for it in all_instances]

        random.shuffle(all_instances_valid)
        all_info = {'train': [], 'test': all_instances_valid}

        with open(data_split_file_path, 'w') as f:
            json.dump(all_info, f, indent=4)


    def __len__(self):
        return len(self.seq_names)
    

    def __getitem__(self, idx):
        img_name = self.seq_names[idx]
        img_path = os.path.join(self.root, img_name)

        # load image and mask
        img, mask = self._load_image(img_path)
        img, mask = self._process_image(img, mask)      # squared, centered

        if self.transform and self.split == 'train':
            img, mask = self.transform(img, mask)
            bkgd_color = 1.0 if self.config.dataset.white_bkg else 0.0
            img = img * mask + (1 - mask) * bkgd_color

        if self.config.train.normalize_img:
            imgs = self._normalize_img(imgs)

        fov = 0.691150367
        fx, fy = 0.5 / math.tan(0.5 * fov), 0.5 / math.tan(0.5 * fov)
        Ks = torch.tensor([[fx * self.render_size, 0., 0.5 * self.render_size],
                          [0., fy * self.render_size, 0.5 * self.render_size],
                          [0., 0., 1.]]).float().unsqueeze(0).repeat(self.num_frame,1,1)
        c2w = get_cameras(self.render_views, 0, self.camera_distance, sampling=self.render_views_sample)
        rays_o, rays_d = get_rays_from_pose(c2w, focal=Ks[:,0,0], size=self.render_size)

        input, input_mask = img.float(), mask.float()   # [c,h,w], [1,h,w]
        input = input * input_mask + (1 - input_mask) * 0.5

        sample = {
                'input_image': input.unsqueeze(0),              # [1,c,h,w]
                'rays_o': rays_o.float(),                       # [n,h,w,3], only used in training
                'rays_d': rays_d.float(),                       # [n,h,w,3], only used in training
                'render_images': img.unsqueeze(0).float(),      # [n=1,c,h,w]
                'render_masks': mask.unsqueeze(0).float(),      # [n=1,1,h,w]
                'Ks': Ks,                                       # [n,3,3]
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


    def _load_image(self, img_path):
        img_pil = Image.open(img_path)
        assert img_pil.mode == 'RGBA'

        r, g, b, a = img_pil.split()
        r_array = np.array(r)
        g_array = np.array(g)
        b_array = np.array(b)
        a_array = np.array(a)

        img = np.stack([r_array, g_array, b_array], axis=-1)
        img = np.asarray(img).transpose((2,0,1)) / 255.0                            # [3,h,w]
        mask = np.asarray(a_array).squeeze()[:,:,np.newaxis].transpose((2,0,1))
        mask = (mask > 225)                                                         # [1,h,w]
        return torch.tensor(img), torch.tensor(mask).float()
    

    def _process_image(self, img, mask, expand_ratio=1.7):
        # img in shape [3,h,w]
        # mask in shape [1,h,w]
        c,h,w = img.shape
        assert img.shape[-2:] == mask.shape[-2:]
        bkgd_color = 1.0 if self.config.dataset.white_bkg else 0.0
        if self.split == 'train':
            expand_ratio = random.random() * 0.5 + 1.45

        larger_dim = max(h, w)
        new_size = int(larger_dim * expand_ratio)
        pad_l = (new_size - w) // 2
        pad_r = new_size - w - pad_l
        pad_t = (new_size - h) // 2
        pad_b = new_size - h - pad_t

        img = F.pad(img, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=bkgd_color)
        mask = F.pad(mask, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0.0)

        # squared image
        processed_img = F.interpolate(img.unsqueeze(0), 
                                        size=(self.img_size, self.img_size),
                                        mode='bilinear', align_corners=False)[0]
        processed_mask = F.interpolate(mask.unsqueeze(0), 
                                        size=(self.img_size, self.img_size),
                                        mode='nearest')[0]
        return processed_img, processed_mask
    

    def _get_cameras_consistency(self, c2w):
        # c2w in shape [n,4,4]
        # c2w[0] is the input view pose (identity rotation)
        c2w_input, c2w_last = c2w[:1], c2w[-self.num_frame_consistency:]
        c2w_invert = c2w_input @ torch.inverse(c2w_last) @ c2w_input
        return c2w_invert
    

    def _normalize_img(self, imgs):
        normalization = transforms.Compose([
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250])),
            ])
        return normalization(imgs)

