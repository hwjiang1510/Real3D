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
from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
from dataset.constant import *
from utils.geo_utils import get_relative_pose
from utils.process_utils import process_images, get_cameras, get_rays_from_pose


'''
OmniObject3D is rendered with camera distance 4.0 (constant), and object is normalized in range [-1,1]
Thus, to make object normalized in [-1,1], we make the camera translation of scale=3.8
It has 5k training instances, 500 test instances
'''

### WARNING: MULTIVIEW is automatically used during test/val
### WARNING: SINGLEVIEW is only used during training
### THIS IS HARD CODED!
SCALE_RAW = 0.5

class Omniobject3D(Dataset):
    def __init__(self, config, split='train', mode='multiview', length=1, multiple_data=False, data_name='',
                 root='/data/dataset/omniobject3d/omniobject3D/OpenXD-OmniObject3D-New/raw/blender_renders'):
        self.config = config
        self.split = split
        self.root = root
        self.multiple_data = multiple_data
        self.data_name = data_name
        self.length = length
        self.mode = mode
        assert split in ['train', 'val', 'test']
        assert mode in ['multiview', 'singleview']

        self.use_consistency = config.train.use_consistency
        self.num_frame_consistency = config.train.num_frame_consistency
        self.consistency_curriculum = config.dataset.sv_curriculum

        self.img_size = config.dataset.img_size
        self.render_size = config.model.render_resolution if split == 'train' else config.test.eval_resolution
        self.normalization = config.dataset.normalization
        if self.mode == 'multiview':
            self.num_frame = config.dataset.omniobject3d_num_views if self.split == 'train' else config.dataset.omniobject3d_num_views + 1
        else:
            self.num_frame = 1

        # self.canonical_distance = SCALE_OBJAVERSE if config.dataset.mv_data_name == 'objaverse' else SCALE_OBJAVERSE_BOTH
        if config.dataset.mv_data_name == 'objaverse':
            self.canonical_distance = SCALE_OBJAVERSE
        elif config.dataset.mv_data_name == 'objaverse_both':
            self.canonical_distance = SCALE_OBJAVERSE_BOTH
        elif config.dataset.mv_data_name == 'none': # raw triposr
            self.canonical_distance = 1.8
        else:
            raise NotImplementedError
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
        data_split_file_path = os.path.join(data_root, 'omniobject3d.json')

        if not os.path.exists(data_split_file_path):
            self._split_data(data_split_file_path)

        with open(data_split_file_path, 'r') as f:
            data_split_file = json.load(f)

        print('Omniobject3D dataset instances: train {}, test {}'.format(len(data_split_file['train']),
                                                                         len(data_split_file['test'])))
        self.data_split.update(data_split_file)


    def _split_data(self, data_split_file_path):
        all_info = {'train': [], 'test': []}

        all_categories = os.listdir(self.root)

        for category in all_categories:
            category_info = {'train': [], 'test': []}
            category_path = os.path.join(self.root, category)
            all_instances = os.listdir(category_path)
            all_instances_valid = []
            for instance in all_instances:
                if category not in instance:
                    continue
                all_instances_valid.append(instance)
            num_instances = len(all_instances_valid)
            num_instances_test = max(1, int(num_instances * 0.1))

            # save the instance names for train and test
            category_info['train'] += all_instances_valid[:num_instances - num_instances_test]
            category_info['test'] += all_instances_valid[num_instances - num_instances_test:]

            for k in category_info.keys():
                all_info[k] += category_info[k]

        with open(data_split_file_path, 'w') as f:
            json.dump(all_info, f)


    def __len__(self):
        return len(self.seq_names)
    

    def __getitem__(self, idx):
        seq_name = self.seq_names[idx]
        category_name = seq_name[:-4]
        seq_path = os.path.join(self.root, category_name, seq_name, 'render')
        with open(os.path.join(seq_path, 'transforms.json'), 'r') as f:
            meta = json.load(f)

        # get intrinsics
        camera_angle_x = meta['camera_angle_x']
        focal_length = 0.5 / math.tan(0.5 * camera_angle_x)   # normalized with pixel
        K = torch.tensor([[focal_length * self.render_size, 0., 0.5 * self.render_size],
                          [0., focal_length * self.render_size, 0.5 * self.render_size],
                          [0., 0., 1.]]).float()
        
        # get image names
        imgs_path = os.path.join(seq_path, 'images')
        rgb_files = os.listdir(imgs_path)
        rgb_files_withIndices = [(it, int(it.split('_')[1].replace('.png', ''))) for it in rgb_files]  # (name in r_xxxxx.png, index)
        rgb_files_withIndices = sorted(rgb_files_withIndices, key=lambda x: x[1])
        rgb_files = [it[0] for it in rgb_files_withIndices]
        len_seq = len(rgb_files)
        if self.split == 'train':
            chosen_index = random.sample(range(len_seq), self.num_frame)
        else:
            chosen_index = list(range(self.num_frame))

        # load image and mask
        chosen_rgb_files = [rgb_files[it] for it in chosen_index]        
        imgs, masks = [], []
        for rgb_file in chosen_rgb_files:
            img, mask = self._load_frame(os.path.join(seq_path, 'images'), rgb_file)
            img = torch.tensor(img)
            mask = torch.tensor(mask)
            imgs.append(img)
            masks.append(mask)
        imgs = torch.stack(imgs)     # [t,c,h,w]
        masks = torch.stack(masks)   # [t,1,h,w]

        if self.mode == 'multiview':
            # get camera poses
            chosen_rgb_files_idx = [it.replace('.png', '') for it in chosen_rgb_files]
            frames_info = [meta['frames'][idx] for idx in chosen_index]
            frames_info_idx = [it['file_path'] for it in frames_info]
            assert frames_info_idx == chosen_rgb_files_idx
            cam_poses_gl = torch.tensor([it['transform_matrix'] for it in frames_info])    # opengl pose
            cam_poses_cv2 = cam_poses_gl @ opengl_to_cv2.unsqueeze(0)

            # normalize camera poses
            cam_poses_normalized_cv2 = self._canonicalize_cam_poses(cam_poses_cv2)
            cam_poses_normalized_opengl = cam_poses_normalized_cv2 @ cv2_to_opengl.unsqueeze(0)   # [n,4,4]

            rays_o, rays_d = get_rays_from_pose(cam_poses_normalized_opengl, focal=K[0,0].unsqueeze(0), size=self.render_size)

            input, input_mask = imgs[0].float(), masks[0].float()   # [c,h,w], [1,h,w]
            input = input * input_mask + (1 - input_mask) * 0.5
            
            sample = {
                    'input_image': input.unsqueeze(0),  # [1,c,h,w]
                    'rays_o': rays_o.float()[:],           # [n,h,w,3]
                    'rays_d': rays_d.float()[:],           # [n,h,w,3]
                    'render_images': imgs.float()[:],      # [n,c,h,w]
                    'render_masks': masks.float()[:],      # [n,1,h,w]
                    'seq_name': seq_name
                }
        else:
            fov = 0.691150367
            fx, fy = 0.5 / math.tan(0.5 * fov), 0.5 / math.tan(0.5 * fov)
            Ks = torch.tensor([[fx * self.render_size, 0., 0.5 * self.render_size],
                            [0., fy * self.render_size, 0.5 * self.render_size],
                            [0., 0., 1.]]).float().unsqueeze(0).repeat(self.num_frame,1,1)
            c2w = get_cameras(self.render_views, 0, self.canonical_distance, sampling=self.render_views_sample)
            rays_o, rays_d = get_rays_from_pose(c2w, focal=Ks[:,0,0], size=self.render_size)

            input, input_mask = imgs[0].float(), masks[0].float()   # [c,h,w], [1,h,w]
            input = input * input_mask + (1 - input_mask) * 0.5

            sample = {
                    'input_image': input.unsqueeze(0),              # [1,c,h,w]
                    'rays_o': rays_o.float(),                       # [n,h,w,3], only used in training
                    'rays_d': rays_d.float(),                       # [n,h,w,3], only used in training
                    'render_images': imgs[0].unsqueeze(0).float(),  # [n=1,c,h,w]
                    'render_masks': masks[0].unsqueeze(0).float(),  # [n=1,1,h,w]
                    'Ks': Ks,                                       # [n,3,3]
                }
            
            if self.use_consistency:
                c2w_consistency = self._get_cameras_consistency(c2w)
                rays_o_consistency, rays_d_consistency = get_rays_from_pose(c2w_consistency, focal=Ks[:,0,0], size=self.render_size)
                sample['rays_o_consistency'] = rays_o_consistency
                sample['rays_d_consistency'] = rays_d_consistency

        return sample
    

    def _get_cameras_consistency(self, c2w):
        # c2w in shape [n,4,4]
        # c2w[0] is the input view pose (identity rotation)
        c2w_input, c2w_last = c2w[:1], c2w[-self.num_frame_consistency:]
        c2w_invert = c2w_input @ torch.inverse(c2w_last) @ c2w_input
        return c2w_invert
    

    def _load_frame(self, seq_path, file_name):
        file_path = os.path.join(seq_path, file_name)
        img_pil = Image.open(file_path)
        img_np = np.asarray(img_pil)
        try:
            mask = Image.fromarray((img_np[:,:,3] > 0).astype(float))
        except:
            mask = Image.fromarray(np.logical_or(img_np[:,:,0]>0,
                                                  img_np[:,:,1]>0,
                                                  img_np[:,:,2]>0).astype(float))

        if self.config.dataset.white_bkg:
            # white background
            mask_255 = mask.point(lambda p: p * 255)
            white_background = Image.new('RGB', img_pil.size, (255, 255, 255))
            rgb = Image.composite(img_pil, white_background, mask_255.convert('L'))
        else:
            # black background
            rgb = Image.fromarray(img_np[:,:,:3])

        rgb = rgb.resize((self.img_size, self.img_size), Image.LANCZOS)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        rgb = np.asarray(rgb).transpose((2,0,1)) / 255.0                            # [3,H,W], in range [0,1]
        mask = np.asarray(mask)[:,:,np.newaxis].transpose((2,0,1))                  # [1,H,W], in range [0,1]

        if not self.config.dataset.white_bkg:
            rgb *= mask
        
        if self.config.train.normalize_img:
            normalization = transforms.Compose([
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250])),
            ])
            rgb = torch.from_numpy(rgb)
            rgb = normalization(rgb).numpy()

        return rgb, mask


    def _canonicalize_cam_poses(self, cam_poses):
        '''
        cam_poses: in opencv convension with shape [n,4,4]
        Process:
            1. get relative poses to the first image (input)
            2. apply a rotation to the camera poses, making the input pose is on z-axis with aligned rotation
            3. apply a scale factor, making the input pose translation is normalized
        '''
        cam_poses_rel = get_relative_pose(cam_poses[0], cam_poses)

        translation = cam_poses[0][:3,3]    # [3]
        scale = torch.sqrt((translation ** 2).sum())

        canonical_pose = self._build_canonical_pose(scale)
        cam_poses_rotated = canonical_pose.unsqueeze(0) @ cam_poses_rel

        cam_poses_scaled = self._normalize_scale(cam_poses_rotated, scale)

        return cam_poses_scaled
    

    def _build_canonical_pose(self, scale):
        '''
        build canonical pose in opencv pose convension
        the canonical pose has an identity rotation and a translation along the z-axis
        '''
        canonical_pose = torch.tensor([[0.0, 0.0, 1.0, scale],
                                        [1.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0]]) @ opengl_to_cv2
        return canonical_pose
    

    def _normalize_scale(self, camera_poses, distance):
        if self.normalization == 'constant_distance':
            # use a fixed object-camera distance
            # let the model guess the object scale itself
            distance_target = self.canonical_distance
            distnce_factor = distance_target / distance
            cam_translation_normalized = camera_poses[:,:3,3] * distnce_factor
            cam_poses_normalized = camera_poses.clone()
            cam_poses_normalized[:,:3,3] = cam_translation_normalized
            return cam_poses_normalized
        elif self.normalization == 'constant_scale':
            # use a fixed object scale
            # let the model guess the object-camera distance by itself
            scale_target = self.canonical_scale
            scale_factor = scale_target / SCALE_RAW
            cam_translation_normalized = camera_poses[:,:3,3] * scale_factor
            cam_poses_normalized = camera_poses.clone()
            cam_poses_normalized[:,:3,3] = cam_translation_normalized
            # print('objaverse-0123 normalizing by object scale, input view translation', distance * scale_factor)
            return cam_poses_normalized
        else:
            raise NotImplementedError

        
