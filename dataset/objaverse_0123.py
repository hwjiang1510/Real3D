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
import glob
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms import functional as func_transforms
from torchvision import transforms
import torchvision
from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
from dataset.constant import *
from utils.geo_utils import get_relative_pose
from utils.process_utils import process_images, get_cameras, get_rays_from_pose
from tsr.utils import get_spherical_cameras
from concurrent.futures import ThreadPoolExecutor
import concurrent

from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
Objaverse (zero-1-to-3) is rendered with random camera poses:
    - The distance is sampled in range [1.5, 2.2]
    - The object is normalized in range [-0.5,0.5]
'''
SCALE_RAW = 0.5

class Objaverse_0123(Dataset):
    def __init__(self, config, split='train', multiple_data=False, data_name='',
                 root='/datastor1/hanwen/objaverse2/views_release'):
        self.config = config
        self.split = split
        self.root = root
        self.multiple_data = multiple_data
        self.data_name = data_name
        assert split in ['train', 'val', 'test']

        self.img_size = config.dataset.img_size
        self.num_frame = config.dataset.num_frame if self.split == 'train' else config.dataset.num_frame + 1
        self.render_size = config.model.render_resolution
        
        self.normalization = config.dataset.normalization
        self.canonical_distance = SCALE_OBJAVERSE
        self.canonical_scale = SCALE_TRIPLANE_SAFE

        # intrinsics
        FoV_degree = 49.1
        FoV_radians = math.radians(FoV_degree)
        self.focal = 0.5 / math.tan(0.5 * FoV_radians)    # normalized

        # split data for traing and testing
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
        data_split_file_path = os.path.join(data_root, 'objaverse_0123.json')

        if not os.path.exists(data_split_file_path):
            self._split_data(data_split_file_path)

        with open(data_split_file_path, 'r') as f:
            data_split_file = json.load(f)

        print('Objaverse (Zero123) dataset instances: train {}, test {}'.format(len(data_split_file['train']),
                                                                         len(data_split_file['test'])))
        
        self.data_split.update(data_split_file)

    
    def _split_data(self, data_split_file_path):
        gobjaverse_split_path = '/data-local/hanwen/objaverse/gobjaverse_280k_index_to_objaverse.json'
        with open(gobjaverse_split_path, 'r') as f:
            gobjaverse_split = json.load(f)
        
        def check_instance(k, v):
            instance_name = v.split('/')[1].split('.')[0]
            instance_path = os.path.join(self.root, instance_name)
            if os.path.exists(instance_path) and len(os.listdir(instance_path)) == 24:
                return instance_name
            return None

        instances = []
        # Use ThreadPoolExecutor to parallelize I/O operations
        with ThreadPoolExecutor(max_workers=100) as executor:
            future_to_instance = {executor.submit(check_instance, k, v): k for k, v in gobjaverse_split.items()}
            for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_instance), total=len(future_to_instance)):
                instance_name = future.result()
                if instance_name:
                    instances.append(instance_name)
        
        num_instances = len(instances)
        num_instances_test = 1000
        random.shuffle(instances)
        all_info = {
            'train': instances[:num_instances - num_instances_test],
            'test': instances[num_instances - num_instances_test:]
        }

        with open(data_split_file_path, 'w') as f:
            json.dump(all_info, f)
            
    
    def __len__(self):
        return len(self.seq_names)
    

    def __getitem__(self, idx):
        seq_name = self.seq_names[idx]
        seq_path = os.path.join(self.root, seq_name)

        rgb_files = glob.glob(os.path.join(seq_path, '*.png'))
        len_seq = len(rgb_files)

        if self.split == 'train':
            chosen_idx = random.sample(range(len_seq), self.num_frame)
        else:
            chosen_idx = list(range(self.num_frame))
        chosen_rgb_files = [rgb_files[it] for it in chosen_idx] 
        chosen_npy_files = [it.replace('.png', '.npy') for it in chosen_rgb_files]

        imgs, masks = [], []
        for rgb_file, npy_file in zip(chosen_rgb_files, chosen_npy_files):
            img, mask = self._load_frame(seq_path, rgb_file)
            img = torch.tensor(img)
            mask = torch.tensor(mask)
            imgs.append(img)
            masks.append(mask)
        imgs = torch.stack(imgs)     # [n,c,h,w]
        masks = torch.stack(masks)   # [n,1,h,w]

        cam_poses_cv2, K = self._load_pose(seq_path, chosen_npy_files)   # [n,4,4], [3,3], actually extrinsics (c2w)

        # normalize camera poses
        cam_poses_normalized_cv2 = self._canonicalize_cam_poses(cam_poses_cv2)
        cam_poses_normalized_opengl = cam_poses_normalized_cv2 @ cv2_to_opengl.unsqueeze(0)   # [n,4,4]

        # get input
        input, input_mask = imgs[0].float(), masks[0].float()   # [c,h,w], [1,h,w]
        input = input * input_mask + (1 - input_mask) * 0.5 #torch.tensor([0.485, 0.456, 0.406])[:,None,None]

        # get rays
        rays_o, rays_d = get_rays_from_pose(cam_poses_normalized_opengl, focal=K[0,0].unsqueeze(0), size=self.render_size)

        # c2w = get_cameras(4, 0, SCALE_OBJAVERSE, sampling='uniform')
        # rays_o, rays_d = get_rays_from_pose(c2w, focal=K[0,0].unsqueeze(0), size=self.render_size)

        sample = {
                'input_image': input.unsqueeze(0),  # [1,c,h,w]
                'rays_o': rays_o.float(),           # [n,h,w,3]
                'rays_d': rays_d.float(),           # [n,h,w,3]
                'render_images': imgs.float(),      # [n,c,h,w]
                'render_masks': masks.float(),      # [n,1,h,w]
                'seq_name': seq_name
            }
        return sample


    def _load_frame(self, seq_path, img_name):
        file_path = os.path.join(seq_path, img_name)
        img_pil = Image.open(file_path)
        img_np = np.asarray(img_pil)

        try:
            mask = Image.fromarray((img_np[:,:,3] > 0).astype(float))
        except:
            mask = Image.fromarray(np.logical_and(img_np[:,:,0]==0,
                                                  img_np[:,:,1]==0,
                                                  img_np[:,:,2]==0).astype(float))

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


    def _load_pose(self, seq_path, npy_files):
        npy_paths = [os.path.join(seq_path, it) for it in npy_files]
        ext = np.array([np.load(it) for it in npy_paths])
        ext = torch.from_numpy(ext).float()
        ext = torch.cat([ext, torch.tensor([0.,0.,0.,1.]).reshape(1,1,4).repeat(len(npy_files),1,1)], dim=1)
        poses_opengl = torch.inverse(ext)
        poses_cv2 = poses_opengl @ opengl_to_cv2.unsqueeze(0)

        fx, fy = self.focal, self.focal
        K = torch.tensor([[fx * self.render_size, 0., 0.5 * self.render_size],
                          [0., fy * self.render_size, 0.5 * self.render_size],
                          [0., 0., 1.]]).float()
        return poses_cv2, K
    

    def _canonicalize_cam_poses(self, cam_poses):
        '''
        cam_poses: in opencv convension with shape [n,4,4]
        Process:
            1. get relative poses to the first image (input)
            2. apply a rotation to the camera poses, making the input pose is on z-axis with aligned rotation
            3. normalize the camera-object distance
        '''
        cam_poses_rel = get_relative_pose(cam_poses[0], cam_poses)

        translation = cam_poses[0][:3,3]    # [3]
        scale = torch.sqrt((translation ** 2).sum())

        canonical_pose = self._build_canonical_pose(scale)
        cam_poses_rotated = canonical_pose.unsqueeze(0) @ cam_poses_rel

        cam_poses_scaled = self._normalize_scale(cam_poses_rotated, scale)

        return cam_poses_scaled

    
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
