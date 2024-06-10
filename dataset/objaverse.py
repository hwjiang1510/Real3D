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
import torch.nn.functional as F
from torchvision.transforms import functional as func_transforms
from torchvision import transforms
import torchvision
from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
from dataset.constant import *
from utils.geo_utils import get_relative_pose
from utils.process_utils import process_images, get_rays_from_pose
from tsr.utils import get_spherical_cameras


'''
Objaverse (Gobjaverse) is rendered with 38 fixed camera poses:
    - 24 on orbit with distance 1.6450
    - 2 for top-down / bottom up with distance 1.6450
    - 12 on orbit with distance 1.9547
And object is normalized in range [-0.5,0.5]
It has 260k training instances, 500 test instances
'''
SCALE_RAW = 0.5

class Objaverse(Dataset):
    def __init__(self, config, split='train', multiple_data=False, data_name='',
                 root='/data-local/hanwen/objaverse'):
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
        data_split_file_path = os.path.join(data_root, 'objaverse.json')

        if not os.path.exists(data_split_file_path):
            self._split_data(data_split_file_path)

        with open(data_split_file_path, 'r') as f:
            data_split_file = json.load(f)

        print('GObjaverse dataset instances: train {}, test {}'.format(len(data_split_file['train']),
                                                                         len(data_split_file['test'])))
        self.data_split.update(data_split_file)


    def _split_data(self, data_split_file_path):
        all_info = {'train': [], 'test': []}
        all_instances_return = []
        all_categories = os.listdir(self.root)

        for category in all_categories:
            category_path = os.path.join(self.root, category)
            all_instances = os.listdir(category_path)
            all_instances_valid = []
            for instance in all_instances:
                instance_path = os.path.join(category_path, instance)
                tmp = os.listdir(instance_path)[0]
                instance_data_path = os.path.join(instance_path, tmp)
                add_flag = True
                if len(os.listdir(instance_data_path)) != 40:
                    add_flag = False
                else:
                    for img_dir in os.listdir(instance_data_path):
                        img_dir_path = os.path.join(instance_data_path, img_dir)
                        if f'{img_dir}.png' not in os.listdir(img_dir_path):
                            add_flag = False
                            break
                if add_flag:
                    all_instances_valid.append(os.path.join(category, instance, tmp))
            all_instances_return += all_instances_valid
        
        num_instances = len(all_instances_return)
        num_instances_test = 1000
        # save the instance names for train and test
        random.shuffle(all_instances_return)
        all_info['train'] += all_instances_return[:num_instances - num_instances_test]
        all_info['test'] += all_instances_return[num_instances - num_instances_test:]

        with open(data_split_file_path, 'w') as f:
            json.dump(all_info, f)


    def __len__(self):
        return len(self.seq_names)
    

    def __getitem__(self, idx):
        seq_name = self.seq_names[idx]
        seq_path = os.path.join(self.root, seq_name)

        # get image names
        img_dirs = os.listdir(seq_path)
        img_dirs = sorted(img_dirs)
        len_seq = len(img_dirs)

        # chosen_img_dirs = self._choose_imgs(img_dirs)
        if self.split == 'train':
            chosen_img_dirs = random.choices(img_dirs[:25], k=1)
            img_dirs = list(set(img_dirs) - set(chosen_img_dirs))
            chosen_img_dirs += random.choices(img_dirs, k=(self.num_frame-1))
        else:
            chosen_img_dirs = img_dirs[:self.num_frame]
        
        # get images, intrinsics, poses
        imgs, masks = [], []
        Ks, cam_poses_cv2 = [], []
        for img_dir in chosen_img_dirs:
            img, mask = self._load_frame(seq_path, img_dir)
            pose, K = self._load_pose(seq_path, img_dir)
            img = torch.tensor(img)
            mask = torch.tensor(mask)
            imgs.append(img)
            masks.append(mask)
            Ks.append(K)
            cam_poses_cv2.append(pose)
        imgs = torch.stack(imgs)     # [n,c,h,w]
        masks = torch.stack(masks)   # [n,1,h,w]
        Ks = torch.stack(Ks)         # [n,3,3]
        cam_poses_cv2 = torch.stack(cam_poses_cv2)  # [n,4,4], actually extrinsics (c2w)
        
        # normalize camera poses
        cam_poses_normalized_cv2 = self._canonicalize_cam_poses(cam_poses_cv2)
        cam_poses_normalized_opengl = cam_poses_normalized_cv2 @ cv2_to_opengl.unsqueeze(0)   # [n,4,4]

        # get input
        input, input_mask = imgs[0].float(), masks[0].float()   # [c,h,w], [1,h,w]
        input = input * input_mask + (1 - input_mask) * 0.5 #torch.tensor([0.485, 0.456, 0.406])[:,None,None]

        # get rays
        rays_o, rays_d = get_rays_from_pose(cam_poses_normalized_opengl, focal=Ks[:,0,0], size=self.render_size)
        # print(rays_o[0,0,0], Ks[0,0,0])

        # c2w = get_cameras(self.num_frame, 0, 1.9)
        # # print(c2w[0])
        # rays_o, rays_d = get_rays_from_pose(c2w, focal=Ks[:,0,0], size=self.render_size)

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
        file_path = os.path.join(seq_path, img_name, img_name + '.png')
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
    

    def _load_pose(self, seq_path, img_name):
        file_path = os.path.join(seq_path, img_name, img_name + '.json')
        with open(file_path, 'r') as f:
            meta = json.load(f)
        # bbox / scale
        # bbox = np.array(meta['bbox'])
        # scale = np.max(np.abs(bbox[0]))
        # # scale_factor = 0.65 / scale
        # scale_factor = 0.6 / scale
        # pose
        camera_matrix = np.eye(4)
        camera_matrix[:3, 0] = np.array(meta['x'])
        camera_matrix[:3, 1] = np.array(meta['y'])
        camera_matrix[:3, 2] = np.array(meta['z'])
        camera_matrix[:3, 3] = np.array(meta['origin']) #* scale_factor
        # camera_matrix[:3, :3] = -camera_matrix[:3, :3]
        camera_matrix = torch.tensor(camera_matrix).float()
        # instrinsics
        fx, fy = 0.5 / math.tan(0.5 * meta['x_fov']), 0.5 / math.tan(0.5 * meta['y_fov'])
        K = torch.tensor([[fx * self.render_size, 0., 0.5 * self.render_size],
                          [0., fy * self.render_size, 0.5 * self.render_size],
                          [0., 0., 1.]]).float()
        return camera_matrix, K


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
            # print('gobjaverse normalizing by object scale, input view translation', distance * scale_factor)
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