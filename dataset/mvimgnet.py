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
import gzip
from scipy.optimize import minimize
from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
from dataset.constant import *
from utils.geo_utils import get_relative_pose
from utils.process_utils import process_images, get_rays_from_pose
from utils import data_utils
from tsr.utils import get_spherical_cameras
from dataset.co3d import get_center_crop_from_mask
import open3d as o3d


class MVIMageNet_MV(Dataset):
    def __init__(self, config, split='test', length=1, mode='multiview', multiple_data=False, data_name='',
                 root='/data-local/hanwen/dataset/mvimgnet'):
        self.config = config
        self.split = split
        self.root = root
        self.multiple_data = multiple_data
        self.data_name = data_name
        self.mode = mode
        assert split in ['val', 'test']
        assert mode in ['multiview']

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
        self.seq_names = self.data_split['test']


    def _load_dataset(self):
        data_root = './dataset/split_info'
        os.makedirs(data_root, exist_ok=True)
        data_split_file_path = os.path.join(data_root, 'mvimgnet_mv.json')

        with open(data_split_file_path, 'r') as f:
            data_split_file = json.load(f)

        self.data_split.update(data_split_file)


    def __len__(self):
        return len(self.seq_names)
    

    def __getitem__(self, idx):
        tmp = self.seq_names[idx]
        seq_name, frame_names = tmp[0], tmp[1:]
        seq_path = os.path.join(self.root, 'image', seq_name)
        seq_path_imgs = os.path.join(seq_path, 'images')

        # load mv img and mask
        imgs, masks = [], []
        for rgb_file in frame_names:
            img, mask = self._load_frame(seq_path_imgs, rgb_file)
            img = torch.tensor(img)
            mask = torch.tensor(mask)
            imgs.append(img)
            masks.append(mask)
        imgs = torch.stack(imgs)                # [n,c,h,w], images without cropping, in range [0,1]
        masks = torch.stack(masks).float()      # [n,1,h,w], with value 0/1
        
        # crop
        bbox, _, _ = get_center_crop_from_mask(imgs[0], masks[0], self.img_size)
        imgs_crop, masks_crop = [], []
        for (img, mask) in zip(imgs, masks):
            _, img_crop, mask_crop = get_center_crop_from_mask(img, mask, self.img_size, bbox=bbox)
            imgs_crop.append(img_crop)
            masks_crop.append(mask_crop)
        imgs_crop = torch.stack(imgs_crop)     # [n,c,h,w]
        masks_crop = torch.stack(masks_crop)   # [n,1,h,w]
        imgs_crop = imgs_crop * masks_crop + (1 - masks_crop) * 1.0

        # get intrinsics
        seq_path_sparse = os.path.join(seq_path, 'sparse/0')
        cameras_path = os.path.join(seq_path_sparse, 'cameras.bin')
        assert os.path.isfile(cameras_path), '{} not exist'.format(cameras_path)
        cameras = data_utils.read_intrinsics_binary(cameras_path)
        K_all, K_all_normalized = self._get_intrinsics(cameras)   # normalized
        K, K_normalized = K_all[1], K_all_normalized[1]
        assert len(K_all.keys()) == 1
        fs = torch.tensor([K_normalized[0,0].item()]).repeat(imgs.shape[0])
        fov_train = 0.691150367
        f_train = 0.5 / math.tan(0.5 * fov_train)
        fs = (fs / fs[0] * f_train)

        # get poses
        poses_path = os.path.join(seq_path_sparse, 'images.bin')
        assert os.path.isfile(poses_path)
        meta = data_utils.read_extrinsics_binary(poses_path)
        w2c_colmap_all, w2c_colmap_dict, all_imgs_name = self._get_all_poses(meta)
        w2c_colmap_selected = [w2c_colmap_dict[it] for it in frame_names]
        w2c_colmap_selected = torch.stack(w2c_colmap_selected).float()     # [n,4,4]

        # load all masks of the sequence
        data_root = './dataset/split_info/mvimgnet_mv_center_info'
        center_info_path = os.path.join(data_root, '{}.pth'.format(idx))
        if os.path.exists(center_info_path):
            center_info = torch.load(center_info_path)
            points_fg, weights_fg = center_info['points_fg'], center_info['weights_fg']
        else:
            all_masks = []
            for img_name in all_imgs_name:
                mask = self._load_mask(seq_path_imgs, img_name)
                mask = torch.tensor(mask)
                all_masks.append(mask)
            all_masks = torch.stack(all_masks)    # [n,1,h,w], binary

            # get points of objects
            points, _, _ = data_utils.read_points3D_binary(poses_path.replace('images.bin', 'points3D.bin'))
            points = torch.tensor(points).float()[:, :3]    # [n,3]
            points_fg, weights_fg = self._get_surface_points(points, w2c_colmap_all, K, all_masks)
            torch.save({'points_fg': points_fg, 'weights_fg': weights_fg}, center_info_path)
        # center_fg = points_fg.mean(dim=0)
        weights_fg_norm = weights_fg / weights_fg.sum()
        center_fg = (points_fg * weights_fg_norm).sum(dim=0)

        # transform the poses
        c2w_colmap_selected = w2c_colmap_selected.inverse()
        c2w_cv2_selected = c2w_colmap_selected
        c2w_cv2_selected[:,:3,3] -= center_fg.unsqueeze(0)
        c2w_cv2_normalized_selected = self._canonicalize_cam_poses(c2w_cv2_selected)
        c2w_gl_normalized_selected = c2w_cv2_normalized_selected @ opengl_to_cv2.unsqueeze(0)

        rays_o, rays_d = get_rays_from_pose(c2w_gl_normalized_selected, focal=fs * self.render_size, size=self.render_size)

        input, input_mask = imgs_crop[0].float(), masks_crop[0].float()   # [c,h,w], [1,h,w]
        input = input * input_mask + (1 - input_mask) * 0.5
        
        sample = {
                'input_image': input.unsqueeze(0),             # [1,c,h,w]
                'rays_o': rays_o.float()[:],                   # [n,h,w,3]
                'rays_d': rays_d.float()[:],                   # [n,h,w,3]
                'render_images': imgs_crop.float()[:],         # [n,c,h,w]
                'render_masks': masks_crop.float()[:],         # [n,1,h,w]
            }
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
    

    def _load_mask(self, img_root, img_name):
        img_path = os.path.join(img_root, img_name)
        mask_path = img_path.replace('/images', '').replace('mvimgnet/image', 'mvimgnet/mask') + '.png'
        img_pil = Image.open(img_path)
        mask_pil = Image.open(mask_path)
        w, h = img_pil.size
        mask_pil = mask_pil.resize((w, h), Image.NEAREST)
        mask = np.asarray(mask_pil).squeeze()[:,:,np.newaxis].transpose((2,0,1))
        mask = (mask > 128)                                                         # [1,H,W]
        return mask
    

    def _get_intrinsics(self, cameras):
        # the returned intrinsics is not normalized with image resolution
        K_all, K_all_normalized = {}, {}
        for k in cameras.keys():
            camera = cameras[k]
            id = camera.id
            f, cx, cy, r = camera.params    # not normalized
            w_origin, h_origin = camera.width, camera.height
            assert camera.model == 'SIMPLE_RADIAL'
            K_normalized = torch.tensor([[f / w_origin, 0, cx / w_origin], 
                                        [0, f / h_origin, cy / h_origin], 
                                        [0, 0, 1]])
            K = torch.tensor([[f, 0, cx], 
                            [0, f, cy], 
                            [0, 0, 1]])
            K_all[id] = K
            K_all_normalized[id] = K_normalized
        return K_all, K_all_normalized
    

    def _get_all_poses(self, meta):
        all_poses, all_poses_dict, all_img_name = [], {}, []
        for k, v in meta.items():
            name = v.name
            all_img_name.append(name)
            tmp = torch.eye(4)
            tmp[:3,:3] = torch.tensor(data_utils.qvec2rotmat(v.qvec))
            tmp[:3,3] = torch.tensor(v.tvec)
            all_poses_dict[v.name] = tmp
            all_poses.append(tmp)
        all_poses = torch.stack(all_poses)
        return all_poses, all_poses_dict, all_img_name
    

    def _get_surface_points(self, points, w2c_cv2_all, K, masks_all):
        points2d = project_points_to_images(points, w2c_cv2_all, K)
        in_mask = check_points_in_masks(points2d, masks_all)
        is_fg = is_foreground(in_mask, threshold=0.8)
        points_fg, weight_fg = points[is_fg], in_mask[is_fg].float().mean(dim=1)
        return points_fg, weight_fg.unsqueeze(1)
    

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
        else:
            raise NotImplementedError
    



def project_points_to_images(point_cloud, world_to_camera_matrices, camera_intrinsics):
    """
    Projects points from a point cloud to multiple image planes.

    Parameters:
    - point_cloud: Tensor of shape [N, 3], where N is the number of points.
    - world_to_camera_matrices: Tensor of shape [K, 4, 4], representing world-to-camera transformations.
    - camera_intrinsics: Tensor of shape [3, 3], representing camera intrinsic matrices.

    Returns:
    - A list of tensors, each of shape [N, 2], representing the (x, y) locations of the projected points on each image plane.
    """
    N = point_cloud.shape[0]
    K = world_to_camera_matrices.shape[0]
    # Add a column of ones to the point cloud to handle homogeneous coordinates for transformation
    homogeneous_points = torch.cat([point_cloud, torch.ones(N, 1)], dim=1)

    projected_points_list = []

    for k in range(K):
        # Transform points to camera space
        camera_space_points = torch.mm(homogeneous_points.double(), world_to_camera_matrices[k].t().double())[:, :3]
        
        # Project points onto the image plane
        # The z-coordinate is used to normalize the x and y coordinates.
        projected_points = torch.mm(camera_space_points, camera_intrinsics.t())
        projected_points = projected_points[:, :2] / projected_points[:, 2:3]

        projected_points_list.append(projected_points)
    projected_points_list = torch.stack(projected_points_list)
    return projected_points_list


def check_points_in_masks(projected_points_list, gt_masks):
    """
    Checks if projected points fall within the ground truth segmentation masks.

    Parameters:
    - projected_points_list: List of tensors, each of shape [N, 2], representing the (x, y) locations
                              of the projected points on each image plane.
    - gt_masks: Tensor of shape [K, 1, H, W], with binary values indicating foreground.

    Returns:
    - Tensor of shape [N, K], where each element is 1 if the point falls within the foreground
      in the corresponding image, and 0 otherwise.
    """
    K = gt_masks.shape[0]  # Number of images
    N = projected_points_list[0].shape[0]  # Number of points in the point cloud

    # Initialize the result tensor
    result = torch.zeros((N, K), dtype=torch.uint8)

    for k, points in enumerate(projected_points_list):
        # Round points to nearest integer to use as indices, and clamp to image dimensions
        points = torch.round(points).long()
        H, W = gt_masks.shape[2], gt_masks.shape[3]
        points[:, 0] = torch.clamp(points[:, 0], 0, W - 1)
        points[:, 1] = torch.clamp(points[:, 1], 0, H - 1)

        # Check if each point is within the foreground in the ground truth mask
        for i, (x, y) in enumerate(points):
            result[i, k] = gt_masks[k, 0, y, x]
    return result


def is_foreground(in_mask, threshold=0.8):
    res = in_mask.float().mean(dim=1) > threshold
    return res


def distance_to_rays(point, rays_start, rays_dir):
    """
    Calculate the sum of squared distances from a point to all rays, given rays' starting points and directions.
    """
    point = np.array(point)
    vectors = rays_start - point
    distances_squared = np.sum(np.square(vectors - np.sum(vectors * rays_dir, axis=1)[:, np.newaxis] * rays_dir), axis=1)
    return np.sum(distances_squared)


def camera_optimization(camera_positions, camera_directions):
    def objective(t):
        x = camera_positions[0] + t * camera_directions[0]
        total_distance = 0
        for i in range(1, len(camera_positions)):
            pi = camera_positions[i]
            di = camera_directions[i]
            projection = pi + np.dot(x - pi, di) * di
            total_distance += np.sum((x - projection)**2)
        return total_distance
    
    result = minimize(objective, 0.0)
    optimal_point = camera_positions[0] + result.x * camera_directions[0]
    return torch.tensor(optimal_point)