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
from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
from dataset.constant import *
from utils.geo_utils import get_relative_pose
from utils.process_utils import process_images, get_rays_from_pose
from tsr.utils import get_spherical_cameras
import open3d as o3d


class CO3D(Dataset):
    def __init__(self, config, split='test', length=1, mode='multiview', multiple_data=False, data_name='',
                 root='/data/dataset/co3d_mv', root_img='/data-local/hanwen/repo/segment-anything/co3d_mv'):
        self.config = config
        self.split = split
        self.root = root
        self.root_img = root_img
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
        data_split_file_path = os.path.join(data_root, 'co3d_mv.json')

        with open(data_split_file_path, 'r') as f:
            data_split_file = json.load(f)

        self.data_split.update(data_split_file)


    def __len__(self):
        return len(self.seq_names)
    

    def __getitem__(self, idx):
        tmp = self.seq_names[idx]
        cat_seq, frame_names = tmp[0], tmp[1:]
        frame_names = ['frame{:06d}.jpg'.format(int(it.split('.')[0].replace('frame', ""))) for it in frame_names]
        category, sequence_name = cat_seq.split('/')
        meta_path = os.path.join(self.root, category, 'frame_annotations.jgz')
        file = gzip.open(meta_path, 'rt')
        info = json.loads(file.read())
        meta = {}
        for it in info:
            if it['sequence_name'] == sequence_name and it['image']['path'].split('/')[-1] in frame_names:
                meta[it['image']['path'].split('/')[-1]] = it

        # load images and masks
        rgb_files = [os.path.join(self.root, cat_seq, 'images', frame_name) for frame_name in frame_names]
        imgs, masks, masks_float = [], [], []
        for rgb_file in rgb_files:
            img, mask, mask_float = self._load_frame(rgb_file)
            img = torch.tensor(img)
            mask = torch.tensor(mask)
            mask_float = torch.tensor(mask_float)
            imgs.append(img)
            masks.append(mask)
            masks_float.append(mask_float)
        imgs = torch.stack(imgs).float()     # [n,c,h,w]
        masks = torch.stack(masks).float()   # [n,1,h,w]
        masks_float = torch.stack(masks_float).float()   # [n,1,h,w]

        # crop images and masks
        bbox, _, _ = get_center_crop_from_mask(imgs[0], masks[0], self.img_size)    # input view
        imgs_crop, masks_crop = [], []
        for (img, mask) in zip(imgs, masks_float):
            _, img_crop, mask_crop = get_center_crop_from_mask(img, mask, self.img_size, bbox=bbox)
            imgs_crop.append(img_crop)
            masks_crop.append(mask_crop)
        imgs_crop = torch.stack(imgs_crop)     # [n,c,h,w]
        masks_crop = torch.stack(masks_crop)   # [n,1,h,w]
        imgs_crop = imgs_crop * masks_crop + (1 - masks_crop) * 1.0

        # get intrinsics
        Ks, fs = [], []
        for frame_name in frame_names:
            K, f = co3d_annotation_to_opencv_intrinsics(meta[frame_name], imgs.shape[-2], imgs.shape[-1])
            Ks.append(K)
            fs.append(f)
        fs = torch.tensor(fs)
        fov_train = 0.691150367
        f_train = 0.5 / math.tan(0.5 * fov_train)
        fs = (fs / fs[0] * f_train)#.unsqueeze(1)

        # get poses
        w2c_cv2_all = []
        for frame_name in frame_names:
            w2c_cv2 = co3d_pose_to_opencv_w2c(meta[frame_name])
            w2c_cv2_all.append(w2c_cv2)
        w2c_cv2_all = torch.stack(w2c_cv2_all) # [n,4,4]

        point_cloud_path = os.path.join(self.root, cat_seq, 'pointcloud.ply')
        point_cloud = o3d.io.read_point_cloud(point_cloud_path)
        points = torch.from_numpy(np.asarray(point_cloud.points)).float()
        pc_center = points.mean(dim=0)

        # w2c_cv2_all[:,:3,3] -= pc_center.unsqueeze(0)
        c2w_cv2_all = w2c_cv2_all.inverse()
        c2w_cv2_all[:,:3,3] -= pc_center.unsqueeze(0)
        c2w_cv2_normalized_all = self._canonicalize_cam_poses(c2w_cv2_all)
        c2w_gl_normalized_all = c2w_cv2_normalized_all @ opengl_to_cv2.unsqueeze(0)

        #print(c2w_gl_normalized_all.shape, fs.shape)
        rays_o, rays_d = get_rays_from_pose(c2w_gl_normalized_all, focal=fs * self.render_size, size=self.render_size)

        input, input_mask = imgs_crop[0].float(), masks_crop[0].float()   # [c,h,w], [1,h,w]
        input = input * input_mask + (1 - input_mask) * 0.5
        
        sample = {
                'input_image': input.unsqueeze(0),              # [1,c,h,w]
                'rays_o': rays_o.float()[:],                   # [n,h,w,3]
                'rays_d': rays_d.float()[:],                   # [n,h,w,3]
                'render_images': imgs_crop.float()[:],         # [n,c,h,w]
                'render_masks': masks_crop.float()[:],         # [n,1,h,w]
            }
        return sample


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
        

    def _load_frame(self, rgb_file):
        mask_path = rgb_file.replace('images', 'masks').replace('.jpg', '.png').replace(self.root, self.root_img)
        img_pil = Image.open(rgb_file)
        mask_pil = Image.open(mask_path)

        # if self.config.dataset.white_bkg:
        #     # white background
        #     mask_255 = mask_pil.point(lambda p: p * 255)
        #     white_background = Image.new('RGB', img_pil.size, (255, 255, 255))
        #     img_pil = Image.composite(img_pil, white_background, mask_255.convert('L'))
        # else:
        #     # black background
        #     img_pil = Image.fromarray(img_np[:,:,:3])

        img_np = np.asarray(img_pil).transpose((2,0,1)) / 255.0                                 # [3,H,W], in range [0,1]
        mask_np = (np.asarray(mask_pil)[:,:,np.newaxis].transpose((2,0,1)) > 180)               # [1,H,W], in range [0,1]
        mask_np_float = np.asarray(mask_pil)[:,:,np.newaxis].transpose((2,0,1)) / 255.0

        # if not self.config.dataset.white_bkg:
        #     img_np *= mask_np
        
        if self.config.train.normalize_img:
            normalization = transforms.Compose([
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250])),
            ])
            img_np = torch.from_numpy(img_np)
            img_np = normalization(img_np).numpy()

        return img_np, mask_np, mask_np_float




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

def center_box(bbox, h, w):
    x1, y1, x2, y2 = bbox
    Cx = w / 2
    Cy = h / 2
    
    # Calculate distances from the image center to each corner of the original bounding box
    max_dist_x = max(abs(Cx - x1), abs(Cx - x2))
    max_dist_y = max(abs(Cy - y1), abs(Cy - y2))
    
    # Use the larger of the two distances as the radius of the square
    R = max(max_dist_x, max_dist_y)
    S = 2 * R  # Full side length of the square
    
    # Calculate the new bounding box centered at the image center
    new_x1 = Cx - R
    new_y1 = Cy - R
    new_x2 = Cx + R
    new_y2 = Cy + R
    return np.array([new_x1, new_y1, new_x2, new_y2]).astype(np.int32)

def square_bbox(bbox):
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    size = max(width, height)
    center = (bbox[:2] + bbox[2:]) / 2
    leftTop_new, rightBottom_new = center - size // 2, center + size // 2
    bbox_square = np.concatenate([leftTop_new, rightBottom_new]).astype(np.int32)
    return bbox_square

def paste_crop(crop, target, start_x, start_y):
    H, W = crop.shape[-2:]
    target[:, start_y: start_y+H, start_x: start_x+W].copy_(crop)
    return target

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

def get_center_crop_from_mask(img, mask, target_res=512, expand_ratio=1.7, bbox=None):
    # img in shape [3,h,w], mask in [1,h,w]
    C, H, W = img.shape
    if bbox is None:
        bbox = find_bounding_box(mask.numpy())
        bbox = center_box(bbox, H, W)
        #bbox = square_bbox(bbox)
        bbox = expand_bbox(bbox, expand_rate=1.6)
    x1, y1, x2, y2 = bbox
    valid_x1, valid_y1 = max(int(x1), 0), max(int(y1), 0)
    valid_x2, valid_y2 = min(int(x2), W), min(int(y2), H)
    crop_img = img[:, valid_y1:valid_y2, valid_x1:valid_x2]
    crop_mask = mask[:, valid_y1:valid_y2, valid_x1:valid_x2]

    target_img = torch.full((C, y2-y1, x2-x1), 255, dtype=img.dtype)
    target_mask = torch.zeros((1, y2-y1, x2-x1), dtype=mask.dtype)

    target_start_x, target_start_y = max(-int(x1), 0), max(-int(y1), 0)
    target_img = paste_crop(crop_img, target_img, target_start_x, target_start_y)
    target_mask = paste_crop(crop_mask, target_mask, target_start_x, target_start_y)

    target_img = F.interpolate(target_img.unsqueeze(0), 
                                    size=(target_res, target_res),
                                    mode='bilinear', align_corners=False)[0]
    target_mask = F.interpolate(target_mask.unsqueeze(0), 
                                    size=(target_res, target_res),
                                    mode='nearest')[0]
    
    return bbox, target_img, target_mask

def co3d_annotation_to_opencv_intrinsics(entry, h, w):  # https://github.com/facebookresearch/co3d/issues/4
    p = entry['viewpoint']['principal_point']
    f = entry['viewpoint']['focal_length']
    K = np.eye(3)
    s = min(w, h)
    K[0, 0] = f[0] * s / 2
    K[1, 1] = f[1] * s / 2
    K[0, 2] = -p[0] * s + w / 2
    K[1, 2] = -p[1] * s + h / 2
    return K, f[0] / 2

def co3d_pose_to_opencv_w2c(entry):
    R = np.asarray(entry['viewpoint']['R']).T
    t = entry['viewpoint']['T']
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt = torch.tensor(Rt).float()
    Rt = (Rt.inverse() @ torch.tensor([[-1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]).float()).inverse()
    return Rt