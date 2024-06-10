import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from kornia.augmentation import RandomResizedCrop
import random
from tsr.utils import get_ray_directions, get_rays
import math
import torchvision
from torchvision.transforms import functional as TF

def process_images(config, imgs, masks, Ks):
    '''
    imgs: in shape [n,3,h,w]
    masks: in shape [n,1,h,w]
    K: in shape [n,3,3] (normalized)
    '''
    n, _, h, w = imgs.shape
    assert h == w
    min_size, max_size = config.dataset.img_crop_min, config.dataset.img_crop_max
    out_size = config.model.render_resolution
    max_translation = h - max_size

    imgs_render, masks_render, Ks_render = [], [], []
    for i in range(n):
        if i == 0:
            # print(imgs[i].shape, masks[i].shape, Ks[i].shape)
            imgs_render.append(F.interpolate(imgs[i].unsqueeze(0), [out_size, out_size], mode='bilinear')[0])
            masks_render.append(F.interpolate(masks[i].unsqueeze(0), [out_size, out_size], mode='nearest')[0])
            Ks_render.append(Ks[i])
        else:
            img, mask, K = imgs[i], masks[i], Ks[i]
            crop_size = random.randint(min_size, max_size)

            # x1 = torch.randint(0, h - crop_size + 1, (1,)).item()   # x for vertical axis (first dimension)
            # y1 = torch.randint(0, w - crop_size + 1, (1,)).item()

            margin_h = (h - crop_size) // 3  # Dividing by 4 restricts the starting point to be closer to the center
            margin_w = (w - crop_size) // 3
            x1 = torch.randint(margin_h, h - crop_size - margin_h + 1, (1,)).item()
            y1 = torch.randint(margin_w, w - crop_size - margin_w + 1, (1,)).item()
            
            x2, y2 = x1 + crop_size, y1 + crop_size
            img_crop = img[:, x1:x2, y1:y2]
            mask_crop = mask[:, x1:x2, y1:y2]
            img_crop = F.interpolate(img_crop.unsqueeze(0), [out_size, out_size], mode='bilinear')[0]
            mask_crop = F.interpolate(mask_crop.unsqueeze(0), [out_size, out_size], mode='nearest')[0]
            K_crop = process_intrinsics(K, x1, y1, 
                                        crop_size=crop_size, 
                                        original_size=h,
                                        out_size=out_size)

            imgs_render.append(img_crop)
            masks_render.append(mask_crop)
            Ks_render.append(K_crop)

    imgs_render = torch.stack(imgs_render)
    masks_render = torch.stack(masks_render)
    Ks_render = torch.stack(Ks_render)

    return imgs_render, masks_render, Ks_render


def process_intrinsics(K, x1, y1, crop_size, original_size, out_size):
    '''
    K: intrinsics in shape [3,3] (normalized)
    x1, y1: crop top-left coordinate in pixels, x1 is for vertical axis
    crop_size: the crop size in pixels
    original_size: the original size in pixels (assuming square images)
    out_size: the output size in pixels
    '''
    K_new = K.clone()

    norm_x1 = x1 / original_size
    norm_y1 = y1 / original_size

    K_new[0,2] = (K_new[0,2] - norm_y1) / (crop_size / original_size)
    K_new[1,2] = (K_new[1,2] - norm_x1) / (crop_size / original_size)

    K_new[0,0] = K_new[0,0] / (crop_size / original_size) #* (out_size / crop_size)
    K_new[1,1] = K_new[1,1] / (crop_size / original_size) #* (out_size / crop_size)

    return K_new


def get_rays_from_pose(c2w, focal, size):
    # c2w in shape [n,4,4]
    # focal in shape [n]
    n = c2w.shape[0]
    device = c2w.device

    directions_unit_focal = get_ray_directions(
        H=size,
        W=size,
        focal=1.0,
        ).to(device)
    directions = directions_unit_focal[None, :, :, :].repeat(n, 1, 1, 1)
    directions[:, :, :, :2] = (
        directions[:, :, :, :2] / focal[:, None, None, None]
    )   # [n,h,w,3]
    rays_o, rays_d = get_rays(directions, c2w, keepdim=True, normalize=True)    # [n,h,w,3]
    return rays_o, rays_d


def get_cameras(n_views, elevation_deg, camera_distance, sampling='uniform'):
    if sampling == 'uniform':
        assert elevation_deg == 0
        azimuth_deg = torch.linspace(0, 360.0, n_views + 1)[:n_views]
        elevation_deg = torch.full_like(azimuth_deg, elevation_deg)
    elif sampling == 'random':
        assert elevation_deg == 0
        azimuth_deg = 0 + torch.rand(n_views) * (360. - 0)
        azimuth_deg[0] = 0.0
        elevation_deg = torch.full_like(azimuth_deg, elevation_deg)
    elif sampling == 'constraint':
        assert elevation_deg == 0
        range_val = 90
        azimuth_deg = sample_angle_constraint(n_views, range_val)
        azimuth_deg[0] = 0.0
        elevation_deg = torch.full_like(azimuth_deg, elevation_deg)
    elif sampling == 'constraint_elevation':
        azimuth_deg = sample_angle_constraint(n_views, range_val=90)
        elevation_deg = sample_angle_constraint(n_views, range_val=45)
        azimuth_deg[0], elevation_deg[0] = 0., 0.
    elif sampling == 'constraint2_elevation':
        azimuth_deg = sample_angle_constraint(n_views, range_val=120)
        elevation_deg = sample_angle_constraint(n_views, range_val=45)
        azimuth_deg[0], elevation_deg[0] = 0., 0.
    else:
        raise NotImplementedError
    camera_distances = torch.full_like(elevation_deg, camera_distance)

    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180

    c2w = pose_spherical_to_mat(n_views, elevation, azimuth, camera_distances)
    return c2w


def pose_spherical_to_mat(n_views, elevation, azimuth, camera_distances):
    # convert spherical coordinates to cartesian coordinates
    # right hand coordinate system, x back, y right, z up
    # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )

    # default scene center at origin
    center = torch.zeros_like(camera_positions)
    # default camera up direction as +z
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(n_views, 1)

    lookat = F.normalize(center - camera_positions, dim=-1)
    right = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
    c2w[:, 3, 3] = 1.0
    return c2w

        
def sample_angle_constraint(n, range_val=90):
    # Generate N uniform random numbers in [0, 1]
    random_numbers = torch.rand(n)

    # Half of the numbers for [0, 90]
    half_n = n // 2
    first_half = random_numbers[:half_n] * range_val

    # The other half for [270, 360], first scale then shift
    second_half = random_numbers[half_n:] * range_val + (360. - range_val)

    # Combine and return
    combined = torch.cat((first_half, second_half), 0)
    return combined


def get_cameras_curriculum(n_views, camera_distances, min_azi, min_ele, max_azi, max_ele):
    elevation_deg = sample_angle_constraint_rand(n_views, min_ele, max_ele)
    azimuth_deg = sample_angle_constraint_rand(n_views, min_azi, max_azi)
    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180

    c2w = pose_spherical_to_mat(n_views, elevation, azimuth, camera_distances)
    return c2w


def sample_angle_constraint_rand(n, range_min, range_max):
    vals = torch.rand(n) * (range_max - range_min) + range_min
    rand_nums = (torch.rand(n) > 0.5).float() * 2 - 1.
    return vals * rand_nums
    


def get_cameras_consistency(camera_distance):
    azimuth_deg = torch.tensor([0, 30, 60, -30, -60, 30, 60, -30, -60, 30, 60, -30, -60])
    elevation_deg = torch.tensor([0, 0, 0, 0, 0, 30, 30, 30, 30, 60, 60, 60, 60])
    camera_distances = torch.full_like(elevation_deg, camera_distance)

    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180
    n_views = azimuth_deg.shape[0]

    c2w = pose_spherical_to_mat(n_views, elevation, azimuth, camera_distances)
    return c2w



def get_cameras_consistency_invert(c2w: torch.Tensor):
    # c2w in [n,4,4]
    # c2w[0] is the input view pose (identity rotation)
    return c2w[:1] @ torch.inverse(c2w) @ c2w[:1]

    # res = []
    # for pose in c2w:
    #     pose_consistency = c2w[0] @ torch.inverse(pose) @ c2w[0]
    #     res.append(pose_consistency)
    # return torch.stack(res)



class ImageMaskAug:
    def __init__(self):
        self.color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.2)

    def __call__(self, image, mask):
        # Random horizontal flip with the same decision
        if torch.rand(1) < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random rotation with the same angle
        angle = transforms.RandomRotation.get_params([-30, 30])
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)

        # Color jitter (applied to image only)
        image = self.color_jitter(image)

        return image, mask