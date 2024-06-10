import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
import random
import os
import math
import dataclasses
import torch.distributed as dist
import itertools
from einops import rearrange
from utils import exp_utils, loss_utils, vis_utils, eval_utils, process_utils
import lpips
import clip
from torch.cuda.amp import autocast
from utils.vis_utils import transform
from dataset.constant import SCALE_OBJAVERSE,SCALE_OBJAVERSE_BOTH
import torchvision.transforms as transforms
from PIL import Image
from tsr.utils import save_video


logger = logging.getLogger(__name__)


transform_clip = transforms.Compose([
    transforms.Normalize(mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]), 
                         std=torch.tensor([0.26862954, 0.26130258, 0.27577711])),
    ])

transform_clip_resize = transforms.Compose([
    transforms.Normalize(mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]), 
                         std=torch.tensor([0.26862954, 0.26130258, 0.27577711])),
    transforms.Resize((224, 224))
    ])


@torch.no_grad
def evaluation(config, loader, model, data_name,
               epoch, output_dir, device, rank, wandb_run=None, eval=False, vis360=False):
    lpips_vgg = lpips.LPIPS(net="vgg").to(device)
    lpips_vgg.eval()

    model.eval()
    metrics = {'lpips': [], 'psnr': [], 'ssim': []}

    assert config.test.batch_size == 1
    if config.dataset.mv_data_name == 'objaverse':
        camera_distance = SCALE_OBJAVERSE
    elif config.dataset.mv_data_name == 'objaverse_both':
        camera_distance = SCALE_OBJAVERSE_BOTH
    elif config.dataset.mv_data_name == 'none': # raw triposr
        camera_distance = 1.8
    else:
        raise NotImplementedError
    subfolder = f'val_cache_mv/{epoch}/{data_name}' if not eval else f'test_cache_mv/{epoch}/{data_name}'

    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            # print(batch_idx)
            sample = exp_utils.dict_to_cuda(sample, device)
            sample = loss_utils.get_eval_target(config, sample)

            # inference novel view & evauate semantic similarity
            with autocast(enabled=config.train.use_amp, dtype=torch.float16):
                input_image = sample['input_image'].squeeze().permute(1,2,0)    # [h,w,3]
                scene_codes = model.module.get_latent_from_img([input_image], device=device)
                rays_o, rays_d = sample['rays_o'], sample['rays_d']             # [1,n,h,w,3]
                render_images, render_masks = [], []
                for cur_rays_o, cur_rays_d in zip(rays_o, rays_d):
                    image, mask = model.module.renderer(model.module.decoder,
                                                        scene_codes[0],
                                                        cur_rays_o.clone().to(device), 
                                                        cur_rays_d.clone().to(device),
                                                        return_mask=True)
                    render_images.append(image)
                    render_masks.append(mask)
                render_images = torch.stack(render_images).squeeze(0)  # [n,h,w,3]
                render_masks = torch.stack(render_masks).squeeze(0)

                # render 360
                if vis360:
                    render_images_360 = model.module.render_360(scene_codes, 
                                                    n_views=30, 
                                                    camera_distance=camera_distance,
                                                    fovy_deg=40.0,
                                                    height=config.test.eval_resolution,
                                                    width=config.test.eval_resolution,
                                                    return_type="pt")
                    render_images_360 = torch.stack(render_images_360[0])   # [n,h,w,3]
                    visualize_360(render_images_360, output_dir, batch_idx, subfolder)
               
                metrics = eval_nvs(config, render_images, sample, metrics, lpips_vgg)      

            # save images for computing FID
            save_results(config, sample, render_images, output_dir, batch_idx, subfolder)


    del lpips_vgg, sample, scene_codes, render_images

    # format metrics
    metrics_ = format_metrics(metrics)

    mode = 'Valid' if not eval else 'Test'
    if rank == 0 and wandb_run:
        wandb_log = {}
        for k, v in metrics_.items():
            wandb_log['{}/mv_{}_{}'.format(mode, data_name, k)] = torch.tensor(v).mean().item()
        wandb_run.log(wandb_log)

    metrics_mean = {f'{data_name}_{k}': v.mean().item() for k, v in metrics_.items()}
    return metrics_mean


@torch.no_grad
def eval_nvs(config, render_images, sample, metrics, lpips_vgg):
    # render_images: list of tensor in [h,w,c], with value range [0,1]
    preds = render_images.permute(0,3,1,2)  # [n,c,h,w]
    targets = sample['images_target'][0]     # [n,c,h,w]

    # psnr
    psnrs, ssims = [], []
    for (pred, target) in zip(preds, targets):
        psnr, ssim = eval_utils.compute_img_metric(pred.permute(1,2,0).detach().cpu().numpy(), 
                                                   target.squeeze().permute(1,2,0).detach().cpu().numpy())
        psnrs.append(psnr)
        ssims.append(ssim)
    psnrs, ssims = torch.tensor(psnrs), torch.tensor(ssims)
    metrics['psnr'].append(psnrs)
    metrics['ssim'].append(ssims)

    # lpips
    lpips = lpips_vgg(preds, targets, normalize=(config.train.normalize_img==False)).mean(dim=[1,2,3])    # [n]
    metrics['lpips'].append(lpips)
    return metrics


@torch.no_grad
def save_results(config, sample, renders, output_dir, sample_idx, subfolder='val_cache'):
    save_dir = os.path.join(output_dir, subfolder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    pred_dir = os.path.join(save_dir, 'pred')
    os.makedirs(pred_dir, exist_ok=True)

    # save input
    target = sample['images_target'][0]     # shape [n,c,h,w], value range [0,1]
    target = [Image.fromarray((image.squeeze().detach().permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8))
              for image in target]
    video_frames = [Image.fromarray((image.detach().cpu().numpy() * 255.0).astype(np.uint8))
                    for image in renders]
    for frame_idx, (frame, cur_target) in enumerate(zip(target, video_frames)):
        save_frame = np.vstack([np.asarray(frame), np.asarray(cur_target)])
        save_frame = Image.fromarray(save_frame)
        save_frame.save(os.path.join(pred_dir, f'{sample_idx}_{frame_idx}.jpg'))


@torch.no_grad
def format_metrics(metrics):
    metrics['lpips'] = torch.stack(metrics['lpips'])      # [N,n]
    metrics['ssim'] = torch.stack(metrics['ssim'])    # [N,n]
    metrics['psnr'] = torch.stack(metrics['psnr'])    # [N,n]
    N_samples, N_views = metrics['lpips'].shape

    return metrics


@torch.no_grad()
def visualize_360(render_images_360, output_dir, batch_idx, subfolder):
    video_frames = [Image.fromarray((image.detach().cpu().numpy() * 255.0).astype(np.uint8))
                        for image in render_images_360]
    save_dir = os.path.join(output_dir, subfolder, 'pred')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_video(video_frames,
            os.path.join(save_dir, f"{batch_idx}.mp4"), fps=30)