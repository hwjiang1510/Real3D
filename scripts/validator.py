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
               epoch, output_dir, device, rank, wandb_run=None, eval=False):
    lpips_vgg = lpips.LPIPS(net="vgg").to(device)
    lpips_vgg.eval()

    clip_model, _ = clip.load("ViT-B/16", device=device)    # clip input should be normalized with ImageNet params

    model.eval()
    metrics = {'nvs_clip': [], 'nvs_lpips': [], 'nvs_fid': []}

    camera_distance = SCALE_OBJAVERSE if config.dataset.mv_data_name == 'objaverse' else SCALE_OBJAVERSE_BOTH

    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            sample = exp_utils.dict_to_cuda(sample, device)
            sample = loss_utils.get_eval_target(config, sample)

            # inference novel view & evauate semantic similarity
            with autocast(enabled=config.train.use_amp, dtype=torch.float16):
                input_image = sample['input_image'].squeeze().permute(1,2,0)    # [h,w,3]
                scene_codes = model.module.get_latent_from_img([input_image], device=device)
                render_images = model.module.render_360(scene_codes, 
                                                 n_views=30, 
                                                 camera_distance=camera_distance,
                                                 fovy_deg=40.0,
                                                 height=config.test.eval_resolution,
                                                 width=config.test.eval_resolution,
                                                 return_type="pt")
                render_images = torch.stack(render_images[0])   # [n,h,w,3]
               
                # metrics = eval_nvs(config, render_images, sample, metrics, lpips_vgg, clip_model)      

            # visualize images
            subfolder = f'val/{epoch}/{data_name}' if not eval else f'test/{epoch}/{data_name}'
            vis_results(config, input_image, render_images, output_dir, batch_idx, subfolder)

            # save images for computing FID
            subfolder = f'val_cache/{epoch}/{data_name}' if not eval else f'test_cache/{epoch}/{data_name}'
            save_results(config, sample, render_images, output_dir, batch_idx, subfolder)


    del lpips_vgg, sample, scene_codes, render_images

    # format metrics
    metrics_ = format_metrics(metrics)

    mode = 'Valid' if not eval else 'Test'
    if rank == 0 and wandb_run:
        wandb_log = {}
        for k, v in metrics_.items():
            wandb_log['{}/NVS_{}_{}'.format(mode, data_name, k)] = torch.tensor(v).mean().item()
        wandb_run.log(wandb_log)

    metrics_mean = {k: v.mean().item() for k, v in metrics_.items()}
    return metrics_mean


@torch.no_grad
def eval_nvs(config, render_images, sample, metrics, lpips_vgg, clip_model):
    # render_images: list of tensor in [h,w,c], with value range [0,1]
    preds = render_images[::config.test.eval_interval].permute(0,3,1,2)  # [n,c,h,w]
    target = sample['images_target'][0]    # [1,1,c,h,w] with value range [0,1] -> [1,c,h,w]
    
    # clip similarity
    all_imgs = torch.cat([target, preds], dim=0)    # [n+1,c,h,w] in [0,1]
    all_imgs = transform_clip(all_imgs)             # normalized
    all_features = F.normalize(clip_model.encode_image(all_imgs), dim=-1, p=2.0)    # [n+1,c]
    target_features, preds_features = all_features[:1], all_features[1:]
    target_features = target_features.repeat(preds_features.shape[0],1)
    clip_sim = torch.einsum('nc,nc->n', preds_features, target_features)    # [n]
    metrics['nvs_clip'].append(clip_sim)

    # lpips
    target_repeat = target.repeat(preds_features.shape[0],1,1,1)
    lpips = lpips_vgg(preds, target_repeat, normalize=(config.train.normalize_img==False)).mean(dim=[1,2,3])    # [n]
    metrics['nvs_lpips'].append(lpips)
    return metrics


@torch.no_grad
def vis_results(config, input, renders, output_dir, sample_idx, subfolder='val'):
    save_dir = os.path.join(output_dir, subfolder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # save input image
    image = Image.fromarray((input.detach().cpu().numpy() * 255.0).astype(np.uint8))
    image.save(os.path.join(save_dir, f'{sample_idx}.jpg'))

    # save video
    video_frames = [Image.fromarray((image.detach().cpu().numpy() * 255.0).astype(np.uint8))
                    for image in renders]
    save_video(video_frames,
               os.path.join(save_dir, f"{sample_idx}.mp4"), fps=30)
    

@torch.no_grad
def save_results(config, sample, renders, output_dir, sample_idx, subfolder='val_cache'):
    save_dir = os.path.join(output_dir, subfolder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    gt_dir = os.path.join(save_dir, 'gt')
    pred_dir = os.path.join(save_dir, 'pred')
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    # save input
    target = sample['images_target'][0]     # shape [c,h,w], value range [0,1]
    target = Image.fromarray((target.squeeze().detach().permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8))
    target.save(os.path.join(gt_dir, f'{sample_idx}.jpg'))

    # save predictions
    video_frames = [Image.fromarray((image.detach().cpu().numpy() * 255.0).astype(np.uint8))
                    for image in renders]
    for frame_idx, frame in enumerate(video_frames):
        frame.save(os.path.join(pred_dir, f'{sample_idx}_{frame_idx}.jpg'))
    
@torch.no_grad
def format_metrics(metrics):
    metrics['nvs_clip'] = torch.stack(metrics['nvs_clip'])      # [N,n]
    metrics['nvs_lpips'] = torch.stack(metrics['nvs_lpips'])    # [N,n]
    N_samples, N_views = metrics['nvs_clip'].shape

    metrics_ = {}
    for i in range(N_views):
        metrics_[f'nvs_clip_view{i}'] = metrics['nvs_clip'][:,i]
        metrics_[f'nvs_lpips_view{i}'] = metrics['nvs_lpips'][:,i]
    metrics_[f'nvs_clip_all'] = metrics['nvs_clip'].view(-1)
    metrics_[f'nvs_lpips_all'] = metrics['nvs_lpips'].view(-1)

    return metrics_

@torch.no_grad
def evaluation_fid(config, output_dir, device, world_size, local_rank, epoch, wandb_run=None, eval=False):
    subfolder = f'val_cache/{epoch}' if not eval else f'test_cache/{epoch}'
    save_path = os.path.join(output_dir, subfolder)
    gt_path = os.path.join(save_path, 'gt')
    pred_path = os.path.join(save_path, 'pred')

    from pytorch_fid.inception import InceptionV3
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_v3 = InceptionV3([block_idx], normalize_input=True).cuda(device)
    inception_v3.eval()

    gt_imgs_path = [os.path.join(gt_path, it) for it in os.listdir(gt_path)][local_rank::world_size]
    pred_imgs_path = [os.path.join(pred_path, it) for it in os.listdir(pred_path) if '_0.jpg' in it][local_rank::world_size]
    
    all_x = [[] for _ in range(world_size)]
    all_pred = [[] for _ in range(world_size)]
    total_num_img, total_num_pred = 0, 0
    with torch.no_grad():
        # get all gt images
        for img_path in gt_imgs_path:
            img = torch.tensor(np.array(Image.open(img_path)) / 255.0).to(device).float()   # [h,w,3]
            img = img.unsqueeze(0).permute(0,3,1,2) # [1,3,h,w] in range [0,1]
            feat_img = inception_v3(img)[0]
            if feat_img.size(2) != 1 or feat_img.size(3) != 1:
                feat_img = F.adaptive_avg_pool2d(feat_img, (1,1))
            feat_img = feat_img.squeeze(3).squeeze(2)
            gather_feat_img = [torch.zeros_like(feat_img) for _ in range(world_size)]
            dist.all_gather(gather_feat_img, feat_img)
            for j in range(world_size):
                all_x[j].append(gather_feat_img[j].detach().cpu())
            total_num_img += world_size

        # get all pred images
        for pred_path in pred_imgs_path:
            pred = torch.tensor(np.array(Image.open(pred_path)) / 255.0).to(device).float()   # [h,w,3]
            pred = pred.unsqueeze(0).permute(0,3,1,2) # [1,3,h,w] in range [0,1]
            feat_pred = inception_v3(pred)[0]
            if feat_pred.size(2) != 1 or feat_pred.size(3) != 1:
                feat_pred = F.adaptive_avg_pool2d(feat_pred, (1,1))
            feat_pred = feat_pred.squeeze(3).squeeze(2)
            gather_feat_pred = [torch.zeros_like(feat_pred) for _ in range(world_size)]
            dist.all_gather(gather_feat_pred, feat_pred)
            for j in range(world_size):
                all_pred[j].append(gather_feat_pred[j].detach().cpu())
            total_num_pred += world_size

    # for images
    for j in range(world_size):
        all_x[j] = torch.cat(all_x[j], dim=0).numpy()
    all_x_reorg = []
    for j in range(total_num_img):
        all_x_reorg.append(all_x[j % world_size][j // world_size])
    all_x = np.vstack(all_x_reorg)
    m2, s2 = np.mean(all_x, axis=0), np.cov(all_x, rowvar=False)

    # for preds
    for j in range(world_size):
        all_pred[j] = torch.cat(all_pred[j], dim=0).numpy()
    all_pred_reorg = []
    for j in range(total_num_pred):
        all_pred_reorg.append(all_pred[j % world_size][j // world_size])
    all_pred = np.vstack(all_pred_reorg)
    m1, s1 = np.mean(all_pred, axis=0), np.cov(all_pred, rowvar=False)

    fid_score = eval_utils.calculate_frechet_distance(m1, s1, m2, s2)

    mode = 'Valid' if not eval else 'Test'
    if local_rank == 0 and wandb_run:
        wandb_log = {}
        wandb_log['{}/NVS_{}'.format(mode, 'fid')] = fid_score
        wandb_run.log(wandb_log)

    return fid_score


@torch.no_grad
def evaluation_consistency(config, loader, model, data_name,
               epoch, output_dir, device, rank, wandb_run=None, eval=False):
    lpips_vgg = lpips.LPIPS(net="vgg").to(device)
    lpips_vgg.eval()

    clip_model, _ = clip.load("ViT-B/16", device=device)    # clip input should be normalized with ImageNet params

    model.eval()
    metrics = {'consistency_clip': [], 'consistency_lpips': [], 'consistency_psnr': [], 'consistency_ssim': []}

    camera_distance = SCALE_OBJAVERSE if config.dataset.mv_data_name == 'objaverse' else SCALE_OBJAVERSE_BOTH

    focal_length = torch.tensor([0.5 / math.tan(0.5 * 0.691150367)]).reshape(1,1)
    render_size = 512

    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            sample = exp_utils.dict_to_cuda(sample, device)
            sample = loss_utils.get_eval_target(config, sample)

            with autocast(enabled=config.train.use_amp, dtype=torch.float16):
                # inference novel views
                input_image = sample['input_image'].squeeze().permute(1,2,0)    # [h,w,3]
                scene_codes = model.module.get_latent_from_img([input_image], device=device)
                render_poses = process_utils.get_cameras_consistency(camera_distance=camera_distance)    # [n,4,4]
                n_views = render_poses.shape[0]
                rays_o, rays_d = process_utils.get_rays_from_pose(render_poses, 
                                                                  focal=focal_length * render_size, 
                                                                  size=render_size)     # [n,h,w,3]
                rays_o, rays_d = rays_o.unsqueeze(0), rays_d.unsqueeze(0)
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

                # inference cycle
                render_masks_binary = (render_masks > 0.5).float()
                input_images_consistency = render_images * render_masks_binary + (1. - render_masks_binary) * 0.5 # [n,h,w,3]
                render_poses_consistency = process_utils.get_cameras_consistency_invert(render_poses)   # [n,4,4]
                rays_o_consistency, rays_d_consistency = process_utils.get_rays_from_pose(render_poses_consistency, 
                                                                  focal=focal_length * render_size, 
                                                                  size=render_size)
                
                render_images_consistency = []
                for i in range(n_views):
                    input_image_consistency = input_images_consistency[i]
                    scene_code_consistency = model.module.get_latent_from_img([input_image_consistency], device=device)
                    render_image_consistency, render_mask_consistency = model.module.renderer(model.module.decoder,
                                                                                            scene_code_consistency[0],
                                                                                            rays_o_consistency[i].clone().to(device),
                                                                                            rays_d_consistency[i].clone().to(device),
                                                                                            return_mask=True)
                    render_images_consistency.append(render_image_consistency)
                render_images_consistency = torch.stack(render_images_consistency)

                subfolder = f'val_consistency/{epoch}/{data_name}' if not eval else f'test_consistency/{epoch}/{data_name}'
                save_results_consistency(config, sample, render_images, render_images_consistency,
                                         output_dir, batch_idx, subfolder)
                
                metrics = eval_consistency(config, render_images_consistency, sample, metrics, lpips_vgg, clip_model)
    
    del lpips_vgg, sample, scene_codes, scene_code_consistency, render_images

    # format metrics
    metrics_ = format_metrics_consistency(metrics)

    mode = 'Valid' if not eval else 'Test'
    if rank == 0 and wandb_run:
        wandb_log = {}
        for k, v in metrics_.items():
            wandb_log['{}/Consistency_{}_{}'.format(mode, data_name, k)] = torch.tensor(v).mean().item()
        wandb_run.log(wandb_log)

    metrics_mean = {k: v.mean().item() for k, v in metrics_.items()}

    return metrics_mean
                


@torch.no_grad
def save_results_consistency(config, sample, renders, render_consistency, output_dir, sample_idx, subfolder='val_consistency'):
    save_dir = os.path.join(output_dir, subfolder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    gt_dir = os.path.join(save_dir, 'gt')
    pred_dir = os.path.join(save_dir, 'pred')
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    # save input
    target = sample['render_images'][0]     # shape [c,h,w], value range [0,1]
    target = Image.fromarray((target.squeeze().detach().permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8))
    target.save(os.path.join(gt_dir, f'{sample_idx}.jpg'))

    # save predictions
    video_frames = [Image.fromarray((image.detach().cpu().numpy() * 255.0).astype(np.uint8))
                    for image in renders]
    for frame_idx, frame in enumerate(video_frames):
        frame.save(os.path.join(pred_dir, f'{sample_idx}_{frame_idx}.jpg'))

    # save predictions of inputs
    video_frames = [Image.fromarray((image.detach().cpu().numpy() * 255.0).astype(np.uint8))
                    for image in render_consistency]
    for frame_idx, frame in enumerate(video_frames):
        frame.save(os.path.join(pred_dir, f'{sample_idx}_{frame_idx}_input.jpg'))


@torch.no_grad
def eval_consistency(config, render_images, sample, metrics, lpips_vgg, clip_model):
    # render_images: list of tensor in [h,w,c], with value range [0,1]
    preds = render_images[1:].permute(0,3,1,2)      # [n,c,h,w]
    target = sample['render_images'][0]             # [1,c,h,w]

    # psnr
    psnrs, ssims = [], []
    for pred in preds:
        psnr, ssim = eval_utils.compute_img_metric(pred.permute(1,2,0).detach().cpu().numpy(), 
                                                   target.squeeze().permute(1,2,0).detach().cpu().numpy())
        psnrs.append(psnr)
        ssims.append(ssim)
    psnrs, ssims = torch.tensor(psnrs), torch.tensor(ssims)
    metrics['consistency_psnr'].append(psnrs)
    metrics['consistency_ssim'].append(ssims)
    
    # clip similarity
    all_imgs = torch.cat([target, preds], dim=0)    # [n+1,c,h,w] in [0,1]
    all_imgs = transform_clip_resize(all_imgs)             # normalized
    all_features = F.normalize(clip_model.encode_image(all_imgs), dim=-1, p=2.0)    # [n+1,c]
    target_features, preds_features = all_features[:1], all_features[1:]
    target_features = target_features.repeat(preds_features.shape[0],1)
    clip_sim = torch.einsum('nc,nc->n', preds_features, target_features)    # [n]
    metrics['consistency_clip'].append(clip_sim)

    # lpips
    target_repeat = target.repeat(preds_features.shape[0],1,1,1)
    lpips = lpips_vgg(preds, target_repeat, normalize=(config.train.normalize_img==False)).mean(dim=[1,2,3])    # [n]
    metrics['consistency_lpips'].append(lpips)

    return metrics


@torch.no_grad
def format_metrics_consistency(metrics):
    metrics['consistency_psnr'] = torch.stack(metrics['consistency_psnr'])  # [N,n]
    metrics['consistency_ssim'] = torch.stack(metrics['consistency_ssim'])
    metrics['consistency_clip'] = torch.stack(metrics['consistency_clip'])      
    metrics['consistency_lpips'] = torch.stack(metrics['consistency_lpips'])
    N_samples, N_views = metrics['consistency_clip'].shape

    metrics_ = {}
    for i in range(N_views):
        metrics_[f'consistency_psnr_view{i}'] = metrics['consistency_psnr'][:,i]
        metrics_[f'consistency_ssim_view{i}'] = metrics['consistency_ssim'][:,i]
        metrics_[f'consistency_clip_view{i}'] = metrics['consistency_clip'][:,i]
        metrics_[f'consistency_lpips_view{i}'] = metrics['consistency_lpips'][:,i]
    metrics_[f'consistency_psnr_all'] = metrics['consistency_psnr'].view(-1)
    metrics_[f'consistency_ssim_all'] = metrics['consistency_ssim'].view(-1)
    metrics_[f'consistency_clip_all'] = metrics['consistency_clip'].view(-1)
    metrics_[f'consistency_lpips_all'] = metrics['consistency_lpips'].view(-1)

    return metrics_
                    


               
                






    
