import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import train_utils
from einops import rearrange
from scripts.validator import transform_clip

def get_loss_target(config, sample):
    '''
    For training only
    '''
    if sample['render_images'].shape[-1] != config.model.render_resolution:
        imgs, masks = sample['render_images'], sample['render_masks']
        b,n,c,h,w = imgs.shape
        imgs = rearrange(imgs, 'b n c h w -> (b n) c h w')
        masks = rearrange(masks, 'b n c h w -> (b n) c h w')
        imgs_small = F.interpolate(imgs, 
                                   size=(config.model.render_resolution, config.model.render_resolution), 
                                   mode='bilinear', align_corners=False)
        masks_small = F.interpolate(masks, 
                                   size=(config.model.render_resolution, config.model.render_resolution), 
                                   mode='nearest')
        sample['images_target'] = rearrange(imgs_small, '(b n) c h w -> b n c h w', b=b, n=n)
        sample['masks_target'] = rearrange(masks_small, '(b n) c h w -> b n c h w', b=b, n=n)
    else:
        sample['images_target'] = sample['render_images']
        sample['masks_target'] = sample['render_masks']
    return sample


def get_loss_target_consistency(config, sample):
    '''
    For training with self-consistency only
    The sample dictionary contains:
        --- For normal single view training ---
        'input_image': [b,1,c,h,w]
        'rays_o' and 'rays_d': [b,n,h,w,3], n=0 is for the input view
        'render_images' and 'render_masks': [b,1,c,h,w], n=0 is for the input view
        --- For normal single view training ---
        'rays_o_consistency' and 'rays_d_consistency': [b,1,h,w,3], for the single input view
    '''
    # format the data used for single-view prediction
    if sample['render_images'].shape[-1] != config.model.render_resolution:
        imgs, masks = sample['render_images'], sample['render_masks']
        b,n,c,h,w = imgs.shape
        imgs = rearrange(imgs, 'b n c h w -> (b n) c h w')
        masks = rearrange(masks, 'b n c h w -> (b n) c h w')
        imgs_small = F.interpolate(imgs, 
                                   size=(config.model.render_resolution, config.model.render_resolution), 
                                   mode='bilinear', align_corners=False)
        masks_small = F.interpolate(masks, 
                                   size=(config.model.render_resolution, config.model.render_resolution), 
                                   mode='nearest')
        sample['images_target'] = rearrange(imgs_small, '(b n) c h w -> b n c h w', b=b, n=n)
        sample['masks_target'] = rearrange(masks_small, '(b n) c h w -> b n c h w', b=b, n=n)
    else:
        sample['images_target'] = sample['render_images']
        sample['masks_target'] = sample['render_masks']

    # format the data for single-view self-consistency
    rays_o_consistency, rays_d_consistency = sample['rays_o_consistency'], sample['rays_d_consistency']
    sample_consistency = {
        'input_image': None,
        'rays_o': rays_o_consistency,                       # [b,n=1,h,w,3]
        'rays_d': rays_d_consistency,                       # [b,n=1,h,w,3]
    }
    return sample, sample_consistency


def get_eval_target(config, sample):
    '''
    For testing only
    '''
    if sample['render_images'].shape[-1] != config.test.eval_resolution:
        imgs, masks = sample['render_images'], sample['render_masks']
        b,n,c,h,w = imgs.shape
        imgs = rearrange(imgs, 'b n c h w -> (b n) c h w')
        masks = rearrange(masks, 'b n c h w -> (b n) c h w')
        imgs_small = F.interpolate(imgs, 
                                   size=(config.test.eval_resolution, config.test.eval_resolution), 
                                   mode='bilinear', align_corners=False)
        masks_small = F.interpolate(masks, 
                                   size=(config.test.eval_resolution, config.test.eval_resolution), 
                                   mode='nearest')
        sample['images_target'] = rearrange(imgs_small, '(b n) c h w -> b n c h w', b=b, n=n)
        sample['masks_target'] = rearrange(masks_small, '(b n) c h w -> b n c h w', b=b, n=n)
    else:
        sample['images_target'] = sample['render_images']
        sample['masks_target'] = sample['render_masks']
    return sample


def get_losses(config, pred, sample, perceptural_loss):
    losses = {}

    render_imgs, render_masks = pred['images_rgb'], pred['images_weight']
    imgs, masks = sample['images_target'], sample['masks_target']
    assert render_imgs.shape == imgs.shape
    
    b,n,c,h,w = render_imgs.shape
    loss_rgb = F.mse_loss(render_imgs, imgs, reduction='none').mean(dim=[2,3,4]).mean(dim=1).mean()
    # loss_rgb = loss_rgb.reshape(b,n).mean(dim=1).mean()
    loss_perceptual = perceptural_loss(render_imgs.reshape(-1,c,h,w), 
                                       imgs.reshape(-1,c,h,w), 
                                       normalize=(config.train.normalize_img==False)).mean(dim=[1,2,3])
    loss_perceptual = loss_perceptual.reshape(b,n).mean(dim=1).mean()
    losses.update({
        'loss_rgb': loss_rgb,
        'loss_perceptual': loss_perceptual,
        'weight_rgb': config.loss.weight_render_rgb,
        'weight_perceptual': config.loss.weight_perceptual
    })
   
    if config.loss.weight_render_mask:
        loss_mask = F.mse_loss(render_masks, masks, reduction='none').mean(dim=[2,3,4]).mean(dim=1).mean()
        losses.update({
            'loss_mask': loss_mask,
            'weight_mask': config.loss.weight_render_mask
        })
    return losses


def get_losses_sv(config, pred, sample, perceptural_loss, clip_model=None):
    losses = {}

    render_imgs, render_masks = pred['images_rgb'], pred['images_weight']   # [b,n,c,h,w]
    imgs, masks = sample['images_target'], sample['masks_target']           # [b,1,c,h,w]
    assert render_imgs.ndim == imgs.ndim
    b,n,c,h,w = render_imgs.shape

    render_imgs_input, render_masks_input = render_imgs[:,:1], render_masks[:,:1]
    render_imgs_novel, render_masks_novel = render_imgs[:,1:], render_masks[:,1:]

    # calculate losses on input view
    loss_rgb = F.mse_loss(render_imgs_input, imgs, reduction='none').mean(dim=[2,3,4]).mean(dim=1).mean()
    # loss_perceptual = perceptural_loss(render_imgs_input.reshape(-1,c,h,w), 
    #                                    imgs.reshape(-1,c,h,w), 
    #                                    normalize=(config.train.normalize_img==False)).mean(dim=[1,2,3])
    # loss_perceptual = loss_perceptual.reshape(b,-1).mean(dim=1).mean()
    losses.update({
        'loss_rgb_sv': loss_rgb,
        # 'loss_perceptual_sv': loss_perceptual,
        'weight_rgb_sv': config.loss.weight_render_rgb * 0.3,
        # 'weight_perceptual_sv': config.loss.weight_perceptual
    })
    if config.loss.weight_render_mask:
        loss_mask = F.mse_loss(render_masks_input, masks, reduction='none').mean(dim=[2,3,4]).mean(dim=1).mean()
        losses.update({
            'loss_mask_sv': loss_mask,
            'weight_mask_sv': config.loss.weight_render_mask * 0.3
        })

    # calculate losses on novel views
    if n > 1:
        # lpips loss
        # loss_perceptual_novel = perceptural_loss(render_imgs_novel.reshape(-1,c,h,w), 
        #                                          imgs.repeat(1,n-1,1,1,1).reshape(-1,c,h,w), 
        #                                          normalize=(config.train.normalize_img==False)).mean(dim=[1,2,3])
        # loss_perceptual_novel = loss_perceptual_novel.reshape(b,-1).mean(dim=1).mean()
        # losses.update({
        #     'loss_perceptual_novel': loss_perceptual_novel,
        #     'weight_perceptual_novel': config.loss.weight_perceptual #* 0.1
        # })

        # clip loss
        clip_imgs = torch.cat([render_imgs_novel.reshape(b*(n-1),c,h,w), 
                               imgs.reshape(b,c,h,w)], dim=0)
        clip_imgs = F.interpolate(clip_imgs, 
                                  size=(224, 224), 
                                  mode='bilinear', align_corners=False)
        clip_imgs = transform_clip(clip_imgs)
        clip_feat = F.normalize(clip_model.encode_image(clip_imgs), dim=-1, p=2.0)
        clip_feat_novel = clip_feat[:b*(n-1)]
        clip_feat_input = clip_feat[b*(n-1):].reshape(b,1,-1).repeat(1,n-1,1).reshape(b*(n-1),-1)
        clip_loss = -1.0 * torch.einsum('bc,bc->b', clip_feat_novel, clip_feat_input).reshape(b,n-1)#.mean()
        clip_loss = torch.max(clip_loss, dim=-1)[0].mean()
        # clip_loss = clip_loss.mean()
        losses.update({
            'loss_clip_novel': clip_loss,
            'weight_clip_novel': 1.0 #0.1, #1.0
        })
    return losses


def format_consistency_input_output(config, sample_sv_consistency, sample_sv, results_sv, results_sv_hres):
    '''
    results_sv:
        'images_rgb' and 'images_weight': predicted color and mask in [b,n,c,h,w]
        we use the last image as the image for self-consistency prediction (hard coded)
    results_sv_hres:
        results rendered with higher resolution
    '''
    # get input view (last rendered view)
    if config.train.rerender_consistency_input:
        rgb = results_sv_hres['images_rgb'][:,-1:].detach()                      # [b,1,c,h,w], value [0,1]
        mask = (results_sv_hres['images_weight'][:,-1:].detach() > 0.5).float()
    else:
        rgb = results_sv['images_rgb'][:,-1:].detach()                      # [b,1,c,h,w], value [0,1]
        mask = (results_sv['images_weight'][:,-1:].detach() > 0.5).float()
    rgb = rgb * mask + (1. - mask) * 0.5
    b,n,c,h,w = rgb.shape
    rgb = rearrange(rgb, 'b n c h w -> (b n) c h w')
    if h != config.dataset.img_size or w != config.dataset.img_size:
        rgb = F.interpolate(rgb, size=(config.dataset.img_size, config.dataset.img_size), mode='bilinear', align_corners=False)
    rgb = rearrange(rgb, '(b n) c h w -> b n c h w', b=b, n=n)
    sample_sv_consistency['input_image'] = rgb

    # get loss target
    num_frame_consistency = config.train.num_frame_consistency
    input_rgb, input_mask = sample_sv['images_target'][:,:1], sample_sv['masks_target'][:,:1]     # raw input view, [b,1,c,h,w]
    num_frame_rest = num_frame_consistency - 1
    if num_frame_rest > 0:
        render_rgb_rest = results_sv['images_rgb'][:,1:1+num_frame_rest].detach().to(input_rgb)
        render_mask_rest = results_sv['images_weight'][:,1:1+num_frame_rest].detach().to(input_mask)
        input_rgb = torch.cat([input_rgb, render_rgb_rest], dim=1)      # [b,k,c,h,w]
        input_mask = torch.cat([input_mask, render_mask_rest], dim=1)   # [b,k,c,h,w]
    sample_sv_consistency['images_target'] = input_rgb
    sample_sv_consistency['masks_target'] = input_mask

    return sample_sv_consistency


def get_losses_consistency(config, pred, sample, perceptural_loss, clip_model=None):
    losses = {}

    render_imgs, render_masks = pred['images_rgb'], pred['images_weight']
    imgs, masks = sample['images_target'], sample['masks_target']
    assert render_imgs.shape == imgs.shape
    
    b,n,c,h,w = render_imgs.shape
    loss_rgb = F.mse_loss(render_imgs, imgs, reduction='none').mean(dim=[2,3,4]).mean(dim=1).mean()
    loss_perceptual = perceptural_loss(render_imgs.reshape(-1,c,h,w), 
                                       imgs.reshape(-1,c,h,w), 
                                       normalize=(config.train.normalize_img==False)).mean(dim=[1,2,3])
    loss_perceptual = loss_perceptual.reshape(b,n).mean(dim=1).mean()
    losses.update({
        'loss_rgb_cy': loss_rgb,
        'loss_perceptual_cy': loss_perceptual,
        'weight_rgb_cy': config.loss.weight_render_rgb * 10.0,
        'weight_perceptual_cy': config.loss.weight_perceptual * 10.0
    })
   
    if config.loss.weight_render_mask:
        loss_mask = F.mse_loss(render_masks, masks, reduction='none').mean(dim=[2,3,4]).mean(dim=1).mean()
        losses.update({
            'loss_mask_cy': loss_mask,
            'weight_mask_cy': config.loss.weight_render_mask * 10.0
        })
    return losses