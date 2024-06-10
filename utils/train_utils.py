import os
import pprint
import random
import numpy as np
import torch
import warnings
import torch.nn as nn
import kornia
import kornia.augmentation as K
from kornia.constants import DataKey
import copy
from einops import rearrange
from torch.utils.data import ConcatDataset, Dataset
from dataset.omniobject3d import Omniobject3D
from dataset.objaverse import Objaverse
from dataset.objaverse_0123 import Objaverse_0123
from dataset.mvimagenet_sv import MVIMageNet_SV
from dataset.wild_sv import WILD_SV
from dataset.omniobject3d import Omniobject3D
from dataset.co3d import CO3D
from dataset.mvimgnet import MVIMageNet_MV
from dataset.demo import DEMO_SV
from dataset.wild_sv_unfiltered import WILD_SV_UNFILTERED
from dataset.constant import *
from utils import loader_utils, process_utils
import shutil
from torch.distributed.optim import ZeroRedundancyOptimizer


def split_param_for_weight_decay(param_list):
    wd_params, non_wd_params = [], []
    for (name, param) in param_list:
        if (param.ndim < 2) or ('bias' in name) or ('ln' in name) or ('bn' in name) or ('norm' in name):
            non_wd_params.append(param)
        else:
            wd_params.append(param)
    return wd_params, non_wd_params


def get_optimizer(config, model):
    '''
    Split the params into four parts:
        1. 3D latent parameters
        2. backbone parameters -- bias and weights
        3. backbone parameters -- others
        4. other model parameters -- bias and weights
        5. other model parameters -- others
    '''
    # get 3d latent parameters
    pe_list = ['tokenizer.embeddings']
    pe_params = []
    for it in pe_list:
        pe_params += list(filter(lambda kv: it in kv[0], model.named_parameters()))
    
    # get backbone parameters
    backbone_param = list(filter(lambda kv: 'image_tokenizer' in kv[0], model.named_parameters()))
    backbone_param_wd, backbone_param_non_wd = split_param_for_weight_decay(backbone_param)

    # other parameters
    non_other_param_names = list(map(lambda x: x[0], pe_params)) + list(map(lambda x: x[0], backbone_param))
    other_param = [(name, param) for name, param in model.named_parameters() if name not in non_other_param_names]
    other_param_wd, other_param_non_wd = split_param_for_weight_decay(other_param)

    pe_params = list(map(lambda x: x[1], pe_params))
    wd = config.train.weight_decay
    # get optimizer
    if not config.train.use_zeroRO:
        if config.model.backbone_fix:
            for param in model.backbone.parameters():
                param.requires_grad = False
            optimizer = torch.optim.AdamW([{'params': pe_params, 'lr': config.train.lr_embeddings, 'weight_decay': wd},
                                        {'params': other_param_wd, 'lr': config.train.lr, 'weight_decay': wd},
                                        {'params': other_param_non_wd, 'lr': config.train.lr, 'weight_decay': 0}],
                                        lr=config.train.lr,
                                        weight_decay=config.train.weight_decay,
                                        betas=(config.train.beta1, config.train.beta2),
                                        eps=config.train.eps)
        else:
            optimizer = torch.optim.AdamW([{'params': pe_params, 'lr': config.train.lr_embeddings, 'weight_decay': wd},
                                        {'params': backbone_param_wd, 'lr': config.train.lr_backbone, 'weight_decay': wd},
                                        {'params': backbone_param_non_wd, 'lr': config.train.lr_backbone, 'weight_decay': 0},
                                        {'params': other_param_wd, 'lr': config.train.lr, 'weight_decay': wd},
                                        {'params': other_param_non_wd, 'lr': config.train.lr, 'weight_decay': 0}],
                                        lr=config.train.lr,
                                        weight_decay=config.train.weight_decay,
                                        betas=(config.train.beta1, config.train.beta2),
                                        eps=config.train.eps)
    else:
        if config.model.backbone_fix:
            for param in model.backbone.parameters():
                param.requires_grad = False
            params_dict = [{'params': pe_params, 'lr': config.train.lr_embeddings, 'weight_decay': wd},
                            {'params': other_param_wd, 'lr': config.train.lr, 'weight_decay': wd},
                            {'params': other_param_non_wd, 'lr': config.train.lr, 'weight_decay': 0}]
        else:
            params_dict = [{'params': pe_params, 'lr': config.train.lr_embeddings, 'weight_decay': wd},
                            {'params': backbone_param_wd, 'lr': config.train.lr_backbone, 'weight_decay': wd},
                            {'params': backbone_param_non_wd, 'lr': config.train.lr_backbone, 'weight_decay': 0},
                            {'params': other_param_wd, 'lr': config.train.lr, 'weight_decay': wd},
                            {'params': other_param_non_wd, 'lr': config.train.lr, 'weight_decay': 0}]
        optimizer = ZeroRedundancyOptimizer(params_dict,
                                            optimizer_class=torch.optim.AdamW,
                                            lr=config.train.lr,
                                            weight_decay=config.train.weight_decay,
                                            betas=(config.train.beta1, config.train.beta2),
                                            eps=config.train.eps)
            

    return optimizer


def get_dataset_multiview(config, split):
    name = config.dataset.mv_data_name
    if name == 'objaverse':
        data = Objaverse(config, split)
    elif name == 'objaverse0123':
        data = Objaverse_0123(config, split)
    elif name == 'objaverse_both':
        data1 = Objaverse(config, split)
        data2 = Objaverse_0123(config, split)
        data = ConcatDataset([data1, data2])
    else:
        NotImplementedError('Not implemented dataset')  
    return data


def get_dataset_single_view(config, split='train', length=1):
    if split == 'train':
        names = config.dataset.sv_data_name.split('_')
    else:
        raise NotImplementedError
    datas, weights = [], []
    for name in names:
        if name == 'mvimgnet':
            datas.append(MVIMageNet_SV(config, split, length))
            #weights.append(WEIGHT_MVIMGNET)
            weights.append(1.0)
        elif name == 'wild':
            datas.append(WILD_SV(config, split, length))
            #weights.append(WEIGHT_WILD)
            weights.append(1.0)
        elif name == 'omniobject3d':
            datas.append(Omniobject3D(config, split, mode='singleview', length=length))
            weights.append(1.0)
        elif name == 'wild-unfiltered':
            datas.append(WILD_SV_UNFILTERED(config, split, length))
            weights.append(1.0)
        else:
            NotImplementedError('Not implemented dataset')
    
    # normalize weights
    weights = [it / sum(weights) for it in weights]

    # replicate weights for all instances of each dataset
    weights_replicate = loader_utils.replicate_weights(weights, datas)

    # get maximum number of iterations
    num_samples = loader_utils.calculate_max_num_samples(datas)
    
    datas = ConcatDataset(datas)
    print('Using {} single-view data instances for {} in total'.format(len(datas), split))
    return datas, weights_replicate, num_samples


def get_dataset_testing(config, split='test'):
    name = config.dataset.sv_test_data_name
    if name == 'mvimgnet':
        data = MVIMageNet_SV(config, split)
    elif name == 'wild':
        data = WILD_SV(config, split)
    elif name == 'omniobject3d':
        data = Omniobject3D(config, split, mode='multiview')
    elif name == 'co3d':
        data = CO3D(config, split)
    elif name == 'mvimgnet_mv':
        data = MVIMageNet_MV(config, split)
    elif name == 'demo':
        data = DEMO_SV(config, split)
    else:
        NotImplementedError('Not implemented dataset')  
    return data



def resume_training(model, optimizer, schedular, scaler, output_dir, cpt_name='cpt_last.pth.tar', strict=True, device=None):
    output_dir = os.path.join(output_dir, cpt_name)
    if os.path.isfile(output_dir):
        print("=> loading checkpoint {}".format(output_dir))
        if device is not None:
            checkpoint = torch.load(output_dir, map_location=device)
        else:
            checkpoint = torch.load(output_dir, map_location=torch.device('cpu'))
        
        # load model
        if "module" in list(checkpoint["state_dict"].keys())[0]:
            state_dict = {key.replace('module.',''): item for key, item in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint["state_dict"]
        missing_states = set(model.state_dict().keys()) - set(state_dict.keys())
        if len(missing_states) > 0:
            warnings.warn("Missing keys ! : {}".format(missing_states))
        model.load_state_dict(state_dict, strict=strict)

        # load optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])

        # load schedular
        schedular.load_state_dict(checkpoint['schedular'])

        # load scaler
        scaler.load_state_dict(checkpoint['scaler'])

        # load epoch
        start_epoch = checkpoint['epoch']

        # load data
        best_sim = checkpoint['best_sim'] if 'best_sim' in checkpoint.keys() else 0.0
        best_psnr = checkpoint['best_psnr'] if 'best_psnr' in checkpoint.keys() else 0.0

        del checkpoint, state_dict

        return model, optimizer, schedular, scaler, start_epoch, best_sim, best_psnr
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(output_dir))
    

def load_pretrain(config, model, cpt_path, strict=True):
    checkpoint = torch.load(cpt_path, map_location=torch.device('cpu'))
    # load model
    if "module" in list(checkpoint["state_dict"].keys())[0]:
        state_dict = {key.replace('module.',''): item for key, item in checkpoint["state_dict"].items()}
    else:
        state_dict = checkpoint["state_dict"]

    state_dict_new = copy.deepcopy(state_dict)

    missing_states = set(model.state_dict().keys()) - set(state_dict_new.keys())
    if len(missing_states) > 0:
        warnings.warn("Missing keys ! : {}".format(missing_states))
    model.load_state_dict(state_dict_new, strict=strict)

    del checkpoint, state_dict, state_dict_new
    return model


def save_checkpoint(state, checkpoint="checkpoint", filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


def copy_folder(src, dst):
    # Check if the destination directory exists, if not, create it
    if not os.path.exists(dst):
        os.makedirs(dst)
    # Copy the entire source directory to destination
    shutil.copytree(src, dst, dirs_exist_ok=True)


def set_iteration(loader, cur_iter):
    if isinstance(loader.dataset, ConcatDataset):
        for data in loader.dataset.datasets:
            data._set_iteration(cur_iter)
    elif isinstance(loader.dataset, Dataset):
        loader.dataset._set_iteration(cur_iter)


def set_pose_curriculum(config, cur_iter, sample):
    '''
    The function for setting the camera for consistency with curriculum
    sample = {
                'input_image':      [b,1,c,h,w]
                'rays_o':           [b,n,h,w,3]
                'rays_d':           [b,n,h,w,3], for rendering n novel views
                'render_images':    [b,n=1,c,h,w]
                'render_masks':     [b,n=1,1,h,w]
                'ks':               [b,n,3,3]
                'seq_name': seq_name
            }
    Use the last novel view for cycle-consistency
    '''
    b, device = sample['input_image'].shape[0], sample['input_image'].device
    n_consistency = config.train.num_frame_consistency
    assert n_consistency == 1
    camera_distance = SCALE_OBJAVERSE if config.dataset.mv_data_name == 'objaverse' else SCALE_OBJAVERSE_BOTH
    total_iter = config.train.total_iteration
    sampling = config.dataset.sv_render_views_sample
    Ks = sample['Ks']   # [b,n,3,3]
    render_size = config.model.render_resolution
    input_size = config.dataset.img_size
    
    curriculum = config.dataset.sv_curriculum
    azimuths, elevations = curriculum.split('_')
    min_az, max_az = azimuths.split('-')
    min_el, max_el = elevations.split('-')
    min_az, max_az, min_el, max_el = float(min_az), float(max_az), float(min_el), float(max_el)
    cur_progress = cur_iter / total_iter
    cur_max_az = min_az + cur_progress * (max_az - min_az)
    cur_max_el = min_el + cur_progress * (max_el - min_el)

    c2w_all = []
    for _ in range(b):
        c2w = process_utils.get_cameras_curriculum(n_consistency, camera_distance, 
                                                   min_az, min_el, cur_max_az, cur_max_el)  # [n_consistency,4,4]
        c2w_all.append(c2w)
    c2w_all = torch.stack(c2w_all).to(device)                                               # [b,n_consistency,4,4]

    c2w_canonical = process_utils.get_cameras(1, 0, camera_distance, sampling).unsqueeze(0).repeat(b,1,1,1).to(device)  # [b,1,4,4]
    c2w_all_consistency = c2w_canonical @ torch.inverse(c2w_all) @ c2w_canonical            # [b,n_consistency,4,4]

    rays_o, rays_d, rays_o_consistency, rays_d_consistency = [], [], [], []
    for i in range(b):
        ray_o, ray_d = process_utils.get_rays_from_pose(c2w_all[i], focal=Ks[i,:n_consistency,0,0], size=render_size)
        ray_o_c, ray_d_c = process_utils.get_rays_from_pose(c2w_all_consistency[i], focal=Ks[i,:n_consistency,0,0], size=render_size)
        rays_o.append(ray_o)
        rays_d.append(ray_d)
        rays_o_consistency.append(ray_o_c)
        rays_d_consistency.append(ray_d_c)
    rays_o = torch.stack(rays_o)
    rays_d = torch.stack(rays_d)
    rays_o_consistency = torch.stack(rays_o_consistency)
    rays_d_consistency = torch.stack(rays_d_consistency)
    sample['rays_o'][:,-n_consistency:] = rays_o
    sample['rays_d'][:,-n_consistency] = rays_d
    sample['rays_o_consistency'] = rays_o_consistency
    sample['rays_d_consistency'] = rays_d_consistency

    if config.train.rerender_consistency_input:
        rays_o_hres, rays_d_hres = [], []
        for i in range(b):
            ray_o_hres, ray_d_hres = process_utils.get_rays_from_pose(c2w_all[i], 
                                                                    focal=Ks[i,:n_consistency,0,0] / render_size * input_size, 
                                                                    size=input_size)
            rays_o_hres.append(ray_o_hres)
            rays_d_hres.append(ray_d_hres)
        rays_o_hres = torch.stack(rays_o_hres)
        rays_d_hres = torch.stack(rays_d_hres)
        sample['rays_o_hres'][:,-n_consistency:] = rays_o_hres
        sample['rays_d_hres'][:,-n_consistency] = rays_d_hres

    return sample








