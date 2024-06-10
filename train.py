import os
import pprint
import random
import numpy as np
import json
import torch
import torch.nn.parallel
import torch.optim
import itertools
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
from torch.utils.data import WeightedRandomSampler
import argparse
import wandb
import transformers
from omegaconf import OmegaConf
from config.parse_config import parse_config
from utils import exp_utils, dist_utils, train_utils, loader_utils
from scripts.trainer import train_epoch
from scripts.trainer_with_wild import train_epoch_wild
from scripts.trainer_with_wild_consistency import train_epoch_wild_with_consistency
from scripts.validator import evaluation, evaluation_fid, evaluation_consistency
from scripts.validator_mv import evaluation as evaluation_mv
from dataset.omniobject3d import Omniobject3D
from dataset.co3d import CO3D
from dataset.mvimgnet import MVIMageNet_MV
import lpips
from tsr.system import TSR


def parse_args():
    parser = argparse.ArgumentParser(description='Train Wild3D')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        '--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument(
        '--eval', action='store_true', default=False)
    parser.add_argument(
        '--debug', action='store_true', default=False)
    args, rest = parser.parse_known_args()
    return args


def main():
    # Get args and config
    args = parse_args()
    config = OmegaConf.load(args.cfg)
    config = parse_config(exp_utils.to_easydict_recursively(config))
    logger, output_dir = exp_utils.create_logger(config, args.cfg, phase='train')

    # set random seeds
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # set device
    gpus = range(torch.cuda.device_count())
    distributed = torch.cuda.device_count() > 1
    device = torch.device('cuda') if len(gpus) > 0 else torch.device('cpu')
    if "LOCAL_RANK" in os.environ:
        dist_utils.dist_init(int(os.environ["LOCAL_RANK"]))
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        logger.info(pprint.pformat(args))
        logger.info(pprint.pformat(config))

    # set wandb
    if local_rank == 0 and (not args.debug):
        wandb_name = config.exp_name
        wandb_proj_name = config.exp_group
        wandb_run = wandb.init(project=wandb_proj_name, group=wandb_name)
    else:
        wandb_run = None

    # get model
    model = TSR.from_pretrained(config.model.pretrain_path,
                                config_name="config.yaml",
                                weight_name=config.model.model_name).to(device)
    model.renderer.set_chunk_size(config.model.render_chunk_size)
    model.renderer.set_num_samples_per_ray(config.model.render_num_samples_per_ray)
    model.backbone.gradient_checkpointing = config.train.use_checkpointing
    perceptual_loss = lpips.LPIPS(net="vgg", eval_mode=True).to(device)
    config.model_config = model.cfg
    if local_rank == 0:
        logger.info(pprint.pformat(model.cfg))

    # get optimizer
    optimizer = train_utils.get_optimizer(config, model)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=config.train.warmup_iter // config.train.accumulation_step,
                                                             num_training_steps=config.train.total_iteration // config.train.accumulation_step)
    scaler = torch.cuda.amp.GradScaler()

    # load pre-trained model
    if len(config.train.pretrain_path) > 0:
        model = train_utils.load_pretrain(config, model, config.train.pretrain_path, strict=True)

    # resume training
    best_sim, best_psnr = 0.0, 0.0
    ep_resume = None
    if config.train.resume:
        model, optimizer, scheduler, scaler, ep_resume, best_sim, best_psnr = train_utils.resume_training(
                                                                        model, optimizer, scheduler, scaler,
                                                                        output_dir, cpt_name='cpt_last.pth.tar')
        print('LR after resume {}'.format(optimizer.param_groups[0]['lr']))
    else:
        print('No resume training')

    # distributed training
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = True
        device_ids = range(torch.cuda.device_count())
        print("using {} cuda".format(len(device_ids)))
        find_unused = True if (not config.model.backbone_fix) else False
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=find_unused)
        device_num = len(device_ids)

    # training set multiview
    train_data = train_utils.get_dataset_multiview(config, split='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.train.batch_size, 
                                               shuffle=False,
                                               num_workers=int(config.workers), 
                                               pin_memory=True, 
                                               drop_last=True,
                                               sampler=train_sampler)
    # training set single-view
    train_function = None
    if config.dataset.sv_data_use:
        train_data_sv, weights_sv, num_samples_sv = train_utils.get_dataset_single_view(config, split='train', length=len(train_data))
        #train_sampler_sv = loader_utils.DistributedWeightedSampler(num_samples_sv, weights_sv, replacement=True)
        train_sampler_sv = torch.utils.data.distributed.DistributedSampler(train_data_sv)
        train_loader_sv = torch.utils.data.DataLoader(train_data_sv,
                                                        batch_size=config.train.batch_size_sv, 
                                                        shuffle=False,
                                                        num_workers=int(config.workers), 
                                                        pin_memory=True, 
                                                        drop_last=True,
                                                        sampler=train_sampler_sv)
        train_function = train_epoch_wild_with_consistency if config.train.use_consistency else train_epoch_wild
    else:
        train_loader_sv = None
        train_function = train_epoch

    # validation dataset (single-view)
    val_split = 'val' #if not args.eval else 'test'
    val_data = train_utils.get_dataset_testing(config, split=val_split)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=config.test.batch_size, 
                                             shuffle=False,
                                             num_workers=int(config.workers), 
                                             pin_memory=True, 
                                             drop_last=False)
    
    val_data_mv1 = MVIMageNet_MV(config, split=val_split)
    val_data_mv2 = CO3D(config, split=val_split)
    val_data_mv3 = Omniobject3D(config, split=val_split, mode='multiview')
    val_loader_mv1 = torch.utils.data.DataLoader(val_data_mv1, batch_size=config.test.batch_size, shuffle=False,
                                                num_workers=int(config.workers), pin_memory=True, drop_last=False)
    val_loader_mv2 = torch.utils.data.DataLoader(val_data_mv2, batch_size=config.test.batch_size, shuffle=False,
                                                num_workers=int(config.workers), pin_memory=True, drop_last=False)
    val_loader_mv3 = torch.utils.data.DataLoader(val_data_mv3, batch_size=config.test.batch_size, shuffle=False,
                                                num_workers=int(config.workers), pin_memory=True, drop_last=False)

    
    start_ep = ep_resume - 1 if ep_resume is not None else 0
    end_ep = 100 if not args.eval else (start_ep + 1)
    
    # train
    for epoch in range(start_ep, end_ep):
        if not args.eval:
            if config.dataset.sv_data_use:
                train_sampler_sv.set_epoch(epoch)
            train_sampler.set_epoch(epoch)
            train_function(config,
                        loader=train_loader,
                        loader_sv=train_loader_sv,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        output_dir=output_dir,
                        device=device,
                        rank=local_rank,
                        perceptual_loss=perceptual_loss,
                        wandb_run=wandb_run
                        )

            
        print('Doing validation')
        # evaluate novel view semantic similarity
        metrics = evaluation(config,
                            loader=val_loader,
                            model=model,
                            epoch=epoch,
                            output_dir=output_dir,
                            device=device,
                            rank=local_rank,
                            wandb_run=wandb_run,
                            eval=args.eval,
                            data_name=config.dataset.sv_test_data_name)
        
        # evaluate fid score of novel views
        # cur_fid = evaluation_fid(config, 
        #                          output_dir,
        #                          device=device,
        #                          world_size=device_num,
        #                          local_rank=local_rank,
        #                          epoch=epoch,
        #                          wandb_run=wandb_run,
        #                          eval=args.eval)
        # metrics['nvs_fid_all'] = cur_fid
        # metrics['nvs_fid_all'] = float('inf')

        # evaluate cycle-consistency of inputs
        # metrics_consistency = evaluation_consistency(config,
        #                                             loader=val_loader,
        #                                             model=model,
        #                                             epoch=epoch,
        #                                             output_dir=output_dir,
        #                                             device=device,
        #                                             rank=local_rank,
        #                                             wandb_run=wandb_run,
        #                                             eval=args.eval)
        # metrics.update(metrics_consistency)
        
        # cur_sim = metrics['nvs_clip_all']
        # if cur_sim > best_sim:
        #     best_sim = cur_sim
        #     if config.train.use_zeroRO:
        #         print('Consolidated on rank {} because of ZeRO'.format(local_rank))
        #         optimizer.consolidate_state_dict(0)
        #     save_path = os.path.join(output_dir, "cpt_best_nvs_sim_{}_consistency_psnr_{}.pth.tar".format(best_sim, metrics['consistency_psnr_all']))
        #     dist_utils.save_on_master({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'optimizer' : optimizer.state_dict() if dist_utils.is_main_process() else None,
        #         'scaler': scaler.state_dict(),
        #         'schedular': scheduler.state_dict(),
        #         'best_sim': best_sim,
        #         'eval_dict': metrics,
        #     }, save_path=save_path)

        metrics_mv = evaluation_mv(config, val_loader_mv1, model, 'mvimgnet',
                                   epoch, output_dir, device, local_rank, wandb_run, eval=args.eval)
        metrics.update(metrics_mv)
        metrics_mv = evaluation_mv(config, val_loader_mv2, model, 'co3d',
                                   epoch, output_dir, device, local_rank, wandb_run, eval=args.eval)
        metrics.update(metrics_mv)
        metrics_mv = evaluation_mv(config, val_loader_mv3, model, 'omniobject3d',
                                   epoch, output_dir, device, local_rank, wandb_run, eval=args.eval)
        metrics.update(metrics_mv)

        cur_psnr = metrics['mvimgnet_psnr']
        cur_sim = metrics['nvs_clip_all']
        best_sim = max(best_sim, cur_sim)
        if cur_psnr > best_psnr:
            best_psnr = cur_psnr
            if config.train.use_zeroRO:
                print('Consolidated on rank {} because of ZeRO'.format(local_rank))
                optimizer.consolidate_state_dict(0)
            save_path = os.path.join(output_dir, "cpt_best_mv_psnr_{}_nvs_sim_{}.pth.tar".format(best_psnr, cur_sim))
            dist_utils.save_on_master({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict() if dist_utils.is_main_process() else None,
                'scaler': scaler.state_dict(),
                'schedular': scheduler.state_dict(),
                'best_sim': best_sim,
                'best_psnr': best_psnr,
                'eval_dict': metrics,
            }, save_path=save_path)

        
        if local_rank == 0:
            with open(os.path.join(output_dir, 'result_ep{}.json'.format(epoch)), 'w') as f:
                json.dump(metrics, f, indent=4)
        
        dist.barrier()
        torch.cuda.empty_cache()

        
if __name__ == '__main__':
    main()