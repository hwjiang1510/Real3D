import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
import random
import os
import dataclasses
import torch.distributed as dist
import itertools
from torch.cuda.amp import autocast
from utils import exp_utils, loss_utils, vis_utils, train_utils, dist_utils
import itertools
from scripts.validator import transform_clip
import clip

logger = logging.getLogger(__name__)


def train_epoch_wild(config, loader, loader_sv, model, optimizer, scheduler, scaler,
                epoch, output_dir, device, rank, perceptual_loss, wandb_run):
    time_meters = exp_utils.AverageMeters()
    loss_meters = exp_utils.AverageMeters()
    loss_meters_sv = exp_utils.AverageMeters()
    loader_sv_cycle = itertools.cycle(loader_sv)

    model.train()
    perceptual_loss.eval()
    clip_model, _ = clip.load("ViT-B/16", device=device)
    clip_model.eval()

    batch_end = time.time()
    loader_sv_iter = iter(loader_sv)
    for batch_idx, sample in enumerate(loader):
        try:
            sample_sv = next(loader_sv_iter)
        except StopIteration:
            loader_sv_iter = iter(loader_sv)
            sample_sv = next(loader_sv_iter)

        if batch_idx > 3000:   # a hacky control of epoch size
            break
        iter_num = batch_idx + 3000 * epoch
        # iter_num = batch_idx + len(loader) * epoch
        sample = exp_utils.dict_to_cuda(sample, device)
        sample = loss_utils.get_loss_target(config, sample)
        sample_sv = exp_utils.dict_to_cuda(sample_sv, device)
        sample_sv = loss_utils.get_loss_target(config, sample_sv)
        time_meters.add_loss_value('Data time', time.time() - batch_end)
        end = time.time()

        # --------------------------  multiview data training --------------------------  
        with autocast(enabled=config.train.use_amp, dtype=torch.bfloat16):
            results = model(sample['input_image'],
                            sample['rays_o'],
                            sample['rays_d'])
            time_meters.add_loss_value('Prediction time', time.time() - end)
            end = time.time()

            losses = loss_utils.get_losses(config, results, sample, perceptual_loss)
            total_loss = 0.0
            for k, v in losses.items():
                if 'loss' in k:
                    total_loss += losses[k.replace('loss_', 'weight_')] * v
                    loss_meters.add_loss_value(k, v.detach().item())
            total_loss = total_loss / config.train.accumulation_step

        if config.train.use_amp:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        time_meters.add_loss_value('Loss time', time.time() - end)
        time_meters.add_loss_value('Batch time', time.time() - batch_end)

        if iter_num % config.print_freq == 0:  
            msg = 'Epoch {0}, Iter {1}, rank {2}, ' \
                'Time: data {data_time:.3f}s, pred {recon_time:.3f}s, loss {loss_time:.3f}s ({batch_time_avg:.3f}s), Loss: '.format(
                epoch, iter_num, rank,
                data_time=time_meters.average_meters['Data time'].avg,
                recon_time=time_meters.average_meters['Prediction time'].avg,
                loss_time=time_meters.average_meters['Loss time'].avg,
                batch_time_avg=time_meters.average_meters['Batch time'].avg
            )
            for k, v in loss_meters.average_meters.items():
                tmp = '{0}: {loss.val:.4f} ({loss.avg:.4f}), '.format(
                        k, loss=v)
                msg += tmp
            msg = msg[:-2]
            logger.info(msg)

        if iter_num % config.vis_freq == 0 and rank == 0:
            vis_utils.vis_seq(vid_clips=sample['images_target'],
                            vid_masks=sample['masks_target'],
                            recon_clips=results['images_rgb'],
                            recon_masks=results['images_weight'],
                            iter_num=iter_num,
                            output_dir=output_dir,
                            subfolder='train_seq',
                            inv_normalize=False)
        # --------------------------  multiview data training ends--------------------------  
            
        
        # --------------------------  single-view data training --------------------------  
        with autocast(enabled=config.train.use_amp, dtype=torch.bfloat16):
            results_sv = model(sample_sv['input_image'],
                            sample_sv['rays_o'],
                            sample_sv['rays_d'])
            time_meters.add_loss_value('Prediction time (SV)', time.time() - end)
            end = time.time()

            losses_sv = loss_utils.get_losses_sv(config, results_sv, sample_sv, perceptual_loss, clip_model)
            total_loss_sv = 0.0
            for k, v in losses_sv.items():
                if 'loss' in k:
                    total_loss_sv += losses_sv[k.replace('loss_', 'weight_')] * v
                    loss_meters_sv.add_loss_value(k, v.detach().item())
            total_loss_sv = total_loss_sv / config.train.accumulation_step

        if config.train.use_amp:
            scaler.scale(total_loss_sv).backward()
            if (batch_idx+1) % config.train.accumulation_step == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_max, norm_type=2.0)
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()
                scheduler.step()
        else:
            total_loss_sv.backward()
            if (batch_idx+1) % config.train.accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_max, norm_type=2.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        time_meters.add_loss_value('Loss time (SV)', time.time() - end)
        time_meters.add_loss_value('Batch time (SV)', time.time() - batch_end)

        if iter_num % config.print_freq == 0:  
            msg = 'Epoch {0}, Iter {1}, rank {2}, ' \
                'Time: data {data_time:.3f}s, pred {recon_time:.3f}s, loss {loss_time:.3f}s ({batch_time_avg:.3f}s), Loss: '.format(
                epoch, iter_num, rank,
                data_time=time_meters.average_meters['Data time'].avg,
                recon_time=time_meters.average_meters['Prediction time (SV)'].avg,
                loss_time=time_meters.average_meters['Loss time (SV)'].avg,
                batch_time_avg=time_meters.average_meters['Batch time (SV)'].avg
            )
            for k, v in loss_meters_sv.average_meters.items():
                tmp = '{0}: {loss.val:.4f} ({loss.avg:.4f}), '.format(
                        k, loss=v)
                msg += tmp
            msg = msg[:-2]
            logger.info(msg)

        if iter_num % config.vis_freq == 0 and rank == 0:
            n = results_sv['images_rgb'].shape[1]
            vis_utils.vis_seq(vid_clips=sample_sv['images_target'].repeat(1,n,1,1,1),
                            vid_masks=sample_sv['masks_target'].repeat(1,n,1,1,1),
                            recon_clips=results_sv['images_rgb'],
                            recon_masks=results_sv['images_weight'],
                            iter_num=iter_num,
                            output_dir=output_dir,
                            subfolder='train_seq_sv',
                            inv_normalize=False)
        # --------------------------  single-view data training -------------------------- 
            
        if rank == 0 and wandb_run is not None:
            wandb_log = {'Train/loss': total_loss.item(),
                         'Train/loss_sv': total_loss_sv.item(),
                        'Train/lr': optimizer.param_groups[0]['lr']}
            for k, v in losses.items():
                if 'loss' in k:
                    wandb_log['Train/{}'.format(k)] = v.item()
            for k, v in losses_sv.items():
                if 'loss' in k:
                    wandb_log['Train/{}'.format(k)] = v.item()
            wandb_run.log(wandb_log)

        # if iter_num % 1000 == 0 and rank == 0:
        #     if config.train.use_zeroRO:
        #         optimizer.consolidate_state_dict(to=rank)
        #     train_utils.save_checkpoint(
        #             {
        #                 'epoch': epoch + 1,
        #                 'state_dict': model.module.state_dict(),
        #                 'optimizer': optimizer.state_dict(),
        #                 'scaler': scaler.state_dict(),
        #                 'schedular': scheduler.state_dict(),
        #             }, 
        #             checkpoint=output_dir, filename="cpt_last.pth.tar")

        if iter_num % 1000 == 0 and iter_num != 0:
            if config.train.use_zeroRO:
                print('Consolidated on rank {} because of ZeRO'.format(rank))
                optimizer.consolidate_state_dict(0)
            dist_utils.save_on_master({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict() if dist_utils.is_main_process() else None,
                'scaler': scaler.state_dict(),
                'schedular': scheduler.state_dict()
            }, save_path=os.path.join(output_dir, "cpt_last.pth.tar"))
        
        dist.barrier()
        batch_end = time.time()
    
    del losses, total_loss, results, sample

