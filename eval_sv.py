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
from scripts.validator import evaluation, evaluation_consistency
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
    parser.add_argument(
        '--vis360', action='store_true', default=False)
    args, rest = parser.parse_known_args()
    return args


def main():
    # Get args and config
    args = parse_args()
    config = OmegaConf.load(args.cfg)
    config = parse_config(exp_utils.to_easydict_recursively(config))
    logger, output_dir = exp_utils.create_logger(config, args.cfg, phase='train')
    logger.info(pprint.pformat(output_dir))

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

    # load pre-trained model
    if len(config.train.pretrain_path) > 0:
        model = train_utils.load_pretrain(config, model, config.train.pretrain_path, strict=True)

    # distributed training
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = True
        device_ids = range(torch.cuda.device_count())
        print("using {} cuda".format(len(device_ids)))
        find_unused = True if (not config.model.backbone_fix) else False
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=find_unused)
        device_num = len(device_ids)

    # validation dataset (single-view)
    val_split = 'val'
    val_data = train_utils.get_dataset_testing(config, split=val_split)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=config.test.batch_size, 
                                             shuffle=False,
                                             num_workers=int(config.workers), 
                                             pin_memory=True, 
                                             drop_last=False)
   
    
    print('Doing validation')
    # evaluate novel view semantic similarity
    metrics = evaluation(config,
                        loader=val_loader,
                        model=model,
                        output_dir=output_dir,
                        device=device,
                        rank=local_rank,
                        wandb_run=wandb_run,
                        epoch=0,
                        eval=args.eval,
                        data_name=config.dataset.sv_test_data_name)
    # metrics = {}
    # metrics_consistency = evaluation_consistency(config,
    #                     loader=val_loader,
    #                     model=model,
    #                     output_dir=output_dir,
    #                     device=device,
    #                     rank=local_rank,
    #                     wandb_run=wandb_run,
    #                     epoch=0,
    #                     eval=args.eval,
    #                     data_name=config.dataset.sv_test_data_name)
    # metrics.update(metrics_consistency)
    
    print(metrics)

        
if __name__ == '__main__':
    main()