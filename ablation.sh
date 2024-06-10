# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 8888 --nproc_per_node=7 \
# train.py --cfg ./config/ablations/no_curriculum.yaml

# CUDA_VISIBLE_DEVICES=0, python -m torch.distributed.launch --master_port 9999 --nproc_per_node=1 \
# eval_mv.py --cfg ./config/ablations/ablation_e2e-consistency-curriculum.yaml

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 8888 --nproc_per_node=7 \
# train.py --cfg ./config/ablations/ablation_unfiltered-data_all-loss.yaml

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 8888 --nproc_per_node=7 \
# train.py --cfg ./config/ablations/ablation_semantic-only.yaml

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 8888 --nproc_per_node=7 \
# train.py --cfg ./config/ablations/ablation_semantic-only-naive.yaml

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 8888 --nproc_per_node=7 \
# train.py --cfg ./config/ablations/ablation_input-render-only-no-curation.yaml

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 8888 --nproc_per_node=7 \
train.py --cfg ./config/ablations/ablation_data_amount.yaml