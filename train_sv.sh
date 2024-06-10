# this is slow but might be more accurate
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 8888 --nproc_per_node=8 \
train.py --cfg ./config/objaverse_both_all/train_sv-novel_consistency-curriculum-rerender.yaml

# this is faster but might be a bit less accurate
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 8888 --nproc_per_node=8 \
# train.py --cfg ./config/objaverse_both_all/train_sv-novel_consistency-curriculum.yaml