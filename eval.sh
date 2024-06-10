# CUDA_VISIBLE_DEVICES=0, python -m torch.distributed.launch --master_port 9999 --nproc_per_node=1 \
# eval_mv.py --cfg ./config/eval/eval_mv_co3d.yaml --vis360

# CUDA_VISIBLE_DEVICES=0, python -m torch.distributed.launch --master_port 9999 --nproc_per_node=1 \
# eval_mv.py --cfg ./config/eval/eval_mv_mvimgnet.yaml --vis360

# CUDA_VISIBLE_DEVICES=0, python -m torch.distributed.launch --master_port 9999 --nproc_per_node=1 \
# eval_mv.py --cfg ./config/eval/eval_mv_omniobject3d.yaml --vis360

# CUDA_VISIBLE_DEVICES=0, python -m torch.distributed.launch --master_port 9999 --nproc_per_node=1 \
# eval_sv.py --cfg ./config/eval/eval_sv.yaml --vis360
