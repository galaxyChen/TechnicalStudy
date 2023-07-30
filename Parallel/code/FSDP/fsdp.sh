# python -m torch.distributed.launch \
# --nnodes=2 \
# --node_rank=0 \
# --nproc_per_node 8 \
# --master_adderss $CHIEF_IP \
# --master_port 1234  \
# main.py


# export WORLD_SIZE=8 
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
torchrun \
--standalone  \
--nnodes=1    \
--nproc_per_node=8 \
--max-restarts=1   \
--rdzv-id=0  \
--rdzv-backend=c10d  \
--rdzv-endpoint=$CHIEF_IP  \
/apdcephfs/share_916081/shared_info/tingchenfu/AlpacaLora/code/fsdp.py
