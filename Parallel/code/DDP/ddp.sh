python -m torch.distributed.launch \
--nnodes=2 \
--node_rank= $INDEX \
--nproc_per_node $HOST_GPU_NUM \
--master_adderss $CHIEF_IP \
--master_port 1234  \
/apdcephfs/share_916081/shared_info/tingchenfu/AlpacaLora/code/ddp.py 


# export WORLD_SIZE=8 
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
torchrun \
--standalone  \
--nnodes= 2   \
--nproc_per_node= $HOST_GPU_NUM \
--max-restarts=3   \
--rdzv-id=0  \
--rdzv-backend=c10d  \
--rdzv-endpoint=$CHIEF_IP  \
/apdcephfs/share_916081/shared_info/tingchenfu/AlpacaLora/code/ddp.py