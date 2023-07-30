export LD_LIBRARY_PATH="/usr/local/cuda-11.1/targets/x86_64-linux/lib:$LD_LIBRARY_PATH" # Unknown Bug
export HTTPS_PROXY="http://star-proxy.oa.com:3128"

# pip3 install git+https://github.com/huggingface/transformers.git[]
#pip3 install git+https://github.com/huggingface/peft.git
# ROOT_DIR=$(dirname $(dirname $0))
# ROOT_DIR=/apdcephfs/share_916081/effidit_shared_data/hilllzhang/llm_finetune
# DEVICE=0
# BASE_MODEL=${ROOT_DIR}/checkpoints/llama-7b
# LORA_MODEL=${ROOT_DIR}/checkpoints/alpaca-lora-7b
# ALPACA_DATA=${ROOT_DIR}/LLM-Adapters/alpaca_data_cleaned.json
# TS=$(date "+%Y%0m%0d_%T")

# bsz=128
# mbsz=8
# epc=5
# lr=3e-4
# lora_r=16

# MODEL_NAME=alpaca-lora-7b-mine-bsz-${bsz}-epc-${bsz}-lr-${lr}-lora_r-${lora_r}-lora_t-qkvo-proj
# MODEL_OUTPUT=${ROOT_DIR}/checkpoints/${MODEL_NAME}

WORLD_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1234 /apdcephfs/share_916081/shared_info/tingchenfu/AlpacaLora/code/finetune.py \
    --num_epochs=1 \
    --cutoff_len=512 \
    --group_by_length \
    --output_dir='/apdcephfs/share_916081/shared_info/tingchenfu/AlpacaLora/dump/debug' \
    --lora_target_modules='[query_key_value,dense]' \
    --lora_r=16  \
    --micro_batch_size=2  \
