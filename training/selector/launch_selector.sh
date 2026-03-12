#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ $# -lt 3 ]]; then
    echo "Usage: bash launch_selector.sh <lr> <epoch> <model_name_or_path> [finetune_data_path] [storage_dir] [seed]" >&2
    exit 1
fi

lr="$1"
epoch="$2"
model_name_or_path="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cache_dir="${MODEL_CACHE_DIR:-${PROJECT_ROOT}/artifacts/model_cache}"
default_data="${PROJECT_ROOT}/artifacts/training_data/selector.json"
finetune_data_name="${4:-${FINETUNE_DATA_PATH:-${default_data}}}"
default_storage="${PROJECT_ROOT}/artifacts/trained_models/selector"
storage_dir="${5:-${MODEL_STORAGE_DIR:-${default_storage}}}"
seed="${6:-${SEED:-$(od -An -N2 -t u2 /dev/urandom | tr -d ' ')}}"

config_file="${ACCELERATE_CONFIG:-${PROJECT_ROOT}/training/config/SFT.json}"
num_processes="${NUM_PROCESSES:-4}"
attn_impl="${ATTN_IMPL:-flash_attention_2}"
torch_dtype="${TORCH_DTYPE:-bfloat16}"

mkdir -p "${storage_dir}"

accelerate launch --config_file="${config_file}" \
    --num_processes "${num_processes}" \
    "${SCRIPT_DIR}/train_selector.py" \
    --cache_dir "${cache_dir}" \
    --model_storage_dir "${storage_dir}" \
    --finetune_data_dir "${finetune_data_name}" \
    --model_name_or_path "${model_name_or_path}" \
    --attn_implementation "${attn_impl}" \
    --torch_dtype "${torch_dtype}" \
    --output_dir "${storage_dir}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --learning_rate "${lr}" \
    --num_train_epochs "${epoch}" \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 10 \
    --bf16 true \
    --max_seq_length 8192 \
    --seed "${seed}" \
    --eval_strategy no \
    --save_strategy no \
    --report_to none
