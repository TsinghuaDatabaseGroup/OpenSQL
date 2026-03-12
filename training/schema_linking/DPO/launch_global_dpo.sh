#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [[ $# -lt 5 ]]; then
    echo "Usage: bash launch_global_dpo.sh <lr> <beta> <rpo_alpha> <epoch> <sft_model_path> [finetune_data_path] [storage_dir] [seed]" >&2
    exit 1
fi

dpo_lr="$1"
dpo_beta="$2"
rpo_alpha="$3"
dpo_epoch="$4"
sft_model_path="$5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cache_dir="${MODEL_CACHE_DIR:-${PROJECT_ROOT}/artifacts/model_cache}"
default_data="${PROJECT_ROOT}/artifacts/training_data/global_schema_linking_dpo.json"
finetune_data_name="${6:-${FINETUNE_DATA_PATH:-${default_data}}}"
default_storage="${PROJECT_ROOT}/artifacts/trained_models/global_linker_dpo"
storage_dir="${7:-${MODEL_STORAGE_DIR:-${default_storage}}}"
seed="${8:-${SEED:-$(od -An -N2 -t u2 /dev/urandom | tr -d ' ')}}"

config_file="${ACCELERATE_CONFIG:-${PROJECT_ROOT}/training/config/DPO.json}"
num_processes="${NUM_PROCESSES:-4}"
attn_impl="${ATTN_IMPL:-flash_attention_2}"
torch_dtype="${TORCH_DTYPE:-bfloat16}"

mkdir -p "${storage_dir}"

accelerate launch --config_file="${config_file}" \
    --num_processes "${num_processes}" \
    "${SCRIPT_DIR}/schema_dpo.py" \
    --cache_dir "${cache_dir}" \
    --sft_model_path "${sft_model_path}" \
    --dpo_model_storage_dir "${storage_dir}" \
    --finetune_data_dir "${finetune_data_name}" \
    --attn_implementation "${attn_impl}" \
    --output_dir "${storage_dir}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --learning_rate "${dpo_lr}" \
    --num_train_epochs "${dpo_epoch}" \
    --lr_scheduler_type cosine \
    --torch_dtype "${torch_dtype}" \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --bf16 true \
    --report_to none \
    --max_length 7500 \
    --max_prompt_length 7500 \
    --beta "${dpo_beta}" \
    --rpo_alpha "${rpo_alpha}" \
    --seed "${seed}"
