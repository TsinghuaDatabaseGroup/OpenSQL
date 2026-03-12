#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=4
export NCCL_TIMEOUT=36000
export NCCL_ASYNC_ERROR_HANDLING=1
export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=4,5
export VLLM_WORKER_MULTIPROC_METHOD=spawn

CURRENT_BENCHMARK=${1:?Usage: ./start_pipeline.sh <BENCHMARK> <SETTING>}
SETTING=${2:?Usage: ./start_pipeline.sh <BENCHMARK> <SETTING>}

LOCAL_MODEL=/mnt/sdb/ruilin/Final-Release/LocalLinker-3B

case "${SETTING}" in
    OmniSQL-7B)
        GLOBAL_MODEL=/mnt/sdb/ruilin/Final-Release/OmniSQL-7B/SchemaLinker
        GENERATOR_MODEL=/mnt/sdb/ruilin/Final-Release/OmniSQL-7B/Generator
        SELECT_MODEL=/mnt/sdb/ruilin/Final-Release/OmniSQL-7B/Selector
        ;;
    Qwen2.5-Coder-7B)
        GLOBAL_MODEL=/mnt/sdb/ruilin/Final-Release/Qwen2.5-Coder-7B/SchemaLinker
        GENERATOR_MODEL=/mnt/sdb/ruilin/Final-Release/Qwen2.5-Coder-7B/Revise-Generator-Qwen-7B
        SELECT_MODEL=/mnt/sdb/ruilin/Final-Release/Qwen2.5-Coder-7B/Selector
        ;;
    Llama-3.1-8B)
        GLOBAL_MODEL=/mnt/sdb/ruilin/Final-Release/Llama-3.1-8B/SchemaLinker
        GENERATOR_MODEL=/mnt/sdb/ruilin/Final-Release/Llama-3.1-8B/Generator
        SELECT_MODEL=/mnt/sdb/ruilin/Final-Release/Llama-3.1-8B/Selector
        ;;
    Qwen2.5-Coder-14B)
        GLOBAL_MODEL=/mnt/sdb/ruilin/Final-Release/Qwen2.5-Coder-14B/SchemaLinker
        GENERATOR_MODEL=/mnt/sdb/ruilin/Final-Release/Qwen2.5-Coder-14B/Generator
        SELECT_MODEL=/mnt/sdb/ruilin/Final-Release/Qwen2.5-Coder-14B/Selector
        ;;
    Qwen2.5-Coder-32B)
        GLOBAL_MODEL=/mnt/sdb/ruilin/Final-Release/Qwen2.5-Coder-32B/SchemaLinker
        GENERATOR_MODEL=/mnt/sdb/ruilin/Final-Release/Qwen2.5-Coder-32B/Generator
        SELECT_MODEL=/mnt/sdb/ruilin/ablation-32B/direct_rerank/Qwen2.5-Coder-32B-Instruct_lr_1e-05_epoch_4.0_seed_8546
        ;;
    *)
        echo "Unknown setting: ${SETTING}" >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARTIFACTS_DIR="${PROJECT_ROOT}/artifacts"

CACHE_DIR="${ARTIFACTS_DIR}/model_cache"
IR_DIR="${ARTIFACTS_DIR}/ir"
INDEX_DIR="${ARTIFACTS_DIR}/value_index"
EVALUATION_DIR="${ARTIFACTS_DIR}/schema_input"
DB_PATH="${ARTIFACTS_DIR}/database_files"

if [ ! -d "${DB_PATH}" ]; then
    DB_PATH="${PROJECT_ROOT}/dataset"
fi

OUTPUT_DIR="./cache/${SETTING}/${CURRENT_BENCHMARK}/$(date +%m%d_%H%M%S)"
AVAILABLE_GPU=2
mkdir -p "${OUTPUT_DIR}"

python inference.py \
    --global-model-path "${GLOBAL_MODEL}" \
    --local-model-path "${LOCAL_MODEL}" \
    --generator-model-path "${GENERATOR_MODEL}" \
    --selector-model-path "${SELECT_MODEL}" \
    --evaluation-benchmark "${CURRENT_BENCHMARK}" \
    --ir-data-dir "${IR_DIR}" \
    --evaluation-dir "${EVALUATION_DIR}" \
    --index-dir "${INDEX_DIR}" \
    --db-base-path "${DB_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --hf-cache-dir "${CACHE_DIR}" \
    --num-generations-per-token 8 \
    --tensor-parallel-size "${AVAILABLE_GPU}" \
    --gpu-memory-utilization 0.95 \
    --cpu-workers 128 \
    --sql-timeout 100 \
    --step3-max-num-seqs 128 \
    --local-files-only
