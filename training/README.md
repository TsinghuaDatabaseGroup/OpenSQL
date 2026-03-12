# Training Pipeline

All launch scripts can be run from any working directory and default to paths under `artifacts/`.

## Layout

- `training/train.sh`: unified stage entrypoint
- `training/config/SFT.json`: accelerate config for SFT training
- `training/config/DPO.json`: accelerate config for DPO training
- `training/schema_linking/SFT/launch_global_sft.sh`
- `training/schema_linking/DPO/launch_global_dpo.sh`
- `training/schema_linking/Local/launch_local.sh`
- `training/generation/launch_generation.sh`
- `training/selector/launch_selector.sh`

## Default Paths

If not overridden by arguments or environment variables, scripts use:

- model cache: `artifacts/model_cache`
- training data base: `artifacts/training_data`
- trained model outputs: `artifacts/trained_models/<stage>`
- accelerate config:
  - SFT stages: `training/config/SFT.json`
  - DPO stage: `training/config/DPO.json`

## Training Data

Put training JSON files under:

- `artifacts/training_data/`

Recommended layout:

```text
artifacts/
  training_data/
    global_schema_linking_sft.json
    global_schema_linking_dpo.json
    local_schema_linking.json
    generation.json
    selector.json
```

Stage-to-file mapping:

- `global-sft` -> `artifacts/training_data/global_schema_linking_sft.json`
- `global-dpo` -> `artifacts/training_data/global_schema_linking_dpo.json`
- `local-linker` -> `artifacts/training_data/local_schema_linking.json`
- `generator` / `generation` -> `artifacts/training_data/generation.json`
- `selector` -> `artifacts/training_data/selector.json`

If your files are stored elsewhere, pass the path as `[finetune_data_path]`
in stage commands, or set `FINETUNE_DATA_PATH=/your/path/file.json`.

## Unified Entrypoint

```bash
bash training/train.sh <stage> [stage_args...]
```

Supported stages:

- `global-sft`
- `global-dpo`
- `local-linker` (`local`)
- `generator` (`generation`)
- `selector`

## Stage Commands

### 1) Global Schema Linker (SFT)

```bash
bash training/train.sh global-sft <lr> <epoch> <model_name_or_path> \
  [finetune_data_path] [storage_dir]
```

Default `finetune_data_path`:

- `artifacts/training_data/global_schema_linking_sft.json`

### 2) Global Schema Linker (DPO)

```bash
bash training/train.sh global-dpo <lr> <beta> <rpo_alpha> <epoch> <sft_model_path> \
  [finetune_data_path] [storage_dir]
```

Default `finetune_data_path`:

- `artifacts/training_data/global_schema_linking_dpo.json`

### 3) Local Schema Linker

```bash
bash training/train.sh local-linker <lr> <epoch> <model_name_or_path> \
  [finetune_data_path] [storage_dir] [eval_steps]
```

Default `finetune_data_path`:

- `artifacts/training_data/local_schema_linking.json`

### 4) SQL Generator

```bash
bash training/train.sh generator <lr> <epoch> <model_name_or_path> \
  [finetune_data_path] [storage_dir]
```

Default `finetune_data_path`:

- `artifacts/training_data/generation.json`

### 5) Selector

```bash
bash training/train.sh selector <lr> <epoch> <model_name_or_path> \
  [finetune_data_path] [storage_dir]
```

Default `finetune_data_path`:

- `artifacts/training_data/selector.json`

## Environment Overrides

Each launch script supports the following optional environment variables:

- `MODEL_CACHE_DIR`
- `FINETUNE_DATA_PATH`
- `MODEL_STORAGE_DIR`
- `ACCELERATE_CONFIG`
- `NUM_PROCESSES`
- `ATTN_IMPL` (default: `flash_attention_2`)
- `TORCH_DTYPE` (default: `bfloat16`)
- `EVAL_STEPS` (local linker only, default: `200`)

Example:

```bash
MODEL_CACHE_DIR=/mnt/models \
NUM_PROCESSES=8 \
bash training/train.sh generator 1e-5 3 /mnt/models/Qwen2.5-Coder-7B-Instruct
```
