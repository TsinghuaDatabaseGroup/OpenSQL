# Inference Pipeline

`infer/inference.py` is the unified end-to-end inference pipeline.

It executes the full workflow in one process:

1. Global schema linking
2. Local schema linking
3. SQL generation
4. Pairwise rerank

## Entrypoints

Direct:

```bash
python inference.py ...
```

Wrapper:

```bash
bash start_pipeline.sh <BENCHMARK> <SETTING>
```

`start_pipeline.sh` uses project-local paths by default:

- `${PROJECT_ROOT}/artifacts/model_cache`
- `${PROJECT_ROOT}/artifacts/ir`
- `${PROJECT_ROOT}/artifacts/value_index`
- `${PROJECT_ROOT}/artifacts/schema_input`
- `${PROJECT_ROOT}/artifacts/database_files` (fallback to `${PROJECT_ROOT}/dataset` if missing)

## Pipeline Data Flow

Input artifacts:

- Dynamic evaluation input JSON
- IR JSON
- Value index files (`*.pkl`)
- SQLite databases
- Four model paths (global/local/generator/selector)

Output artifacts:

- `step0_output.json`
- `step1_output.json`
- `step2_latest_run.json`
- `step2_best_record.json`
- `best_record.json`
- `report-sql-generation-*.txt`
- `step3_results.json`
- `step3_rerank_report-*.txt`

## Quick Start

```bash
python inference.py \
  --global-model-path /path/to/global_model \
  --local-model-path /path/to/local_model \
  --generator-model-path /path/to/generator_model \
  --selector-model-path /path/to/selector_model \
  --evaluation-benchmark Spider_dev \
  --ir-data-dir /path/to/ir \
  --evaluation-dir /path/to/evaluation_input \
  --index-dir /path/to/value_index \
  --db-base-path /path/to/database_files \
  --output-dir /path/to/output
```

## Parameter Reference

All parameters below are from `python inference.py --help`.

### Model Paths

- `--global-model-path` (required): model path for global schema linking.
- `--local-model-path` (required): model path for local schema linking.
- `--generator-model-path` (required): model path for SQL generation.
- `--selector-model-path` (required): model path for pairwise rerank.
- `--embedding-model-name-or-path` (default: `Alibaba-NLP/gte-large-en-v1.5`): embedding model used in local schema linking.

### Data Paths

- `--evaluation-benchmark` (required): benchmark name. Supported patterns in code are base sets like `BIRD_dev`, `Spider_dev`, `Spider_test`, `KaggleDBQA`, `MIMIC`, `science`, and prefixed sets like `DB_*`, `NLQ_*`, `SQL_*`.
- `--ir-data-dir` (required): directory containing IR JSON files.
- `--evaluation-dir` (required): directory containing dynamic evaluation inputs.
- `--index-dir` (required): directory containing value index `.pkl` files.
- `--db-base-path` (required): root directory of benchmark SQLite databases.
- `--output-dir` (required): directory used to save all pipeline outputs and reports.
- `--hf-cache-dir` (default: `None`): Hugging Face cache directory for tokenizer/embedding loading in step1.
- `--embedding-device` (default: `cuda`): device used by the embedding model, for example `cuda`, `cuda:0`, or `cpu`.

### Parallel and vLLM Runtime

- `--tensor-parallel-size` (default: `1`): global tensor parallel size for all stages.
- `--gpu-memory-utilization` (default: `0.9`): global vLLM GPU memory utilization ratio.
- `--step3-max-num-seqs` (default: `256`): vLLM `max_num_seqs` for step3.
- `--cpu-workers` (default: `128`): number of multiprocessing workers for SQL execution checks.
- `--sql-timeout` (default: `100`): timeout (seconds) for executing one SQL query.

### Generation Controls

- `--step0-max-new-tokens` (default: `1024`): max generated tokens per sample in step0.
- `--step0-temperature` (default: `0.0`): sampling temperature in step0.
- `--step1-max-new-tokens` (default: `8`): max generated tokens per sample in step1.
- `--step1-temperature` (default: `0.0`): sampling temperature in step1.
- `--num-generations-per-token` (default: `8`): number of SQL candidates generated per control token in step2.
- `--step2-max-new-tokens` (default: `1024`): max generated tokens per SQL candidate in step2.
- `--step2-temperature` (default: `1.5`): sampling temperature in step2.
- `--step2-top-p` (default: `0.95`): top-p sampling in step2.
- `--step2-top-k` (default: `50`): top-k sampling in step2.
- `--step3-max-new-tokens` (default: `2048`): max generated tokens per pairwise judgement in step3.
- `--fallback-rounds` (default: `3`): maximum fallback rounds in step2 when no valid execution result is found.
- `--seed` (default: `None`): random seed used by Python `random`.

### Local/Online Loading Switches

- `--local-files-only` (default behavior): force local-only loading where supported.
- `--online-enabled`: allow online loading where supported.

Use `python inference.py --help` to view the same options directly from code.
