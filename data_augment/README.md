## Prerequisites

Before running data augmentation, make sure the following prerequisites are satisfied:

1. Dataset files are already prepared under `dataset/` (from `preprocess/prepare_datasets.py`).
2. IR files are generated (mandatory for schema linking augmentation):

   - `artifacts/ir/Spider_train.json`
   - `artifacts/ir/BIRD_train.json`

   You can generate them with:

   ```bash
   python preprocess/schema_to_ir.py --bench Spider_train
   python preprocess/schema_to_ir.py --bench BIRD_train
   ```
3. Python dependencies are installed from `requirements.txt` (including `openai`, `transformers`, `vllm`, and `func-timeout`).
4. For scripts that call external LLM APIs (`sql_augment.py`, `compare_augment.py`), configure credentials in `data_augment/llm.py` before running.

## (1) Schema Linking

Generate schema-linking training data (local + global).

Default command:

```bash
python data_augment/schema_linking_augment.py
```

Default output:

- `artifacts/schema_link/local_schema_linking.json`
- `artifacts/schema_link/global_schema_linking_sft.json`
- `artifacts/schema_link/global_schema_linking_dpo.json`
- `artifacts/schema_link/config.json`

Single benchmark mode writes into a same-name subdirectory:

- `python data_augment/schema_linking_augment.py --bench BIRD_train`
- output dir: `artifacts/schema_link/BIRD_train/`

If you need intermediate files (such as per-benchmark split files), add `--full-output`.

```bash
# Single benchmark
python data_augment/schema_linking_augment.py --bench BIRD_train

# Custom parameters
python data_augment/schema_linking_augment.py --dpo-ratio 0.4 --num-col-samples 5 --seed 0

# Disable table-deletion negative samples
python data_augment/schema_linking_augment.py --no-table-deletion

# Skip noise augmentation
python data_augment/schema_linking_augment.py --no-noise

# Write full intermediate outputs
python data_augment/schema_linking_augment.py --full-output
```

## (2) SQL Augmentation

Generate multiple reasoning-path SQL variants (Normal / CTE / Subquery) via LLM, then validate them against SQLite databases with iterative self-correction.

Output is written to `artifacts/sql_augment`.

Input should be `*_schema_link.json` generated with `--full-output`.

```bash
python data_augment/sql_augment.py --input artifacts/schema_link/Spider_train_schema_link.json

# Custom parameters
python data_augment/sql_augment.py --input artifacts/schema_link/Spider_train_schema_link.json \
  --gen-workers 100 --val-workers 8 --query-timeout 60 --max-corrections 3
```

## (3) SQL Selection

Sample SQL candidates, build pairwise correct/incorrect pairs, then annotate each pair with Chain-of-Thought reasoning.

### vLLM Sampling

Use an LLM to generate multiple SQL candidates per question, validate them against SQLite, and separate them into correct/incorrect buckets.

Output is written to `artifacts/vllm_sample`.

Input should be `*_schema_link.json` generated with `--full-output`.

```bash
python data_augment/vLLM_sample.py \
  --model /path/to/pretrained-model \
  --input artifacts/schema_link/Spider_train_schema_link.json
```

### Pairwise Data Construction

Sample correct + incorrect SQL pairs.

Output is written to `artifacts/pairwise_data`.

```bash
python data_augment/create_pairwise_data.py \
  --input artifacts/vllm_sample/Spider_train_schema_link_structured.json
```

### CoT Annotation

Annotate each pairwise sample with Chain-of-Thought analysis via an LLM.

Output is written to `artifacts/cot_star`.

```bash
python data_augment/compare_augment.py \
  --input artifacts/pairwise_data/Spider_train_schema_link_structured_pairwise.json
```
