## Outline

The following are the preprocessing steps required to run this project (or to run it on a new dataset).

1. Prepare dataset layout
2. Build Vector Indexes for the DB contents.
3. Generate IR (intermediate-representation) for the DB schemas.
4. Prepare schema input for running the datasets.


## (1) Dataset Preparation

> Process the downloaded official dataset archive into a file organization structure usable by this project.

Take Spider and BIRD as examples, please organize the files into the following directory tree.

```
OpenSQL
├── dataset
│   ├── dev.zip         // Official BIRD-dev set 
│   ├── train.zip       // Official BIRD-train set
│   └── spider_data.zip // Official Spider dataset
├── ...
```

Use the `preprocess/prepare_datasets.py` script to prepare datasets. This script will extract the downloaded data, rename it, and clean up unnecessary files.

Examples:

```bash
python preprocess/prepare_datasets.py --dataset BIRD_dev # Prepare BIRD_dev
python preprocess/prepare_datasets.py --dataset BIRD_train # Prepare BIRD_train
python preprocess/prepare_datasets.py --dataset Spider # Prepare Spider
python preprocess/prepare_datasets.py --all # Prepare all datasets
```

## (2) Build Vector Index

Build value indexes before running `schema_input.py`. This step reads database contents and writes vector indexes to:

```
artifacts/value_index/<bench>/<db_id>.pkl
```

Use `preprocess/build_value_index.py`.

Examples:

```bash
python preprocess/build_value_index.py --bench Spider_train --device cuda:7 # Build one benchmark
python preprocess/build_value_index.py --all --device cuda:7 # Build all supported benchmarks

# Force local-only model loading (offline)
python preprocess/build_value_index.py --bench Spider_train --device cuda:7 --model-load-mode local

# Force online model loading
python preprocess/build_value_index.py --bench Spider_train --device cuda:7 --model-load-mode online
```

## (3) Generate IR

Read the database and supplementary information to construct an intermediate representation file of the database schema.

Use `preprocess/schema_to_ir.py`.

By default, output files are written to:

```
artifacts/ir/<bench>.json
```

Examples:

```bash
python preprocess/schema_to_ir.py --bench Spider_train
python preprocess/schema_to_ir.py --bench Spider_dev
python preprocess/schema_to_ir.py --bench Spider_test
python preprocess/schema_to_ir.py --bench BIRD_train
python preprocess/schema_to_ir.py --bench BIRD_dev

# Run all fixed benchmarks
python preprocess/schema_to_ir.py --all

# Dr.Spider benchmark
python preprocess/schema_to_ir.py --bench DB_DBcontent_equivalence
python preprocess/schema_to_ir.py --bench DB_schema_abbreviation
python preprocess/schema_to_ir.py --bench DB_schema_synonym
```

## (4) Prepare Schema Input

This step prepares the linearized database schema, and for each user question retrieves from the vector index the database values related to that question.

Use `preprocess/schema_input.py`

```bash
python preprocess/schema_input.py --bench BIRD_dev --device cuda:7 # For BIRD_dev
python preprocess/schema_input.py --bench Spider_dev --device cuda:7 # For Spider_dev
python preprocess/schema_input.py --bench Spider_test --device cuda:7 # For Spider_test

# Local-only model loading (offline)
python preprocess/schema_input.py --bench Spider_dev --device cuda:7 --model-load-mode local
```


## About Embedding Model Loading (Step 2 & 4)

`preprocess/build_value_index.py` and `preprocess/schema_input.py` both use SentenceTransformer embeddings and now support:

- `--model-load-mode auto` (default): try local cache first, then try online download.
- `--model-load-mode local`: only use local files (offline mode).
- `--model-load-mode online`: allow online download directly.

If your environment has internet access, the default `auto` mode is recommended.

If your network is unstable or restricted, manually download the model (for example `Alibaba-NLP/gte-large-en-v1.5`) and use a local path:

```bash
python preprocess/build_value_index.py --bench Spider_train \
  --model-name /path/to/local/gte-large-en-v1.5 --model-load-mode local
```

You can also place downloaded model files under `artifacts/model_cache/` and provide `--cache-folder` if needed.
