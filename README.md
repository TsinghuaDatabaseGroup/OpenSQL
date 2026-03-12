A modular text-to-SQL project with a full pipeline for:

1. Dataset preprocessing
2. SQL data augmentation
3. Model training
4. Multi-stage inference

This repository is organized by stage. The root README is a navigation guide.
For detailed usage and arguments, please go to each subdirectory README.

**🚧 Note: This repository is under construction. 🚧**

## subdirectories

- Preprocessing: [`preprocess/README.md`](preprocess/README.md)
- Data augmentation: [`data_augment/README.md`](data_augment/README.md)
- Training: [`training/README.md`](training/README.md)
- Inference: [`infer/README.md`](infer/README.md)
- Artifacts layout: [`artifacts/README.md`](artifacts/README.md)

## Repository Layout

```text
├── preprocess/      # dataset prep, IR generation, schema input construction
├── data_augment/    # schema linking / SQL augmentation / pairwise CoT annotation
├── training/        # stage-wise training entrypoints and launch scripts
├── infer/           # unified inference pipeline
├── schema_utils/    # IR -> schema rendering utilities
├── value_index/     # value embedding and vector index utilities
├── artifacts/       # intermediate files and generated artifacts
└── dataset/         # benchmark datasets (can be a symlink)
```

## Environment Setup

Because vLLM and training stacks have different dependency constraints, we recommend separate environments.

### Training Environment

```bash
conda create -n train_env python=3.12
conda activate train_env
pip install -r requirements-train.txt
conda install ninja
MAX_JOBS=64 pip install flash-attn --no-build-isolation
```

### Inference Environment

```bash
conda create -n eval_env python=3.12
conda activate eval_env
pip install -r requirements-eval.txt
```

## Workflow

1. Prepare datasets

```bash
python preprocess/prepare_datasets.py --all
```

2. Build IR / index / schema input

```bash
python preprocess/schema_to_ir.py --all
python preprocess/build_value_index.py --all --device cuda:0
python preprocess/schema_input.py --bench Spider_dev --device cuda:0
```

3. Generate training data

```bash
python data_augment/schema_linking_augment.py
```

4. Train stage models

```bash
bash training/train.sh global-sft <lr> <epoch> <model_name_or_path>
bash training/train.sh local-linker <lr> <epoch> <model_name_or_path>
bash training/train.sh generator <lr> <epoch> <model_name_or_path>
bash training/train.sh selector <lr> <epoch> <model_name_or_path>
```

5. Run unified inference

```bash
python infer/inference.py --help
# or
bash infer/start_pipeline.sh <BENCHMARK> <SETTING>
```
