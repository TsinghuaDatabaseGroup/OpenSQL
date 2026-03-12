"""Build schema benchmark inputs with IR2Schema.

Example:
    python preprocess/schema_input.py --bench Spider_train --device cuda:7
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
else:
    SentenceTransformer = Any

ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DEFAULT_MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"
DEFAULT_CACHE_FOLDER = ARTIFACTS_ROOT / "model_cache"
DEFAULT_IR_DIR = ARTIFACTS_ROOT / "ir"
DEFAULT_INDEX_ROOT = ARTIFACTS_ROOT / "value_index"
DEFAULT_OUTPUT_DIR = ARTIFACTS_ROOT / "schema_input"
MODEL_LOAD_MODES = ("auto", "local", "online")

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset"


@dataclass(frozen=True)
class BenchmarkSpec:
    dataset_path: Path
    ir_path: Path
    index_dir: Path


def pick_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_default_specs() -> dict[str, BenchmarkSpec]:
    return {
        "Spider_train": BenchmarkSpec(
            dataset_path=DEFAULT_DATASET_ROOT / "spider_data/train.json",
            ir_path=pick_path(
                DEFAULT_IR_DIR / "Spider_train.json",
                DEFAULT_IR_DIR / "Spider.json",
                DEFAULT_IR_DIR / "Spider_ir.json",
            ),
            index_dir=DEFAULT_INDEX_ROOT / "Spider_train",
        ),
        "Spider_dev": BenchmarkSpec(
            dataset_path=DEFAULT_DATASET_ROOT / "spider_data/dev.json",
            ir_path=pick_path(
                DEFAULT_IR_DIR / "Spider_dev.json",
                DEFAULT_IR_DIR / "Spider.json",
                DEFAULT_IR_DIR / "Spider_ir.json",
            ),
            index_dir=DEFAULT_INDEX_ROOT / "Spider_dev",
        ),
        "Spider_test": BenchmarkSpec(
            dataset_path=DEFAULT_DATASET_ROOT / "spider_data/test.json",
            ir_path=pick_path(
                DEFAULT_IR_DIR / "Spider_test.json",
                DEFAULT_IR_DIR / "Spider_test_ir.json",
            ),
            index_dir=DEFAULT_INDEX_ROOT / "Spider_test",
        ),
        "BIRD_train": BenchmarkSpec(
            dataset_path=DEFAULT_DATASET_ROOT / "BIRD_train/train.json",
            ir_path=pick_path(
                DEFAULT_IR_DIR / "BIRD_train.json",
                DEFAULT_IR_DIR / "BIRD_train_ir.json",
            ),
            index_dir=DEFAULT_INDEX_ROOT / "BIRD_train",
        ),
        "BIRD_dev": BenchmarkSpec(
            dataset_path=DEFAULT_DATASET_ROOT / "BIRD_dev/dev.json",
            ir_path=pick_path(
                DEFAULT_IR_DIR / "BIRD_dev.json",
                DEFAULT_IR_DIR / "BIRD_dev_ir.json",
            ),
            index_dir=DEFAULT_INDEX_ROOT / "BIRD_dev",
        ),
        "KaggleDBQA": BenchmarkSpec(
            dataset_path=DEFAULT_DATASET_ROOT / "KaggleDBQA/KaggleDBQA_test.json",
            ir_path=pick_path(
                DEFAULT_IR_DIR / "KaggleDBQA.json",
                DEFAULT_IR_DIR / "KaggleDBQA_ir.json",
            ),
            index_dir=DEFAULT_INDEX_ROOT / "KaggleDBQA",
        ),
        "MIMIC": BenchmarkSpec(
            dataset_path=DEFAULT_DATASET_ROOT / "MIMIC/MIMIC_test.json",
            ir_path=pick_path(
                DEFAULT_IR_DIR / "MIMIC.json",
                DEFAULT_IR_DIR / "MIMIC_ir.json",
            ),
            index_dir=DEFAULT_INDEX_ROOT / "MIMIC",
        ),
        "science": BenchmarkSpec(
            dataset_path=DEFAULT_DATASET_ROOT / "science/science_dev.json",
            ir_path=pick_path(
                DEFAULT_IR_DIR / "science.json",
                DEFAULT_IR_DIR / "science_ir.json",
            ),
            index_dir=DEFAULT_INDEX_ROOT / "science",
        ),
        "spider2_sqlite": BenchmarkSpec(
            dataset_path=DEFAULT_DATASET_ROOT / "spider2_sqlite/spider2_sqlite.json",
            ir_path=pick_path(
                DEFAULT_IR_DIR / "spider2_sqlite.json",
                DEFAULT_IR_DIR / "spider2_sqlite_ir.json",
            ),
            index_dir=DEFAULT_INDEX_ROOT / "spider2_sqlite",
        ),
    }


def import_ir2schema():
    from schema_utils import IR2Schema

    return IR2Schema


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_index_file(path: Path) -> dict[tuple[str, str], Any]:
    with path.open("rb") as file:
        return pickle.load(file)


def load_indexes(index_dir: Path) -> dict[str, dict[tuple[str, str], Any]]:
    indexes: dict[str, dict[tuple[str, str], Any]] = {}
    for pkl_path in sorted(index_dir.glob("*.pkl")):
        indexes[pkl_path.stem] = load_index_file(pkl_path)
    return indexes


def extract_question(datapoint: dict[str, Any]) -> str:
    if isinstance(datapoint.get("question_with_evidence"), str) and datapoint["question_with_evidence"].strip():
        return datapoint["question_with_evidence"]
    if isinstance(datapoint.get("question"), str) and datapoint["question"].strip():
        return datapoint["question"]
    raise ValueError(f"No valid question found in datapoint with db_id={datapoint.get('db_id')!r}")


def resolve_spec(
    bench: str,
    dataset_path: Path | None,
    ir_path: Path | None,
    index_dir: Path | None,
) -> BenchmarkSpec:
    specs = build_default_specs()
    if bench not in specs:
        valid = ", ".join(sorted(specs))
        raise ValueError(f"Unknown benchmark: {bench}. Available benchmarks: {valid}")

    default_spec = specs[bench]
    return BenchmarkSpec(
        dataset_path=dataset_path or default_spec.dataset_path,
        ir_path=ir_path or default_spec.ir_path,
        index_dir=index_dir or default_spec.index_dir,
    )


def generate_schema_data(
    bench: str,
    spec: BenchmarkSpec,
    emb_model: SentenceTransformer,
    print_contain_null: bool = False,
) -> list[dict[str, Any]]:
    if not spec.dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {spec.dataset_path}")
    if not spec.ir_path.exists():
        raise FileNotFoundError(f"IR file not found: {spec.ir_path}")
    if not spec.index_dir.exists():
        raise FileNotFoundError(f"Index directory not found: {spec.index_dir}")

    dataset: list[dict[str, Any]] = load_json(spec.dataset_path)
    ir_set: list[dict[str, Any]] = load_json(spec.ir_path)
    indexes = load_indexes(spec.index_dir)

    ir_by_db_id: dict[str, dict[str, Any]] = {}
    for ir in ir_set:
        ir_by_db_id[ir["db_id"]] = ir

    print(f"Generating schema for {bench}")
    print(f"Dataset path: {spec.dataset_path}")
    print(f"IR path: {spec.ir_path}")
    print(f"Index directory: {spec.index_dir}")

    schema_data: list[dict[str, Any]] = []
    ir2schema_cls = import_ir2schema()
    for datapoint in tqdm(dataset, desc=f"Processing {bench}", total=len(dataset)):
        db_id = datapoint["db_id"]

        if db_id not in ir_by_db_id:
            raise KeyError(f"IR not found for db_id={db_id!r}")

        tindex = indexes.get(db_id)
        if tindex is None:
            raise KeyError(f"Value index not found for db_id={db_id!r} under {spec.index_dir}")

        converter = ir2schema_cls(
            ir=ir_by_db_id[db_id],
            chosen=None,
            tindex=tindex,
            question=extract_question(datapoint),
            emb_model=emb_model,
            print_contain_null=print_contain_null,
        )
        schema_text, _ = converter.render_schema()

        output_datapoint = deepcopy(datapoint)
        output_datapoint["schema"] = schema_text
        schema_data.append(output_datapoint)

    print(f"Total datapoints written: {len(schema_data)}")
    return schema_data


def parse_args() -> argparse.Namespace:
    default_benchmarks = sorted(build_default_specs().keys())

    parser = argparse.ArgumentParser(description="Generate benchmark input data with schema rendered from IR2Schema.")
    parser.add_argument("--bench", required=True, choices=default_benchmarks, help="Benchmark split name.")
    parser.add_argument("--dataset-path", type=Path, default=None, help="Override benchmark dataset path.")
    parser.add_argument("--ir-path", type=Path, default=None, help="Override benchmark IR path.")
    parser.add_argument("--index-dir", type=Path, default=None, help="Override benchmark index directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output JSON.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="SentenceTransformer model name or local path.")
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_CACHE_FOLDER, help="Model cache folder.")
    parser.add_argument(
        "--model-load-mode",
        choices=MODEL_LOAD_MODES,
        default="auto",
        help="Embedding model load mode: auto=try local cache then online download; local=only local files; online=allow online download directly.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Embedding device, e.g., cuda:0 or cpu.")
    parser.add_argument("--print-contain-null", action="store_true", help="Use full column definitions (including nullable details) in output schema.")
    return parser.parse_args()


def load_embedding_model(model_name: str, cache_folder: Path, device: str, model_load_mode: str):
    try:
        from sentence_transformers import SentenceTransformer as STModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("sentence_transformers is required. Install it or run in an environment that provides it.") from exc

    cache_folder.mkdir(parents=True, exist_ok=True)

    if model_load_mode == "auto":
        load_attempts = [True, False]
    elif model_load_mode == "local":
        load_attempts = [True]
    else:
        load_attempts = [False]

    last_error: Exception | None = None
    for local_files_only in load_attempts:
        mode_desc = "local-only" if local_files_only else "online-enabled"
        print(f"Loading embedding model ({mode_desc}): {model_name}")
        try:
            return STModel(
                model_name,
                trust_remote_code=True,
                cache_folder=str(cache_folder),
                local_files_only=local_files_only,
                device=device,
            )
        except Exception as exc:
            last_error = exc
            print(f"Model load failed ({mode_desc}): {exc}")

    raise RuntimeError(
        "Failed to load embedding model. You can either allow online download "
        "(--model-load-mode auto/online) or manually download the model and run "
        "with --model-name <local_model_path> --model-load-mode local."
    ) from last_error


def main() -> None:
    args = parse_args()
    spec = resolve_spec(
        bench=args.bench,
        dataset_path=args.dataset_path,
        ir_path=args.ir_path,
        index_dir=args.index_dir,
    )

    emb_model = load_embedding_model(
        model_name=args.model_name,
        cache_folder=args.cache_folder,
        device=args.device,
        model_load_mode=args.model_load_mode,
    )

    schema_data = generate_schema_data(
        bench=args.bench,
        spec=spec,
        emb_model=emb_model,
        print_contain_null=args.print_contain_null,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.bench}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(schema_data, file, ensure_ascii=False, indent=2)

    print(f"Prepared datapoints: {len(schema_data)}")
    print(f"Schema dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
