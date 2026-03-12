#!/usr/bin/env python3
# preprocess/build_value_index.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_BENCHES = ("Spider_train", "Spider_dev", "Spider_test", "BIRD_train", "BIRD_dev")
DEFAULT_MODEL_NAME = "Alibaba-NLP/gte-large-en-v1.5"
DEFAULT_CACHE_FOLDER = PROJECT_ROOT / "artifacts/model_cache"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/value_index"
DATASET_ROOT = PROJECT_ROOT / "dataset"
MODEL_LOAD_MODES = ("auto", "local", "online")

BENCH_DB_ROOTS: dict[str, Path] = {
    "Spider_train": DATASET_ROOT / "spider_data/database",
    "Spider_dev": DATASET_ROOT / "spider_data/database",
    "Spider_test": DATASET_ROOT / "spider_data/test_database",
    "BIRD_train": DATASET_ROOT / "BIRD_train/train_databases",
    "BIRD_dev": DATASET_ROOT / "BIRD_dev/dev_databases",
}


def discover_db_ids(db_root: Path) -> list[str]:
    db_ids: list[str] = []
    for child in sorted(db_root.iterdir()):
        if not child.is_dir():
            continue
        db_id = child.name
        sqlite_path = child / f"{db_id}.sqlite"
        if sqlite_path.exists():
            db_ids.append(db_id)
    return db_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build value indexes under artifacts/value_index for supported benchmarks.")
    select_group = parser.add_mutually_exclusive_group(required=True)
    select_group.add_argument("--bench", choices=SUPPORTED_BENCHES, help="Build index for one benchmark.")
    select_group.add_argument("--all", action="store_true", help="Build indexes for all supported benchmarks.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="SentenceTransformer model name or local path.")
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_CACHE_FOLDER, help="Local model cache directory.")
    parser.add_argument(
        "--model-load-mode",
        choices=MODEL_LOAD_MODES,
        default="auto",
        help="Embedding model load mode: auto=try local cache then online download; local=only local files; online=allow online download directly.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Embedding device, e.g., cuda:0 or cpu.")
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


def main() -> int:
    args = parse_args()
    benches = SUPPORTED_BENCHES if args.all else (args.bench,)

    try:
        from value_index.build_index import embed_values_in_db
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("value_index dependencies are required (e.g., faiss).") from exc

    emb_model = load_embedding_model(
        model_name=args.model_name,
        cache_folder=args.cache_folder,
        device=args.device,
        model_load_mode=args.model_load_mode,
    )

    DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    failed: list[tuple[str, str, str]] = []
    total_built = 0

    for bench in benches:
        db_root = BENCH_DB_ROOTS[bench]
        if not db_root.exists():
            failed.append((bench, "-", f"Database root not found: {db_root}"))
            print(f"[{bench}] Skip: database root not found: {db_root}")
            continue

        db_ids = discover_db_ids(db_root)
        if len(db_ids) == 0:
            failed.append((bench, "-", f"No <db_id>.sqlite found under {db_root}"))
            print(f"[{bench}] Skip: no <db_id>.sqlite found under {db_root}")
            continue

        print(f"[{bench}] Found {len(db_ids)} databases in {db_root}")
        for db_id in db_ids:
            try:
                out_path = embed_values_in_db(
                    bench=bench,
                    db_base_path=db_root,
                    db_id=db_id,
                    embed_model=emb_model,
                    output_root=DEFAULT_OUTPUT_ROOT,
                )
                print(f"[{bench}] Built: {db_id} -> {out_path}")
                total_built += 1
            except Exception as exc:
                failed.append((bench, db_id, str(exc)))
                print(f"[{bench}] Failed: {db_id} ({exc})")

    print(f"Total built index files: {total_built}")
    if failed:
        print("Failures:")
        for bench, db_id, message in failed:
            print(f"  - bench={bench}, db_id={db_id}, error={message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
