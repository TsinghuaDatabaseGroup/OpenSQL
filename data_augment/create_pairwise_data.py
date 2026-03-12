#!/usr/bin/env python3
# data_augment/create_pairwise_data.py
# Build pairwise training pairs from correct/incorrect SQL buckets.
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "pairwise_data"
DEFAULT_MAX_PAIRS = 3


def create_pairwise_data(
    input_path: Path,
    output_path: Path,
    max_pairs_per_item: int,
) -> list[dict]:
    """Read structured data with correct/incorrect buckets and generate pairwise training pairs."""
    logger.info(f"Loading structured data from {input_path}...")
    source_data: list[dict] = json.loads(input_path.read_text("utf-8"))
    logger.info(f"Loaded {len(source_data)} source datapoints")

    final_pairwise_data: list[dict] = []

    for source_item in tqdm(source_data, desc="Generating pairwise data"):
        schema = source_item.get("dynamic_noised_schema")
        question = source_item.get("question_with_evidence")
        if not schema or not question:
            continue

        # Prepare correct SQL pool (sample without replacement)
        correct_bucket = source_item.get("correct_bucket", {})
        correct_result = correct_bucket.get("execution_result")
        available_correct_sqls = list(correct_bucket.get("sqls", []))

        # Prepare incorrect SQL pool from all incorrect buckets
        available_incorrect: list[dict] = []
        for bucket in source_item.get("incorrect_buckets", []):
            bucket_result = bucket.get("execution_result")
            for sql_info in bucket.get("sqls", []):
                available_incorrect.append({"sql_info": sql_info, "result": bucket_result})

        if not available_correct_sqls or not available_incorrect:
            continue

        random.shuffle(available_correct_sqls)
        random.shuffle(available_incorrect)

        # Create pairs via sampling without replacement
        pairs_created = 0
        while pairs_created < max_pairs_per_item and available_correct_sqls and available_incorrect:
            correct_sql_info = available_correct_sqls.pop()
            incorrect_info = available_incorrect.pop()

            c_sql = correct_sql_info.get("sql")
            i_sql = incorrect_info.get("sql_info", {}).get("sql")
            i_result = incorrect_info.get("result")

            if not c_sql or not i_sql:
                continue

            # Randomize SQL1/SQL2 ordering to avoid positional bias
            if random.random() < 0.5:
                sql1, sql2 = c_sql, i_sql
                result1, result2 = correct_result, i_result
                winner = "SQL1"
            else:
                sql1, sql2 = i_sql, c_sql
                result1, result2 = i_result, correct_result
                winner = "SQL2"

            final_pairwise_data.append({
                "dynamic_noised_schema": schema,
                "question_with_evidence": question,
                "sql1": sql1,
                "sql2": sql2,
                "result1": result1,
                "result2": result2,
                "winner": winner,
            })
            pairs_created += 1

    logger.success(f"Created {len(final_pairwise_data)} pairwise training samples")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final_pairwise_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved to: {output_path}")
    return final_pairwise_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build pairwise training pairs from structured correct/incorrect SQL buckets.",
    )
    p.add_argument("--input", type=Path, required=True, help="Input structured data file (output of vLLM_sample pipeline).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    p.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS, help="Max pairs per source datapoint.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    input_path: Path = args.input
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_pairwise.json"

    create_pairwise_data(input_path, output_path, max_pairs_per_item=args.max_pairs)
    logger.success(f"All done -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
