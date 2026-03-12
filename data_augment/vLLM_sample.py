#!/usr/bin/env python3
# data_augment/vLLM_sample.py
# Sample SQL queries via vLLM batch inference, validate against SQLite, and structure pairwise data.
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import uuid
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import func_timeout
import sqlparse
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "vllm_sample"

BENCH_DB_ROOTS: dict[str, Path] = {
    "Spider_train": DATASET_ROOT / "spider_data" / "database",
    "BIRD_train": DATASET_ROOT / "BIRD_train" / "train_databases",
}

# ── Prompt template ────────────────────────────────────────────────
NL2SQL_TEMPLATE = """You are an expert SQL engineer. Your task is to write a valid SQLite query to answer the user's question based on the provided database schema.

### Instructions:
1.  Analyze the user's question and the database schema carefully.
2.  Choose the most appropriate SQL approach — such as direct JOINs, Common Table Expressions (WITH ... AS), or nested subqueries — to best answer the question.
3.  Construct a valid SQLite query that accurately answers the question.
4.  Your final output must contain only the SQL query, nothing else.

### Database Schema:
{schema}

### User Question:
{question}
"""

# ── Defaults ────────────────────────────────────────────────────────
DEFAULT_TENSOR_PARALLEL = 4
DEFAULT_GPU_MEM_UTIL = 0.9
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_QUERY_TIMEOUT = 30
DEFAULT_NUM_GENERATIONS = 16
DEFAULT_NUM_WORKERS = 128


# =====================================================================
# Helper Functions
# =====================================================================


def execute_sql(db_path: Path, query: str, timeout: int) -> tuple[str, Any]:
    """Execute a SQL query on a SQLite database with a timeout."""
    try:

        @func_timeout.func_set_timeout(timeout)
        def exec_on_db(path: Path, q: str) -> list:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                cursor = conn.cursor()
                cursor.execute(q)
                return cursor.fetchall()

        result = exec_on_db(db_path, query)
        if isinstance(result, list) and all(isinstance(row, tuple) for row in result):
            result = [list(row) for row in result]
        return "success", result
    except func_timeout.FunctionTimedOut:
        return "error", f"Query timed out after {timeout} seconds."
    except Exception as e:
        return "error", f"Execution error: {e}"


def clean_generated_sql(text: str) -> str:
    """Extract and clean a generated SQL query from model output."""
    text = text.strip()
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    return re.sub(r"\s+", " ", text).strip().strip(";")


def execute_sql_wrapper(args: tuple[str, Path, str, int]) -> tuple[str, str, Any]:
    """Multiprocessing-compatible wrapper for execute_sql."""
    request_id, db_path, query, timeout = args
    status, result = execute_sql(db_path, query, timeout)
    return request_id, status, result


def get_result_key(result: Any) -> str | None:
    """Compute a canonical string key for a SQL result set (order-insensitive)."""
    if not isinstance(result, list):
        return None
    try:
        processed_result = set(tuple(map(str, row)) for row in result)
        return str(sorted(list(processed_result)))
    except Exception:
        return None


# =====================================================================
# CLI
# =====================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample SQL queries via vLLM, validate against SQLite, and structure pairwise data.",
    )

    model = p.add_argument_group("model")
    model.add_argument("--model", type=str, required=True, help="Path or name of the pretrained LLM.")
    model.add_argument("--tensor-parallel-size", type=int, default=DEFAULT_TENSOR_PARALLEL, help="Tensor parallel size for vLLM.")
    model.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEM_UTIL, help="GPU memory utilization for vLLM.")

    proc = p.add_argument_group("processing")
    proc.add_argument("--input", type=Path, required=True, help="Input dev data JSON file.")
    proc.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    proc.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Max new tokens per generation.")
    proc.add_argument("--query-timeout", type=int, default=DEFAULT_QUERY_TIMEOUT, help="SQL query execution timeout in seconds.")
    proc.add_argument("--num-generations", type=int, default=DEFAULT_NUM_GENERATIONS, help="SQL generations per question (1 greedy + K-1 sampling).")
    proc.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Parallel SQL validation workers.")
    return p.parse_args()


# =====================================================================
# Main
# =====================================================================


def main() -> int:
    args = parse_args()

    logger.info("--- Starting SQL sampling & validation ---")

    # ── 1. Load model and tokenizer ──────────────────────────────────
    logger.info("Loading model and tokenizer...")
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ── 2. Prepare inference requests ────────────────────────────────
    logger.info("Preparing inference requests...")
    input_path: Path = args.input
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    dev_dataset: list[dict] = json.loads(input_path.read_text("utf-8"))

    vllm_requests: list[tuple[str, str]] = []
    request_metadata: dict[str, dict] = {}
    dp_info_map: dict[int, dict] = {i: dp for i, dp in enumerate(dev_dataset)}

    for i, dp in enumerate(tqdm(dev_dataset, desc="Preparing inference requests")):
        db_path = BENCH_DB_ROOTS[dp["benchmark"]] / dp["db_id"] / f"{dp['db_id']}.sqlite"
        if not db_path.exists():
            dp_info_map.pop(i, None)
            continue
        prompt = NL2SQL_TEMPLATE.format(schema=dp["dynamic_noised_schema"], question=dp["question_with_evidence"])
        messages = [{"role": "user", "content": prompt}]
        final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        request_id = str(uuid.uuid4())
        vllm_requests.append((final_prompt, request_id))
        request_metadata[request_id] = {
            "dp_idx": i,
            "benchmark": dp["benchmark"],
            "db_id": dp["db_id"],
            "type": "generated",
        }
        gold_sql_request_id = str(uuid.uuid4())
        request_metadata[gold_sql_request_id] = {
            "dp_idx": i,
            "benchmark": dp["benchmark"],
            "db_id": dp["db_id"],
            "type": "gold",
            "query": dp["query"],
        }

    # ── 3. vLLM batch inference ──────────────────────────────────────
    logger.info("Performing batch inference with vLLM...")
    prompts_to_run = [req[0] for req in vllm_requests]
    request_ids_to_run = [req[1] for req in vllm_requests]
    greedy_params = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        stop=[tokenizer.eos_token],
    )
    greedy_outputs = llm.generate(prompts_to_run, greedy_params)
    sampling_outputs: list = []
    num_sampling = args.num_generations - 1
    if num_sampling > 0:
        sampling_params = SamplingParams(
            n=num_sampling,
            temperature=1.2,
            top_k=50,
            top_p=0.9,
            max_tokens=args.max_new_tokens,
            stop=[tokenizer.eos_token],
        )
        sampling_outputs = llm.generate(prompts_to_run, sampling_params)
    all_generations: dict[str, list[str]] = {}
    for i, output in enumerate(greedy_outputs):
        all_generations[request_ids_to_run[i]] = [gen.text for gen in output.outputs]
    for i, output in enumerate(sampling_outputs):
        all_generations.setdefault(request_ids_to_run[i], []).extend([gen.text for gen in output.outputs])

    del llm

    # ── 4. Parallel SQL validation ───────────────────────────────────
    logger.info(f"Validating SQLs in parallel with {args.num_workers} workers...")
    sql_execution_tasks: list[tuple[str, Path, str, int]] = []
    for request_id, generated_texts in all_generations.items():
        meta = request_metadata[request_id]
        db_path = BENCH_DB_ROOTS[meta["benchmark"]] / meta["db_id"] / f"{meta['db_id']}.sqlite"
        for j, text in enumerate(generated_texts):
            sub_request_id = f"{request_id}_{j}"
            cleaned_sql = clean_generated_sql(text)
            sql_execution_tasks.append((sub_request_id, db_path, cleaned_sql, args.query_timeout))
            request_metadata[sub_request_id] = {**meta, "sql": cleaned_sql}
    for req_id, meta in request_metadata.items():
        if meta.get("type") == "gold":
            db_path = BENCH_DB_ROOTS[meta["benchmark"]] / meta["db_id"] / f"{meta['db_id']}.sqlite"
            sql_execution_tasks.append((req_id, db_path, meta["query"], args.query_timeout))
    execution_results: dict[str, tuple[str, Any]] = {}
    with Pool(args.num_workers) as pool:
        pbar = tqdm(pool.imap_unordered(execute_sql_wrapper, sql_execution_tasks), total=len(sql_execution_tasks), desc="Executing SQLs in parallel")
        for req_id, status, result in pbar:
            execution_results[req_id] = (status, result)

    # ── 5. Aggregate results and structure pairwise data ─────────────
    logger.info("Aggregating results to generate data source...")
    raw_results_by_dp: dict[int, dict] = {i: {"generated": [], "gold": None} for i in dp_info_map}
    for req_id, (status, result) in execution_results.items():
        meta = request_metadata.get(req_id)
        if not meta:
            continue
        dp_idx = meta["dp_idx"]
        if meta["type"] == "gold":
            raw_results_by_dp[dp_idx]["gold"] = (status, result)
        elif meta.get("sql"):
            raw_results_by_dp[dp_idx]["generated"].append({"sql": meta["sql"], "status": status, "result": result})

    data_for_pairwise: list[dict] = []

    for dp_idx, dp_data in tqdm(raw_results_by_dp.items(), desc="Structuring data source"):
        gold_info = dp_data.get("gold")
        if not gold_info or gold_info[0] != "success":
            continue
        gold_result = gold_info[1]
        gold_result_key = get_result_key(gold_result)
        if not gold_result_key:
            continue

        # Group generated SQLs by execution result
        execution_groups: dict[str, dict] = {}
        for gen_res in dp_data["generated"]:
            if gen_res["status"] == "success" and len(gen_res["result"]) < 100000:
                result_key = get_result_key(gen_res["result"])
                if not result_key:
                    continue
                if result_key not in execution_groups:
                    execution_groups[result_key] = {"sqls": [], "execution_result": gen_res["result"]}
                execution_groups[result_key]["sqls"].append({"sql": gen_res["sql"]})

        # Deduplicate SQLs within each bucket using sqlparse normalization
        for result_key in execution_groups:
            group_info = execution_groups[result_key]
            unique_sqls: list[dict] = []
            seen_formatted: set[str] = set()
            for sql_info in group_info["sqls"]:
                raw_sql = sql_info["sql"]
                if not raw_sql or not raw_sql.strip():
                    continue
                formatted_sql = sqlparse.format(raw_sql, keyword_case="upper", reindent=False, strip_whitespace=True)
                if formatted_sql not in seen_formatted:
                    unique_sqls.append(sql_info)
                    seen_formatted.add(formatted_sql)
            group_info["sqls"] = unique_sqls

        # Separate correct and incorrect buckets
        correct_bucket: dict = {}
        incorrect_buckets: list[dict] = []

        for key, group_info in execution_groups.items():
            if key == gold_result_key:
                correct_bucket = group_info
            else:
                incorrect_buckets.append(group_info)

        # Fall back to gold SQL if no generated SQL matched
        original_dp = dp_info_map[dp_idx]
        if not correct_bucket:
            correct_bucket = {"sqls": [{"sql": original_dp["query"]}], "execution_result": gold_result}

        # Only emit if we have both correct and incorrect buckets
        if correct_bucket.get("sqls") and incorrect_buckets:
            data_for_pairwise.append(
                {
                    "benchmark": original_dp["benchmark"],
                    "db_id": original_dp["db_id"],
                    "question": original_dp["question"],
                    "question_with_evidence": original_dp["question_with_evidence"],
                    "dynamic_noised_schema": original_dp["dynamic_noised_schema"],
                    "query": original_dp["query"],
                    "correct_bucket": correct_bucket,
                    "incorrect_buckets": incorrect_buckets,
                }
            )

    # ── Summary & save ───────────────────────────────────────────────
    logger.success("Data source generation complete.")
    logger.success(f"Generated {len(data_for_pairwise)} structured datapoints ready for pairwise creation.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_structured.json"
    output_path.write_text(json.dumps(data_for_pairwise, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Data source saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
