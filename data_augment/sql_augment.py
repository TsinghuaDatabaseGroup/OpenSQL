#!/usr/bin/env python3
# data_augment/sql_augment.py
# End-to-end pipeline: SQL variant generation -> validation & self-correction.
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm import deepseek_completion_json_with_backoff  # noqa: E402

DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "sql_augment"

BENCH_DB_ROOTS: dict[str, Path] = {
    "Spider_train": DATASET_ROOT / "spider_data" / "database",
    "BIRD_train": DATASET_ROOT / "BIRD_train" / "train_databases",
}

WRONG_SQL_MARKER = "<|wrong SQL|>"
UNSUITED_SQL_MARKER = "<|unsuited|>"

DEFAULT_GEN_WORKERS = 150
DEFAULT_VAL_WORKERS = 16
DEFAULT_QUERY_TIMEOUT = 30
DEFAULT_MAX_CORRECTION_ATTEMPTS = 2
DEFAULT_MODEL = "deepseek-reasoner"

# =====================================================================
# Phase 1 — SQL Variant Generation
# =====================================================================

GENERATION_PROMPT = """You are a world-class SQL expert and data engineer. Your primary mission is to create high-quality training data for fine-tuning a Text-to-SQL model.

## Context
We are building an advanced Text-to-SQL model. A key component is a "SQL Generator" that we want to train to produce multiple, semantically equivalent SQL queries for a single user question. These different queries represent distinct "reasoning paths." Your annotations will be used directly to teach the model how to generate this diverse yet accurate set of SQL queries.

## Core Definitions: Reasoning Paths
We have defined three mutually exclusive reasoning paths. To classify any SQL query deterministically, please use the following priority order:

1.  **`CTE` (Highest Priority)**: The query uses the `WITH ... AS` syntax.
2.  **`Subquery`**: The query contains a nested `SELECT` statement in clauses like `SELECT`, `FROM`, or `WHERE`, but is not a `CTE`.
3.  **`Normal` (Lowest Priority)**: Any query that does not fall into the above categories, such as a simple single-table query or a direct multi-table `JOIN` query.

## Your Task
Given a database schema, a user question, and a ground-truth SQL query, you must perform two steps:

1.  **Classify**: Accurately determine the reasoning path of the `ground_truth_sql` based on the priority rules.
2.  **Generate Equivalents**: For the other two reasoning paths, generate semantically equivalent (i.e., returns the exact same result) SQL queries, if and only if a high-quality alternative exists.

## Critical Instructions & Constraints
- **Quality Over Quantity**: The goal is to find alternatives that are **equally or more concise, efficient, or elegant**.
- **Avoid Unnecessary Complexity**: **DO NOT** generate a query for an alternative path by making it artificially complex. For example, do not rewrite a simple `JOIN` into an unnatural multi-level subqueries; or do not simply SELECT all columns from another subquery. If a path only leads to a longer, less efficient, or less natural query, you must not generate it.
- **Adhere to User Intent**: All generated queries must accurately answer the original user's question.

## Output Format
Please provide your response in the following strict JSON format.

**JSON Field Rules**:
- `ground_truth_type`: Your classification of the provided `ground_truth_sql`.
- `Normal`, `CTE`, `Subquery`:
    - The field corresponding to the `ground_truth_type` **must be an empty string** `""`.
    - For any other path where you cannot find a high-quality, natural equivalent, the field **must be an empty string** `""`.
    - Otherwise, provide the generated equivalent SQL query as a string.

# Example Output
{{
    \"ground_truth_type\": \"Subquery\",
    \"Normal\": \"SELECT T1.name FROM employees AS T1 JOIN departments AS T2 ON T1.department_id = T2.id WHERE T2.name != 'Sales'\",
    \"CTE\": \"\",
    \"Subquery\": \"\",
}}

Your Turn: Start Analysis
Database Schema:
{schema}
User Question:
{question}
Ground Truth SQL Query:
{ground_truth_sql}
"""


def _expand_one(dp_item: dict, *, model: str) -> dict:
    """Call LLM to classify the GT SQL and generate equivalent variants."""
    schema = dp_item["dynamic_noised_schema"]
    question = dp_item["question_with_evidence"]
    ground_truth_sql = dp_item["query"]
    prompt = GENERATION_PROMPT.format(schema=schema, question=question, ground_truth_sql=ground_truth_sql)

    response = deepseek_completion_json_with_backoff(prompt, model=model, temperature=0.0)
    response_json = json.loads(response)

    dp_item_copy = deepcopy(dp_item)
    for k, v in response_json.items():
        dp_item_copy[k] = v
    return dp_item_copy


def run_generation(
    input_data: list[dict],
    *,
    output_path: Path,
    max_workers: int = DEFAULT_GEN_WORKERS,
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    """Phase 1: generate SQL variants for all datapoints. Returns the expanded list."""
    logger.info(f"Phase 1: generating SQL variants for {len(input_data)} datapoints (workers={max_workers})...")

    all_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_expand_one, dp, model=model) for dp in input_data]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            try:
                result_dp = future.result(timeout=60)
                all_results.append(result_dp)
                if len(all_results) % 10 == 0 or len(all_results) >= len(input_data) - 10:
                    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
            except concurrent.futures.TimeoutError:
                logger.warning(f"Timeout: {future.exception()}")
            except Exception as exc:
                logger.warning(f"An item generated an exception: {exc}")

    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Phase 1 done: {len(all_results)}/{len(input_data)} expanded -> {output_path}")
    return all_results


# =====================================================================
# Phase 2 — Validation & Self-Correction
# =====================================================================

CORRECTION_PROMPT = """You are a world-class SQL expert specializing in debugging and correcting SQL queries. Your task is to fix an incorrect SQL query so that it becomes semantically equivalent to a provided ground-truth query.

## Context
We are trying to generate multiple, equivalent SQL queries for a single question. We have a specific "reasoning path" (e.g., using a CTE, a Subquery, etc.) in mind for the new query. An attempt was made to generate a query for this path, but it failed. It either produced a syntax error, timed out, or returned results that did not match the ground-truth query.

## Core Definitions: Reasoning Paths
To ensure the corrected query follows the right structure, please adhere to these definitions, using the specified priority order:
1.  **`CTE` (Highest Priority)**: The query uses the `WITH ... AS` syntax.
2.  **`Subquery`**: The query contains a nested `SELECT` statement in clauses like `SELECT`, `FROM`, or `WHERE`, but is not a `CTE`.
3.  **`Normal` (Lowest Priority)**: Any query that does not fall into the above categories, such as a simple single-table query or a direct multi-table `JOIN` query.

## Your Task
Your goal is to analyze the provided information and generate a **new, corrected SQL query** that:
1.  Strictly follows the **Target Reasoning Path** as defined above.
2.  Is semantically equivalent to the `ground_truth_sql` (i.e., returns the exact same result).
3.  Fixes the errors present in the previous failed attempts.
4.  If you cannot generate a high-quality query for the target path, return an empty string `""` instead.

## Critical Instructions & Constraints
- **Quality Over Quantity**: The goal is to find SQL that are **equally or more concise, efficient, or elegant than the ground-truth SQL**.
- **Avoid Unnecessary Complexity**: **DO NOT** generate a query for an alternative path by making it artificially complex. For example, do not rewrite a simple `JOIN` into an unnatural multi-level subqueries; or do not simply SELECT all columns from another subquery. If a path only leads to a longer, less efficient, or less natural query, you must not generate it.
- **Adhere to User Intent**: All generated queries must accurately answer the original user's question.

## Information Provided
- **Database Schema**: The structure of the database.
- **User Question**: The original natural language question.
- **Ground-Truth SQL**: A verified, correct SQL query that provides the target result.
- **Target Reasoning Path**: The specific SQL structure you MUST use (e.g., `CTE`, `Subquery`, `Normal`).
- **History of Failed Attempts**: A list of previously generated queries for this path that were incorrect. Analyze them to avoid repeating the same mistakes.

## Output Format
Please provide your response in the following strict JSON format. Do not add any explanations.

{{
    "corrected_sql": "YOUR NEWLY GENERATED AND CORRECTED SQL QUERY HERE"
}}

--- Begin Analysis ---

### Database Schema
{schema}

### User Question
{question}

### Ground-Truth SQL (Correct Answer)
```sql
{ground_truth_sql}
```

### Target Reasoning Path to Fix
`{target_path}`

### History of Failed Attempts for this Path
{failed_attempts_str}

Now, generate the corrected SQL.
"""


def _execute_query(db_path: Path, query: str) -> list:
    """Execute a SQL query on the given SQLite database (read-only)."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found at: {db_path}")

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode(errors="ignore")
    cursor = con.cursor()
    try:
        cursor.execute(query)
        return cursor.fetchall()
    finally:
        cursor.close()
        con.close()


def _execute_in_process(db_path: Path, query: str, queue: multiprocessing.Queue) -> None:
    """Worker target: execute query and put result (or exception) into *queue*."""
    try:
        queue.put(_execute_query(db_path, query))
    except Exception as e:
        queue.put(e)


def _execute_with_timeout(db_path: Path, query: str, timeout: int) -> list:
    """Execute a query in a subprocess with a hard timeout."""
    q: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_execute_in_process, args=(db_path, query, q))

    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        logger.debug(f"Query timed out. Terminating process {process.pid}...")
        process.terminate()
        process.join()
        raise TimeoutError(f"Query timed out after {timeout} seconds.")

    result = q.get()
    if isinstance(result, Exception):
        raise type(result)(f"Execution failed in subprocess: {result}") from result
    return result


def _results_equal(result_a: list, result_b: list) -> bool:
    """Compare two SQL result sets (order-insensitive)."""
    if len(result_a) != len(result_b):
        return False
    return Counter(result_a) == Counter(result_b)


def _get_corrected_sql(prompt: str, *, model: str) -> str | None:
    """Call LLM to get a corrected SQL query."""
    try:
        response_str = deepseek_completion_json_with_backoff(prompt, model=model, temperature=0.0)
        return json.loads(response_str).get("corrected_sql")
    except Exception as e:
        logger.warning(f"Failed to get corrected SQL from LLM: {e}")
        return None


def _validate_one(
    datapoint: dict,
    *,
    query_timeout: int,
    max_corrections: int,
    model: str,
) -> dict:
    """Validate a single datapoint and iteratively correct wrong SQL variants."""
    db_id = datapoint.get("db_id")
    benchmark = datapoint.get("benchmark")
    ground_truth_sql = datapoint.get("query")
    ground_truth_type = datapoint.get("ground_truth_type")

    validated_dp = deepcopy(datapoint)

    if not all([db_id, benchmark, ground_truth_sql, ground_truth_type]):
        logger.warning(f"Skipping datapoint due to missing fields: {datapoint.get('question')}")
        return validated_dp

    base_db_path = BENCH_DB_ROOTS.get(benchmark)
    if not base_db_path:
        logger.warning(f"Unknown benchmark '{benchmark}' for db_id '{db_id}'. Skipping.")
        return validated_dp
    db_path = base_db_path / db_id / f"{db_id}.sqlite"

    try:
        ground_truth_result = _execute_with_timeout(db_path, ground_truth_sql, query_timeout)
    except Exception as e:
        logger.error(f"Ground-truth query failed for db_id '{db_id}': {e}")
        for key in ["Normal", "CTE", "Subquery"]:
            validated_dp[key] = WRONG_SQL_MARKER
        return validated_dp

    validated_dp[ground_truth_type] = ground_truth_sql

    for variant_type in ["Normal", "CTE", "Subquery"]:
        if variant_type == ground_truth_type:
            continue

        current_sql = datapoint.get(variant_type)
        failed_attempts: list[str] = []
        is_corrected = False

        for attempt in range(max_corrections + 1):
            if not isinstance(current_sql, str) or not current_sql.strip():
                validated_dp[variant_type] = UNSUITED_SQL_MARKER
                break

            try:
                variant_result = _execute_with_timeout(db_path, current_sql, query_timeout)
                if _results_equal(ground_truth_result, variant_result):
                    validated_dp[variant_type] = current_sql
                    is_corrected = True
                    logger.debug(f"SUCCESS: db_id '{db_id}', type '{variant_type}' corrected at attempt {attempt}.")
                    break
                else:
                    raise ValueError("Result mismatch with ground-truth.")
            except Exception as e:
                logger.debug(f"Attempt {attempt} FAILED for db_id '{db_id}', type '{variant_type}': {e}")
                failed_attempts.append(f"-- Attempt {len(failed_attempts) + 1} (failed):\n{current_sql}")

                if attempt < max_corrections:
                    prompt = CORRECTION_PROMPT.format(
                        schema=datapoint.get("dynamic_noised_schema", "Schema not available"),
                        question=datapoint.get("question_with_evidence", "Question not available"),
                        ground_truth_sql=ground_truth_sql,
                        target_path=variant_type,
                        failed_attempts_str="\n".join(failed_attempts),
                    )
                    new_sql = _get_corrected_sql(prompt, model=model)
                    if new_sql and new_sql.strip():
                        current_sql = new_sql
                    else:
                        logger.debug(f"LLM correction returned empty for db_id '{db_id}', type '{variant_type}'.")
                        break
                else:
                    logger.debug(f"All {max_corrections + 1} attempts failed for db_id '{db_id}', type '{variant_type}'.")

        if not is_corrected and validated_dp.get(variant_type) != UNSUITED_SQL_MARKER:
            validated_dp[variant_type] = WRONG_SQL_MARKER

    return validated_dp


def run_validation(
    input_data: list[dict],
    *,
    output_path: Path,
    max_workers: int = DEFAULT_VAL_WORKERS,
    query_timeout: int = DEFAULT_QUERY_TIMEOUT,
    max_corrections: int = DEFAULT_MAX_CORRECTION_ATTEMPTS,
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    """Phase 2: validate all datapoints against SQLite DBs. Returns the validated list."""
    logger.info(f"Phase 2: validating {len(input_data)} datapoints (workers={max_workers}, timeout={query_timeout}s, max_corrections={max_corrections})...")

    processed_results: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_dp = {
            executor.submit(
                _validate_one,
                dp,
                query_timeout=query_timeout,
                max_corrections=max_corrections,
                model=model,
            ): dp
            for dp in input_data
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_dp), total=len(input_data), desc="Validating"):
            try:
                processed_results.append(future.result())
                if len(processed_results) % 10 == 0:
                    output_path.write_text(json.dumps(processed_results, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:
                dp_question = future_to_dp[future].get("question", "N/A")
                logger.error(f"Unexpected error processing '{dp_question}': {exc}")

    output_path.write_text(json.dumps(processed_results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Phase 2 done: {len(processed_results)}/{len(input_data)} validated -> {output_path}")
    return processed_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end pipeline: generate SQL variants via LLM, then validate & self-correct against SQLite.",
    )
    p.add_argument("--input", type=Path, required=True, help="Input JSON file (output of schema_input pipeline).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="LLM model name.")

    gen = p.add_argument_group("generation (Phase 1)")
    gen.add_argument("--gen-workers", type=int, default=DEFAULT_GEN_WORKERS, help="Max concurrent LLM requests for generation.")

    val = p.add_argument_group("validation (Phase 2)")
    val.add_argument("--val-workers", type=int, default=DEFAULT_VAL_WORKERS, help="Max parallel validation workers.")
    val.add_argument("--query-timeout", type=int, default=DEFAULT_QUERY_TIMEOUT, help="SQL query timeout in seconds.")
    val.add_argument("--max-corrections", type=int, default=DEFAULT_MAX_CORRECTION_ATTEMPTS, help="Max LLM correction attempts per variant.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_path: Path = args.input
    output_dir: Path = args.output_dir

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    expanded_path = output_dir / f"{input_path.stem}_expanded.json"
    validated_path = output_dir / f"{input_path.stem}_validated.json"

    # ── Load input ───────────────────────────────────────────────────
    input_data: list[dict] = json.loads(input_path.read_text("utf-8"))
    logger.info(f"Loaded {len(input_data)} datapoints from {input_path}")

    # ── Phase 1: generate SQL variants ───────────────────────────────
    expanded = run_generation(
        input_data,
        output_path=expanded_path,
        max_workers=args.gen_workers,
        model=args.model,
    )

    # ── Phase 2: validate & self-correct ─────────────────────────────
    run_validation(
        expanded,
        output_path=validated_path,
        max_workers=args.val_workers,
        query_timeout=args.query_timeout,
        max_corrections=args.max_corrections,
        model=args.model,
    )

    logger.success(f"All done -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
