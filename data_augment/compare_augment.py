#!/usr/bin/env python3
# data_augment/compare_augment.py
# Annotate pairwise SQL data with Chain-of-Thought via the STaR (Self-Taught Reasoner) method.
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm import deepseek_completion_json_with_backoff  # noqa: E402

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "cot_star"
DEFAULT_MAX_WORKERS = 150
DEFAULT_CHECKPOINT_INTERVAL = 100

# =====================================================================
# Prompt Template
# =====================================================================

PROMPT_TEMPLATE_PAIRWISE = """You are a world-class SQL analyst. Your task is to meticulously compare two SQL queries based on a user's question and the database schema, then produce a rigorous Chain-of-Thought (CoT) analysis to determine which SQL is correct.

### Your Task:
Generate a response in two parts: a detailed analysis following a strict four-step process, and a final choice. **Crucially, do not make any value judgments (e.g., "correct", "incorrect", "better") in Steps 1, 2, and 3.**

1.  **Question Analysis**: Restate the user's intent. What is the core question? What specific columns or calculated values should the final output contain? This defines the ground truth for evaluation.

2.  **Clause Analysis**: Analyze the clauses (SELECT, FROM, JOIN, WHERE, GROUP BY, etc.) in the two SQL candidates.
    * For SQL 1: Sequentially break down its clauses. Describe what each part does factually.
    * For SQL 2: Do the same for its clauses.
    * This step is for objective observation of syntax and structure.

3.  **Semantic Comparison**: Compare the two SQL queries on a semantic level.
    * What is the overall logic of SQL 1? What story does its result tell?
    * What is the overall logic of SQL 2? What story does its result tell?
    * Identify the key logical differences in how they approach the user's question (e.g., different join conditions, filtering logic, aggregation methods).
    * In addition to SQL structure and logic, you also need to pay attention to the execution results to distinguish which SQL is correct: if the execution result is empty, or there are many None values, it indicates that the SQL may have problems.

4.  **SQL Choice**: Based on the reasoning from the previous steps, make a final decision.
    * Synthesize your findings from the clause and semantic analysis.
    * Compare these findings against the user's intent defined in Step 1.
    * Conclude definitively which SQL query (SQL1 or SQL2) is preferred and provide a clear justification for your choice.

### Output Format:
Return your response in JSON format with the following structure:
{{
    "analysis": "Your detailed, four-step analysis here.",
    "final_choice": "SQL1" or "SQL2"
}}

### Provided Information:

[Database Schema]
{schema}

[User Question]
{question}

---
[SQL Candidate 1]
```sql
{sql1}
```

[Execution Result 1]
{result1}
---
[SQL Candidate 2]
```sql
{sql2}
```

[Execution Result 2]
{result2}
---
{STaR_CORRECT_SQL_NOTE}
"""


# =====================================================================
# Helper Functions
# =====================================================================


def format_sql_results_for_llm(execution_result: Any, max_rows_to_show: int = 5, max_row_length: int = 512) -> str:
    """Format SQL execution results for inclusion in a prompt, truncating long outputs."""
    if not isinstance(execution_result, list):
        return str(execution_result)

    total_rows = len(execution_result)
    if total_rows == 0:
        return "Execution result is empty."

    header = f"Execution result contains {total_rows} rows"
    if total_rows <= max_rows_to_show:
        header += ":"
        rows_to_process = execution_result
    else:
        header += f" (displaying the first {max_rows_to_show} rows):"
        rows_to_process = execution_result[:max_rows_to_show]

    formatted_rows = []
    for row in rows_to_process:
        row_str = str(row)
        if len(row_str) > max_row_length:
            row_str = row_str[:max_row_length] + "..."
        formatted_rows.append(row_str)

    return header + "\n" + "\n".join(formatted_rows)


# =====================================================================
# Core STaR Processing
# =====================================================================


def process_data_point_with_star(data_point: dict[str, Any]) -> dict | None:
    """Apply the two-stage STaR annotation flow to a single pairwise data point."""
    try:
        schema = data_point["dynamic_noised_schema"]
        question = data_point["question_with_evidence"]
        sql1 = data_point["sql1"]
        sql2 = data_point["sql2"]
        result1_str = format_sql_results_for_llm(data_point["result1"])
        result2_str = format_sql_results_for_llm(data_point["result2"])
        correct_winner = data_point["winner"]

        if not all([schema, question, sql1, sql2, correct_winner]):
            logger.warning("Skipping data point due to missing essential fields.")
            return None

        # Stage 1 (Rationalization): let LLM reason without hints
        prompt_stage1 = PROMPT_TEMPLATE_PAIRWISE.format(
            schema=schema,
            question=question,
            sql1=sql1,
            sql2=sql2,
            result1=result1_str,
            result2=result2_str,
            STaR_CORRECT_SQL_NOTE="",
        )

        try:
            response_stage1_raw = deepseek_completion_json_with_backoff(prompt_stage1)
            response_stage1_json = json.loads(response_stage1_raw)
            analysis_stage1 = response_stage1_json["analysis"]
            llm_choice_stage1 = response_stage1_json["final_choice"]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response in Stage 1: {e}. Skipping.")
            return None

        # Check Stage 1 result
        if llm_choice_stage1 and correct_winner.lower() == llm_choice_stage1.lower():
            # Stage 1 succeeded: adopt its CoT with the hint-free prompt
            return {
                "schema": schema,
                "question": question,
                "chosen_sql": data_point[correct_winner.lower()],
                "rejected_sql": data_point["sql2" if correct_winner.lower() == "sql1" else "sql1"],
                "prompt_used": prompt_stage1,
                "cot_analysis": analysis_stage1,
                "final_choice": llm_choice_stage1,
                "star_stage": 1,
            }

        # Stage 2 (Refinement): provide the correct answer as a hint
        star_note = (
            f"### Ground Truth Hint:\n"
            f"After careful review, it has been determined that the correct SQL is **{correct_winner}**. "
            f"Please use this information to guide your four-step analysis, but do not explicitly mention this hint in your response."
        )

        prompt_stage2 = PROMPT_TEMPLATE_PAIRWISE.format(
            schema=schema,
            question=question,
            sql1=sql1,
            sql2=sql2,
            result1=result1_str,
            result2=result2_str,
            STaR_CORRECT_SQL_NOTE=star_note,
        )

        try:
            response_stage2_raw = deepseek_completion_json_with_backoff(prompt_stage2)
            response_stage2_json = json.loads(response_stage2_raw)
            analysis_stage2 = response_stage2_json["analysis"]
            llm_choice_stage2 = response_stage2_json["final_choice"]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response in Stage 2: {e}. Skipping.")
            return None

        # Validate Stage 2 output
        if llm_choice_stage2 and correct_winner.lower() == llm_choice_stage2.lower():
            # Stage 2 succeeded: adopt its CoT but save the hint-free prompt
            return {
                "schema": schema,
                "question": question,
                "chosen_sql": data_point[correct_winner.lower()],
                "rejected_sql": data_point["sql2" if correct_winner.lower() == "sql1" else "sql1"],
                "prompt_used": prompt_stage1,
                "cot_analysis": analysis_stage2,
                "final_choice": llm_choice_stage2,
                "star_stage": 2,
            }
        else:
            # LLM failed even with the hint: discard this sample
            logger.warning(f"STaR Stage 2 failed. LLM choice '{llm_choice_stage2}' != correct '{correct_winner}'. Discarding.")
            return None

    except Exception as e:
        logger.error(f"Unexpected error processing a data point: {e}")
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Annotate pairwise SQL data with CoT via the STaR method.",
    )
    p.add_argument("--input", type=Path, required=True, help="Input pairwise data file (output of create_pairwise_data).")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    p.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Max concurrent LLM requests.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_path: Path = args.input
    output_dir: Path = args.output_dir

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"{input_path.stem}_star_train.json"
    analysis_path = output_dir / f"{input_path.stem}_star_analysis.json"

    # ── Load input ───────────────────────────────────────────────────
    source_data: list[dict] = json.loads(input_path.read_text("utf-8"))
    logger.info(f"Starting STaR annotation for {len(source_data)} data points with {args.max_workers} workers...")

    # ── Parallel annotation ──────────────────────────────────────────
    annotated_data: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_dp = {executor.submit(process_data_point_with_star, dp): dp for dp in source_data}

        for future in tqdm(concurrent.futures.as_completed(future_to_dp), total=len(source_data), desc="Annotating pairs"):
            try:
                result = future.result()
                if result:
                    annotated_data.append(result)
                    if len(annotated_data) % DEFAULT_CHECKPOINT_INTERVAL == 0:
                        train_path.write_text(json.dumps(annotated_data, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:
                logger.warning(f"A data point generated an exception: {exc}")

    logger.success(f"Annotated {len(annotated_data)} / {len(source_data)} data points")

    # ── Save training set ────────────────────────────────────────────
    # selector training expects:
    # - instruction_content: prompt text
    # - output_content: analysis + final boxed decision
    # Keep legacy fields as well for backward compatibility.
    final_training_set = []
    for item in annotated_data:
        final_choice = str(item.get("final_choice", "")).strip().upper()
        if final_choice not in {"SQL1", "SQL2"}:
            logger.warning(f"Invalid final_choice={final_choice!r}; skip one sample.")
            continue

        output_content = f"{item['cot_analysis'].rstrip()}\n\\box{{{final_choice}}}"
        final_training_set.append(
            {
                "instruction_content": item["prompt_used"],
                "output_content": output_content,
                "instruction": item["prompt_used"],
                "analysis": item["cot_analysis"],
                "final_choice": final_choice,
            }
        )
    train_path.write_text(json.dumps(final_training_set, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Training data saved to: {train_path}")

    # ── Save analysis set ────────────────────────────
    analysis_set = [
        {
            "instruction": item["prompt_used"],
            "analysis": item["cot_analysis"],
            "final_choice": item["final_choice"],
            "metadata": {
                "schema": item["schema"],
                "question": item["question"],
                "chosen_sql": item["chosen_sql"],
                "rejected_sql": item["rejected_sql"],
                "star_stage": item["star_stage"],
            },
        }
        for item in annotated_data
    ]
    analysis_path.write_text(json.dumps(analysis_set, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Analysis data saved to: {analysis_path}")

    logger.success(f"All done -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
