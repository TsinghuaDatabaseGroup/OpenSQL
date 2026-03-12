#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import pickle
import random
import re
import sqlite3
import sys
import time
import types
import uuid
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import func_timeout
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schema_utils import IR2Schema  # noqa: E402
from value_index.vector_index import ColumnVectorIndex  # noqa: E402


# =============================================================================
# Prompt templates (must stay identical to legacy implementation)
# =============================================================================
SCHEMA_LINK_TEMPLATE = """
You are an expert SQL engineer. Your task is to analyze a user's question and a database schema to identify all the necessary tables and columns required to answer that question.
Instructions:
1. Analyze the user's question to understand the core intent. Identify all the tables and columns that are needed to construct a SQL query that answers the question.
2. The selected tables and columns should be sufficient to construct a SQL query that answers the question. Every chosen table's primary key must be included. If JOIN is needed, the foreign key must be included.
3. Your final output must be a single, valid JSON object. The JSON object should have table names as keys and a list of their corresponding, relevant column names as values.

### Database Schema:
{schema}
### User Question: {question}
"""
SCHEMA_LINK_RESPONSE_TEMPLATE = "[Schema Linking Result]"

LOCAL_CLASSIFICATION_TEMPLATE = """You are a SQL expert. Given a database table, a question, and a column in the table, your task is to determine whether the column is useful to generate a SQL query for answering the question.
Note: Some example values of the specified column are shown to you, these values are actual values in the database. If any example values match the question, the column is likely to be useful.
You should return only one word: True or False.
[Table schema]
{table_schema}
[User Question]
{question_with_evidence}
{column_value_examples_prompt}
[Judge]
Based on all the information above, is the column named '{column_name}' useful to answer the question?
"""
LOCAL_RESPONSE_TEMPLATE = "[Judgement]"

NL2SQL_TEMPLATE = """You are an expert SQL engineer. Your task is to write a valid SQLite query to answer the user's question based on the provided database schema.

### Instructions:
1.  Analyze the user's question and the database schema carefully.
2.  Construct a valid SQLite query that accurately answers the question.
3.  If a high-quality query cannot be generated for the given reasoning path, your output must be only `N/A`.
4.  Otherwise, your final output must contain only the SQL query, nothing else.

### Database Schema:
{schema}

### User Question:
{question}
"""
SQL_RESPONSE_TEMPLATE = "[SQL Query Answer]"
CONTROL_TOKENS = [f"[{path.upper()}]" for path in ["Normal", "CTE", "Subquery"]]
NA_TOKEN = "[N/A]"

PROMPT_TEMPLATE_PAIRWISE = """You are a world-class SQL analyst. Your task is to meticulously compare two SQL queries based on a user's question and the database schema, then produce a rigorous Chain-of-Thought (CoT) analysis to determine which SQL is correct.

### Your Task:
Generate a response in two parts: a detailed analysis following a strict four-step process, and a final choice. **Crucially, do not make any value judgments (e.g., "correct", "incorrect", "better") in Steps 1, 2, and 3.**

1.  **Question Analysis**: Restate the user’s intent. What is the core question? What specific columns or calculated values should the final output contain? This defines the ground truth for evaluation.

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

### Output Format: First generate your four-step analysis, then output \\box{{SQL1}} or \\box{{SQL2}} to indicate which SQL query is correct.

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
"""
RERANK_RESPONSE_TEMPLATE = "[Correct Group Judgement]\n"

BASE_BENCHMARKS = {"BIRD_dev", "Spider_dev", "Spider_test", "KaggleDBQA", "MIMIC", "science"}


def install_pickle_compat_shims() -> None:
    """Register legacy module paths so old pickled index files can be loaded."""
    utils_module = types.ModuleType("utils")
    embed_values_module = types.ModuleType("utils.embed_values")
    embed_values_module.ColumnVectorIndex = ColumnVectorIndex
    ir_to_schema_module = types.ModuleType("utils.ir_to_schema")
    ir_to_schema_module.IR2Schema = IR2Schema
    utils_module.embed_values = embed_values_module
    utils_module.ir_to_schema = ir_to_schema_module

    legacy_embed_values_module = types.ModuleType("embed_values")
    legacy_embed_values_module.ColumnVectorIndex = ColumnVectorIndex

    sys.modules.setdefault("utils", utils_module)
    sys.modules["utils.embed_values"] = embed_values_module
    sys.modules["utils.ir_to_schema"] = ir_to_schema_module
    sys.modules.setdefault("embed_values", legacy_embed_values_module)


def release_vllm_model(llm: Optional[LLM], stage_name: str) -> None:
    """Release vLLM resources aggressively between stages to avoid OOM."""
    if llm is None:
        return

    try:
        llm_engine = getattr(llm, "llm_engine", None)
        if llm_engine is not None:
            model_executor = getattr(llm_engine, "model_executor", None)
            if model_executor is not None and hasattr(model_executor, "shutdown"):
                model_executor.shutdown()
    except Exception as exc:
        logger.debug(f"[{stage_name}] vLLM shutdown raised: {exc}")

    del llm
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def release_embedding_model(emb_model: Optional[SentenceTransformer], stage_name: str) -> None:
    """Release embedding model resources after step1."""
    if emb_model is None:
        return
    try:
        del emb_model
    except Exception as exc:
        logger.debug(f"[{stage_name}] embedding model release raised: {exc}")
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments for the unified pipeline."""
    parser = argparse.ArgumentParser(description="Unified batch inference pipeline for schema linking, SQL generation, and reranking.")

    parser.add_argument("--global-model-path", type=str, required=True, help="Model path for step0 global schema linking.")
    parser.add_argument("--local-model-path", type=str, required=True, help="Model path for step1 local schema linking.")
    parser.add_argument("--generator-model-path", type=str, required=True, help="Model path for step2 SQL generation.")
    parser.add_argument("--selector-model-path", type=str, required=True, help="Model path for step3 rerank.")
    parser.add_argument("--embedding-model-name-or-path", type=str, default="Alibaba-NLP/gte-large-en-v1.5", help="Embedding model used in step1.")

    parser.add_argument("--evaluation-benchmark", type=str, required=True, help="Benchmark name.")
    parser.add_argument("--ir-data-dir", type=Path, required=True, help="IR directory.")
    parser.add_argument("--evaluation-dir", type=Path, required=True, help="Dynamic evaluation input directory.")
    parser.add_argument("--index-dir", type=Path, required=True, help="Value index directory.")
    parser.add_argument("--db-base-path", type=Path, required=True, help="Database root directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--hf-cache-dir", type=Path, default=None, help="Cache directory for local embedding/tokenizer load.")
    parser.add_argument("--embedding-device", type=str, default="cuda", help="Embedding model device.")

    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Global tensor parallel size for all stages.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="Global vLLM GPU memory utilization ratio.")
    parser.add_argument("--step3-max-num-seqs", type=int, default=256, help="max_num_seqs for step3 vLLM engine.")

    parser.add_argument("--step0-max-new-tokens", type=int, default=1024, help="Max new tokens for step0.")
    parser.add_argument("--step0-temperature", type=float, default=0.0, help="Temperature for step0.")
    parser.add_argument("--step1-max-new-tokens", type=int, default=8, help="Max new tokens for step1.")
    parser.add_argument("--step1-temperature", type=float, default=0.0, help="Temperature for step1.")
    parser.add_argument("--num-generations-per-token", type=int, default=8, help="Number of SQL generations per control token in step2.")
    parser.add_argument("--step2-max-new-tokens", type=int, default=1024, help="Max new tokens for step2.")
    parser.add_argument("--step2-temperature", type=float, default=1.5, help="Temperature for step2.")
    parser.add_argument("--step2-top-p", type=float, default=0.95, help="Top-p for step2.")
    parser.add_argument("--step2-top-k", type=int, default=50, help="Top-k for step2.")
    parser.add_argument("--step3-max-new-tokens", type=int, default=2048, help="Max new tokens for step3.")

    parser.add_argument("--sql-timeout", type=int, default=100, help="Per-query SQL execution timeout in seconds.")
    parser.add_argument("--cpu-workers", type=int, default=128, help="CPU worker process count for SQL execution.")
    parser.add_argument("--fallback-rounds", type=int, default=3, help="Fallback generation rounds with dynamic_schema in step2.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    parser.set_defaults(local_files_only=True)
    parser.add_argument(
        "--local-files-only",
        dest="local_files_only",
        action="store_true",
        help="Force local-only model/tokenizer loading where supported (default).",
    )
    parser.add_argument(
        "--online-enabled",
        dest="local_files_only",
        action="store_false",
        help="Allow online model/tokenizer loading where supported.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    """Load JSON with UTF-8 encoding."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    """Dump JSON with stable UTF-8 formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_question(dp: Dict[str, Any]) -> str:
    """Return the canonical question field for a data point."""
    if "question_with_evidence" in dp:
        return dp["question_with_evidence"]
    if "question_refine" in dp:
        return dp["question_refine"]
    return dp["question"]


def resolve_dynamic_input_and_ir_path(evaluation_dir: Path, ir_data_dir: Path, benchmark: str) -> Tuple[Path, Path]:
    """Resolve dynamic input file and IR file for step0."""
    if benchmark in BASE_BENCHMARKS:
        dynamic_candidate = evaluation_dir / f"{benchmark}_dynamic.json"
        plain_candidate = evaluation_dir / f"{benchmark}.json"
        if dynamic_candidate.exists():
            input_data_path = dynamic_candidate
        elif plain_candidate.exists():
            input_data_path = plain_candidate
        else:
            input_data_path = dynamic_candidate

        ir_candidate = ir_data_dir / f"{benchmark}_ir.json"
        ir_plain_candidate = ir_data_dir / f"{benchmark}.json"
        if ir_candidate.exists():
            ir_path = ir_candidate
        elif ir_plain_candidate.exists():
            ir_path = ir_plain_candidate
        else:
            ir_path = ir_candidate
        return input_data_path, ir_path

    if benchmark.startswith("DB_") or benchmark.startswith("NLQ_") or benchmark.startswith("SQL_"):
        input_data_path = evaluation_dir / "Dr-Spider" / f"{benchmark}.json"
        if benchmark.startswith("DB_"):
            ir_candidate = ir_data_dir / f"{benchmark}_ir.json"
            ir_plain_candidate = ir_data_dir / f"{benchmark}.json"
            if ir_candidate.exists():
                ir_path = ir_candidate
            elif ir_plain_candidate.exists():
                ir_path = ir_plain_candidate
            else:
                ir_path = ir_candidate
        else:
            spider_ir_candidate = ir_data_dir / "Spider_ir.json"
            spider_ir_plain_candidate = ir_data_dir / "Spider_dev.json"
            if spider_ir_candidate.exists():
                ir_path = spider_ir_candidate
            elif spider_ir_plain_candidate.exists():
                ir_path = spider_ir_plain_candidate
            else:
                ir_path = spider_ir_candidate
        return input_data_path, ir_path

    raise ValueError(f"Unsupported benchmark: {benchmark}")


def resolve_ir_and_index_path(ir_data_dir: Path, index_dir: Path, benchmark: str) -> Tuple[Path, Path]:
    """Resolve IR path and value-index directory for step1."""
    if benchmark in BASE_BENCHMARKS:
        ir_candidate = ir_data_dir / f"{benchmark}_ir.json"
        ir_plain_candidate = ir_data_dir / f"{benchmark}.json"
        if ir_candidate.exists():
            ir_path = ir_candidate
        elif ir_plain_candidate.exists():
            ir_path = ir_plain_candidate
        else:
            ir_path = ir_candidate
        return ir_path, index_dir / benchmark

    if benchmark.startswith("DB_"):
        ir_candidate = ir_data_dir / f"{benchmark}_ir.json"
        ir_plain_candidate = ir_data_dir / f"{benchmark}.json"
        if ir_candidate.exists():
            ir_path = ir_candidate
        elif ir_plain_candidate.exists():
            ir_path = ir_plain_candidate
        else:
            ir_path = ir_candidate
        return ir_path, index_dir / benchmark

    if benchmark.startswith("NLQ_") or benchmark.startswith("SQL_"):
        spider_ir_candidate = ir_data_dir / "Spider_ir.json"
        spider_ir_plain_candidate = ir_data_dir / "Spider_dev.json"
        if spider_ir_candidate.exists():
            ir_path = spider_ir_candidate
        elif spider_ir_plain_candidate.exists():
            ir_path = spider_ir_plain_candidate
        else:
            ir_path = spider_ir_candidate
        return ir_path, index_dir / "Spider_dev"

    raise ValueError(f"Unsupported benchmark: {benchmark}")


def resolve_db_path(db_base_path: Path, benchmark: str, db_id: str) -> Path:
    """Resolve the SQLite file path for a benchmark/db_id pair."""
    if benchmark == "BIRD_dev":
        return db_base_path / benchmark / "dev_databases" / db_id / f"{db_id}.sqlite"
    if benchmark == "Spider_dev":
        return db_base_path / "spider_data" / "database" / db_id / f"{db_id}.sqlite"
    if benchmark == "Spider_test":
        return db_base_path / "spider_data" / "test_database" / db_id / f"{db_id}.sqlite"
    if benchmark == "KaggleDBQA":
        return db_base_path / benchmark / "databases" / db_id / f"{db_id}.sqlite"
    if benchmark == "MIMIC":
        return db_base_path / benchmark / "MIMIC.db"
    if benchmark == "science":
        return db_base_path / "science_benchmark" / db_id / f"{db_id}.sqlite"
    if benchmark.startswith("DB_"):
        return db_base_path / "Dr-Spider" / "data" / benchmark / "database_post_perturbation" / db_id / f"{db_id}.sqlite"
    if benchmark.startswith("NLQ_") or benchmark.startswith("SQL_"):
        return db_base_path / "spider_data" / "database" / db_id / f"{db_id}.sqlite"
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def extract_json_from_string(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from raw model output."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            return {}

    try:
        json_str = re.sub(r",\s*([\}\]])", r"\1", json_str)
        loaded = json.loads(json_str)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        return {}
    return {}


def normalize_table_names_with_ir(prediction_json: Dict[str, Any], ir_this_db: Dict[str, Any]) -> Dict[str, Any]:
    """Map predicted table keys to IR table names using case-insensitive matching."""
    if not prediction_json:
        return prediction_json

    bad_table_names: List[str] = []
    temp = prediction_json.copy()
    for table_name in prediction_json:
        for table in ir_this_db.get("tables", []):
            if table["table_name"].lower() == table_name.lower():
                temp[table["table_name"]] = prediction_json[table_name].copy()
                bad_table_names.append(table_name)
                break

    for table_name in bad_table_names:
        if table_name in temp:
            del temp[table_name]
    return temp.copy()


def clean_generated_sql(text: str) -> str:
    """Normalize model output into a single SQL string (or NA token)."""
    if SQL_RESPONSE_TEMPLATE in text:
        text = text.split(SQL_RESPONSE_TEMPLATE, 1)[1]
    text = text.strip()
    if text == NA_TOKEN:
        return NA_TOKEN
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    return re.sub(r"\s+", " ", text).strip().strip(";")


def execute_sql(db_path: Path, query: str, timeout: int) -> Tuple[str, Any]:
    """Execute SQL in read-only mode with timeout protection."""
    try:
        if not query.strip():
            return "error", "Empty query."

        @func_timeout.func_set_timeout(timeout)
        def exec_on_db(path: Path, q: str) -> Any:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
                cursor = conn.cursor()
                cursor.execute(q)
                return cursor.fetchall()

        result = exec_on_db(db_path, query)
        return "success", result
    except func_timeout.FunctionTimedOut:
        return "error", f"Query timed out after {timeout} seconds."
    except Exception as exc:
        return "error", f"Execution error: {exc}"


def execute_sql_wrapper(args: Tuple[str, Path, str, int]) -> Tuple[str, str, Any]:
    """Multiprocessing wrapper that preserves request id in returned tuple."""
    request_id, db_path, query, timeout = args
    status, result = execute_sql(db_path, query, timeout)
    return request_id, status, result


def fix_order_table_name(sql: str) -> str:
    """Quote bare `order` identifiers while preserving string literals and ORDER BY."""
    pattern = re.compile(r"('(?:''|[^'])*')|" + r'("[^"]*")|' + r"(ORDER\s+BY)|" + r"(\border\b)", re.IGNORECASE)

    def replace_callback(match: re.Match[str]) -> str:
        if match.group(1) or match.group(2) or match.group(3):
            return match.group(0)
        if match.group(4):
            return '"order"'
        return match.group(0)

    return pattern.sub(replace_callback, sql)


def result_is_empty_or_all_none(result: Any) -> bool:
    """Treat empty/all-NULL result sets as invalid execution outcomes."""
    if result is None:
        return True
    if not isinstance(result, list):
        return False
    if len(result) == 0:
        return True
    for row in result:
        r = row if isinstance(row, (list, tuple)) else (row,)
        if not all(c is None for c in r):
            return False
    return True


def format_sql_results_for_llm(execution_result: Any, max_rows_to_show: int = 5, max_row_length: int = 512) -> str:
    """Format SQL execution results into bounded text for rerank prompts."""
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

    formatted_rows: List[str] = []
    for row in rows_to_process:
        row_str = str(row)
        if len(row_str) > max_row_length:
            row_str = row_str[:max_row_length] + "..."
        formatted_rows.append(row_str)
    return header + "\n" + "\n".join(formatted_rows)


def generate_pairwise_prompt(schema: str, question: str, sql1: str, result1: str, sql2: str, result2: str) -> str:
    """Render the step3 pairwise prompt."""
    return PROMPT_TEMPLATE_PAIRWISE.format(schema=schema, question=question, sql1=sql1, result1=result1, sql2=sql2, result2=result2)


def load_index_dict(index_dir: Path) -> Dict[str, Dict[Tuple[str, str], ColumnVectorIndex]]:
    """Load per-db value indexes from pickle with legacy compatibility shims."""
    install_pickle_compat_shims()

    index_dict: Dict[str, Dict[Tuple[str, str], ColumnVectorIndex]] = {}
    for index_path in sorted(index_dir.glob("*.pkl")):
        db_id = index_path.stem
        with index_path.open("rb") as f:
            index_dict[db_id] = pickle.load(f)
    return index_dict


# =============================================================================
# Stage 0: Global schema linking
# =============================================================================
def run_step0(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Run global schema linking for all samples."""
    logger.info("Step0 started: global schema linking.")
    input_data_path, ir_path = resolve_dynamic_input_and_ir_path(args.evaluation_dir, args.ir_data_dir, args.evaluation_benchmark)
    full_input_dataset = load_json(input_data_path)
    irs = load_json(ir_path)
    ir_map = {ir.get("db_id"): ir for ir in irs}

    llm = LLM(
        model=args.global_model_path,
        tokenizer=args.global_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.global_model_path, local_files_only=args.local_files_only)
    sampling_params = SamplingParams(max_tokens=args.step0_max_new_tokens, temperature=args.step0_temperature)

    # Build one prompt per sample, then run a single batched generation call.
    prompts_batch: List[str] = []
    for dp in tqdm(full_input_dataset, desc="Step0-Preparing Prompts"):
        question = get_question(dp)
        input_prompt = SCHEMA_LINK_TEMPLATE.format(schema=dp["dynamic_schema"], question=question)
        messages = [{"role": "user", "content": input_prompt}, {"role": "assistant", "content": SCHEMA_LINK_RESPONSE_TEMPLATE}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
        prompts_batch.append(formatted_prompt)

    if prompts_batch:
        outputs = llm.generate(prompts_batch, sampling_params, use_tqdm=True)
    else:
        outputs = []

    # Post-process outputs and inject global_schema_linking into each sample.
    all_processed_data: List[Dict[str, Any]] = []
    for dp, output in tqdm(zip(full_input_dataset, outputs), total=len(full_input_dataset), desc="Step0-Postprocess"):
        dp_copy = copy.deepcopy(dp)
        if args.evaluation_benchmark == "MIMIC":
            dp_copy["db_id"] = "MIMIC"
        elif args.evaluation_benchmark == "science":
            dp_copy["db_id"] = dp_copy["benchmark"]

        response_text = output.outputs[0].text
        original_prediction_json = extract_json_from_string(response_text)
        final_prediction_json = original_prediction_json

        ir_this_db = ir_map.get(dp_copy.get("db_id"))
        if ir_this_db is not None:
            original_prediction_json = normalize_table_names_with_ir(original_prediction_json, ir_this_db)
            try:
                if original_prediction_json:
                    if "question_with_evidence" in dp_copy:
                        question_for_converter = dp_copy["question_with_evidence"]
                    else:
                        question_for_converter = dp_copy["question"]
                    converter = IR2Schema(
                        ir=ir_this_db,
                        chosen=original_prediction_json,
                        tindex=None,
                        question=question_for_converter,
                        emb_model=None,
                        print_contain_null=False,
                    )
                    _, enhanced_prediction_json = converter.render_schema()
                    final_prediction_json = enhanced_prediction_json if enhanced_prediction_json is not None else original_prediction_json
            except Exception as exc:
                logger.warning(f"Step0 failed to enhance predicted schema for db_id={dp_copy.get('db_id')}: {exc}")
                final_prediction_json = original_prediction_json
        else:
            final_prediction_json = original_prediction_json

        dp_copy["global_schema_linking"] = final_prediction_json if isinstance(final_prediction_json, dict) else {}
        all_processed_data.append(dp_copy)

    all_processed_data.sort(key=lambda x: (x.get("db_id", ""), x.get("question", "")))
    step0_output_path = args.output_dir / "step0_output.json"
    dump_json(step0_output_path, all_processed_data)
    logger.info(f"Step0 finished. Output: {step0_output_path}")

    release_vllm_model(llm, stage_name="step0")
    return all_processed_data


# =============================================================================
# Stage 1: Local schema linking
# =============================================================================
def run_step1(args: argparse.Namespace, input_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run local schema linking and render prepared_schema_str."""
    logger.info("Step1 started: local schema linking.")
    ir_path, index_path = resolve_ir_and_index_path(args.ir_data_dir, args.index_dir, args.evaluation_benchmark)
    irs = load_json(ir_path)
    ir_map = {ir.get("db_id"): ir for ir in irs}
    index_dict = load_index_dict(index_path)

    llm = LLM(
        model=args.local_model_path,
        tokenizer=args.local_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.local_model_path, local_files_only=args.local_files_only)
    tokenizer.padding_side = "left"

    emb_model = SentenceTransformer(
        args.embedding_model_name_or_path,
        trust_remote_code=True,
        device=args.embedding_device,
        cache_folder=str(args.hf_cache_dir) if args.hf_cache_dir is not None else None,
        local_files_only=args.local_files_only,
    )

    sampling_params = SamplingParams(max_tokens=args.step1_max_new_tokens, temperature=args.step1_temperature)

    full_input_dataset = copy.deepcopy(input_dataset)
    random.shuffle(full_input_dataset)

    # Collect all column-level classification prompts first for batch inference.
    all_prompts: List[str] = []
    prompts_metadata: List[Tuple[int, str, str]] = []

    for i, dp in tqdm(enumerate(full_input_dataset), total=len(full_input_dataset), desc="Step1-Preparing Prompts"):
        db_id = dp["db_id"]
        question = get_question(dp)
        global_schema = dp.get("global_schema_linking", {})

        if not global_schema or db_id not in index_dict:
            dp["local_schema_linking"] = copy.deepcopy(global_schema)
            continue

        dp["local_schema_linking"] = copy.deepcopy(global_schema)
        ir_this_db = ir_map.get(db_id)
        if ir_this_db is None:
            logger.warning(f"Step1 IR not found for db_id={db_id}.")
            continue

        try:
            converter = IR2Schema(
                ir=ir_this_db,
                chosen=None,
                tindex=index_dict[db_id],
                question=question,
                emb_model=emb_model,
                print_contain_null=False,
            )

            for table_name in global_schema.keys():
                table_ir = next((table for table in ir_this_db["tables"] if table["table_name"] == table_name), None)
                if table_ir is None:
                    continue

                pk_indices = set(table_ir.get("primary_keys", []))
                fk_columns = {fk["column"].strip('"') for fk in table_ir.get("foreign_keys", [])}
                candidate_columns = [col["col_name"] for col in table_ir["columns"] if col["col_idx"] not in pk_indices and col["col_name"] not in fk_columns]

                for column_name in candidate_columns:
                    if column_name in dp["local_schema_linking"].get(table_name, []):
                        continue

                    table_statement, column_value_examples = converter.render_table_and_column_examples(table_name, column_name)
                    prompt_content = LOCAL_CLASSIFICATION_TEMPLATE.format(
                        table_schema=table_statement,
                        question_with_evidence=question,
                        column_value_examples_prompt=column_value_examples,
                        column_name=column_name,
                    )
                    messages = [{"role": "user", "content": prompt_content}, {"role": "assistant", "content": LOCAL_RESPONSE_TEMPLATE}]
                    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
                    all_prompts.append(formatted_prompt)
                    prompts_metadata.append((i, table_name, column_name))
        except Exception as exc:
            logger.error(f"Step1 failed in prompt preparation for db_id={db_id}: {exc}")

    if all_prompts:
        outputs = llm.generate(all_prompts, sampling_params)
    else:
        outputs = []

    # Merge model judgements back to local_schema_linking.
    for i, output in tqdm(enumerate(outputs), total=len(outputs), desc="Step1-Merging Results"):
        dp_index, table_name, column_name = prompts_metadata[i]
        response = output.outputs[0].text.strip()
        if "true" in response.lower():
            target_dp = full_input_dataset[dp_index]
            if column_name not in target_dp["local_schema_linking"][table_name]:
                target_dp["local_schema_linking"][table_name].append(column_name)

    # Render final schema strings used by SQL generation.
    for dp in tqdm(full_input_dataset, desc="Step1-Finalizing Schema"):
        db_id = dp["db_id"]
        if "local_schema_linking" not in dp:
            dp["prepared_schema_str"] = ""
            continue

        malformed = any(not isinstance(cols, list) for cols in dp["local_schema_linking"].values())
        if malformed:
            logger.warning(f"Step1 local_schema_linking malformed for db_id={db_id}")
            dp["prepared_schema_str"] = ""
            continue

        ir_this_db = ir_map.get(db_id)
        if ir_this_db is None or db_id not in index_dict:
            dp["prepared_schema_str"] = ""
            continue

        question = get_question(dp)
        final_converter = IR2Schema(
            ir=ir_this_db,
            chosen=dp["local_schema_linking"],
            tindex=index_dict[db_id],
            question=question,
            emb_model=emb_model,
            print_contain_null=True,
        )
        try:
            schema_str, _ = final_converter.render_schema()
        except Exception as exc:
            logger.warning(f"Step1 failed to render final schema for db_id={db_id}: {exc}")
            schema_str = ""
        dp["prepared_schema_str"] = schema_str

    step1_output_path = args.output_dir / "step1_output.json"
    dump_json(step1_output_path, full_input_dataset)
    logger.info(f"Step1 finished. Output: {step1_output_path}")

    release_vllm_model(llm, stage_name="step1")
    release_embedding_model(emb_model, stage_name="step1")
    return full_input_dataset


# =============================================================================
# Stage 2: SQL generation + execution-based consistency
# =============================================================================
def run_step2(args: argparse.Namespace, input_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate SQL candidates, execute them, apply fallback, and aggregate metrics."""
    logger.info("Step2 started: SQL generation and execution consistency.")

    if args.cpu_workers is None or args.cpu_workers <= 0:
        args.cpu_workers = cpu_count()
    logger.info(f"Step2 SQL execution worker count: {args.cpu_workers}")

    llm = LLM(
        model=args.generator_model_path,
        tokenizer=args.generator_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.generator_model_path, local_files_only=args.local_files_only)

    essential_tokens = CONTROL_TOKENS + [NA_TOKEN]
    vocab = tokenizer.get_vocab()
    for token in essential_tokens:
        if token not in vocab:
            raise ValueError(f"Critical token not found in tokenizer vocab: {token}")

    full_input_dataset = copy.deepcopy(input_dataset)
    db_base_path = args.db_base_path
    benchmark = args.evaluation_benchmark

    # Prepare generation requests for all samples and control tokens.
    vllm_requests: List[Tuple[str, str]] = []
    request_metadata: Dict[str, Dict[str, Any]] = {}
    dp_info_map = {i: dp for i, dp in enumerate(full_input_dataset)}

    for i, dp in enumerate(tqdm(full_input_dataset, desc="Step2-Preparing Requests")):
        db_id = dp["db_id"]
        db_path = resolve_db_path(db_base_path, benchmark, db_id)
        if not db_path.exists():
            logger.warning(f"Step2 database not found for db_id={db_id}, path={db_path}.")
            dp_info_map.pop(i, None)
            continue

        question = get_question(dp)
        for token in CONTROL_TOKENS:
            controlled_question = f"{token} {question}"
            prompt = NL2SQL_TEMPLATE.format(schema=dp["prepared_schema_str"], question=controlled_question)
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": SQL_RESPONSE_TEMPLATE}]
            final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
            request_id = str(uuid.uuid4())
            vllm_requests.append((final_prompt, request_id))
            request_metadata[request_id] = {
                "dp_idx": i,
                "db_id": db_id,
                "control_token": token,
                "type": "generated",
                "question": question,
            }

        gold_sql_request_id = str(uuid.uuid4())
        request_metadata[gold_sql_request_id] = {
            "dp_idx": i,
            "db_id": db_id,
            "type": "gold",
            "query": dp["query"] if "query" in dp else dp["sql"],
            "question": question,
        }

    # Batched vLLM generation for all prepared prompts.
    prompts_to_run = [req[0] for req in vllm_requests]
    request_ids_to_run = [req[1] for req in vllm_requests]

    sampling_params = SamplingParams(
        n=args.num_generations_per_token,
        temperature=args.step2_temperature,
        top_p=args.step2_top_p,
        top_k=args.step2_top_k,
        max_tokens=args.step2_max_new_tokens,
        stop=[tokenizer.eos_token],
    )
    outputs = llm.generate(prompts_to_run, sampling_params) if prompts_to_run else []
    all_generations = {request_ids_to_run[i]: [generation.text for generation in output.outputs] for i, output in enumerate(outputs)}
    total_generated_count = sum(len(gens) for gens in all_generations.values())
    logger.info(f"Step2 generated SQL count: {total_generated_count}")

    # Convert generated text to executable SQL tasks.
    sql_execution_tasks: List[Tuple[str, Path, str, int]] = []
    for request_id, generated_texts in all_generations.items():
        meta = request_metadata[request_id]
        db_id = meta["db_id"]
        db_path = resolve_db_path(db_base_path, benchmark, db_id)

        for i, text in enumerate(generated_texts):
            sub_request_id = f"{request_id}_{i}"
            cleaned_sql = clean_generated_sql(text)
            if not cleaned_sql.strip():
                continue
            if cleaned_sql == NA_TOKEN:
                logger.warning(f"Step2 SQL is N/A for db_id={db_id}, question={meta['question']}")
                continue

            fixed_sql = fix_order_table_name(cleaned_sql)
            sql_execution_tasks.append((sub_request_id, db_path, fixed_sql, args.sql_timeout))
            request_metadata[sub_request_id] = {**meta, "sql": fixed_sql}

    for req_id, meta in request_metadata.items():
        if meta.get("type") == "gold":
            db_path = resolve_db_path(db_base_path, benchmark, meta["db_id"])
            sql_execution_tasks.append((req_id, db_path, meta["query"], args.sql_timeout))

    # Execute generated SQL + gold SQL in parallel.
    execution_results: Dict[str, Tuple[str, Any]] = {}
    with Pool(args.cpu_workers) as pool:
        pbar = tqdm(pool.imap_unordered(execute_sql_wrapper, sql_execution_tasks), total=len(sql_execution_tasks), desc="Step2-Executing SQL")
        for req_id, status, result in pbar:
            execution_results[req_id] = (status, result)

    def compute_attempt_and_success() -> Tuple[Dict[int, int], Dict[int, int]]:
        # Only treat non-empty and non-all-NULL successful executions as valid.
        attempts: Dict[int, int] = defaultdict(int)
        successes: Dict[int, int] = defaultdict(int)
        for req_id, (status, _) in execution_results.items():
            meta = request_metadata.get(req_id)
            if (meta is None) or (meta.get("type") != "generated"):
                continue
            dp_idx = meta["dp_idx"]
            attempts[dp_idx] += 1
            if str(status).strip().lower() == "success":
                result = execution_results[req_id][1]
                if not result_is_empty_or_all_none(result):
                    successes[dp_idx] += 1
        return attempts, successes

    _, success_map = compute_attempt_and_success()
    successful_dp = {dp_idx for dp_idx, cnt in success_map.items() if cnt > 0}
    blocked_dp = set()

    # Fallback loop: regenerate with dynamic_schema for unsolved samples.
    for round_idx in range(1, args.fallback_rounds + 1):
        _, success_map = compute_attempt_and_success()
        successful_dp = {dp_idx for dp_idx, cnt in success_map.items() if cnt > 0}
        candidates = [dp_idx for dp_idx in dp_info_map.keys() if dp_idx not in successful_dp and dp_idx not in blocked_dp]
        if not candidates:
            logger.info(f"Step2 fallback round {round_idx}: no candidate remains.")
            break

        prompts_round: List[str] = []
        request_ids_round: List[str] = []
        sql_execution_tasks_round: List[Tuple[str, Path, str, int]] = []

        for dp_idx in candidates:
            dp = dp_info_map[dp_idx]
            db_id = dp["db_id"]
            if not dp.get("dynamic_schema"):
                blocked_dp.add(dp_idx)
                logger.warning(f"Step2 fallback skipped for db_id={db_id}: dynamic_schema missing.")
                continue

            question = get_question(dp)
            for token in CONTROL_TOKENS:
                controlled_question = f"{token} {question}"
                prompt = NL2SQL_TEMPLATE.format(schema=dp["dynamic_schema"], question=controlled_question)
                messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": SQL_RESPONSE_TEMPLATE}]
                final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)

                request_id = str(uuid.uuid4())
                prompts_round.append(final_prompt)
                request_ids_round.append(request_id)
                request_metadata[request_id] = {
                    "dp_idx": dp_idx,
                    "db_id": db_id,
                    "control_token": token,
                    "type": "generated",
                    "schema_source": "dynamic",
                    "question": question,
                }

        if not prompts_round:
            remaining = len([dp_idx for dp_idx in dp_info_map.keys() if dp_idx not in successful_dp and dp_idx not in blocked_dp])
            logger.info(f"Step2 fallback round {round_idx}: no prompts to run. remaining={remaining}")
            break

        # Round-level batched generation + execution.
        outputs_round = llm.generate(prompts_round, sampling_params)
        all_generations_round = {request_ids_round[i]: [generation.text for generation in output.outputs] for i, output in enumerate(outputs_round)}

        for request_id, generated_texts in all_generations_round.items():
            meta = request_metadata[request_id]
            db_path = resolve_db_path(db_base_path, benchmark, meta["db_id"])
            for i, text in enumerate(generated_texts):
                sub_request_id = f"{request_id}_{i}"
                cleaned_sql = clean_generated_sql(text)
                if not cleaned_sql.strip():
                    continue
                if cleaned_sql == NA_TOKEN:
                    continue
                sql_execution_tasks_round.append((sub_request_id, db_path, cleaned_sql, args.sql_timeout))
                request_metadata[sub_request_id] = {**meta, "sql": cleaned_sql}

        if sql_execution_tasks_round:
            with Pool(args.cpu_workers) as pool:
                pbar = tqdm(
                    pool.imap_unordered(execute_sql_wrapper, sql_execution_tasks_round),
                    total=len(sql_execution_tasks_round),
                    desc=f"Step2-Fallback SQL round {round_idx}",
                )
                for req_id, status, result in pbar:
                    execution_results[req_id] = (status, result)

        success_before = set(successful_dp)
        _, success_map = compute_attempt_and_success()
        successful_dp = {dp_idx for dp_idx, cnt in success_map.items() if cnt > 0}
        rescued = len(successful_dp - success_before)
        remaining = len([dp_idx for dp_idx in dp_info_map.keys() if dp_idx not in successful_dp and dp_idx not in blocked_dp])
        logger.info(f"Step2 fallback round {round_idx}: rescued={rescued}, remaining={remaining}")
        if remaining == 0:
            break

    dp_using_dynamic_schema = {meta["dp_idx"] for meta in request_metadata.values() if meta.get("schema_source") == "dynamic"}

    # Aggregate all execution traces per sample.
    aggregated_results: Dict[int, Dict[str, Any]] = {}
    for i, dp in dp_info_map.items():
        dp_copy = {**dp}
        if i in dp_using_dynamic_schema and dp.get("dynamic_schema"):
            dp_copy["prepared_schema_str"] = dp["dynamic_schema"]
        aggregated_results[i] = {
            **dp_copy,
            "candidate_sql_sources": {},
            "execution_details": {},
            "raw_execution_results": [],
            "gold_status": "error",
            "gold_sql_result": None,
        }

    for req_id, (status, result) in execution_results.items():
        if req_id not in request_metadata:
            continue
        meta = request_metadata[req_id]
        dp_idx = meta["dp_idx"]
        if meta["type"] == "generated":
            sql = meta.get("sql")
            if sql:
                aggregated_results[dp_idx]["raw_execution_results"].append({"sql": sql, "status": status, "result": result})
                aggregated_results[dp_idx]["execution_details"][sql] = (status, str(result))
                if sql not in aggregated_results[dp_idx]["candidate_sql_sources"]:
                    aggregated_results[dp_idx]["candidate_sql_sources"][sql] = set()
        elif meta["type"] == "gold":
            aggregated_results[dp_idx]["gold_status"] = status
            aggregated_results[dp_idx]["gold_sql_result"] = result
            aggregated_results[dp_idx]["gold_sql"] = meta["query"]
            if status == "error":
                logger.warning(f"Step2 gold SQL failed for idx={dp_idx}")

    # Self-consistency voting over execution-result buckets.
    final_results: List[Dict[str, Any]] = []
    for idx, dp_data in tqdm(enumerate(aggregated_results.values()), total=len(aggregated_results), desc="Step2-Consistency"):
        raw_results_list = dp_data.get("raw_execution_results", [])
        sql_to_raw_result_map = {item["sql"]: (item["status"], item["result"]) for item in raw_results_list}

        execution_groups: Dict[str, List[str]] = {}
        for item in raw_results_list:
            sql, status, result = item["sql"], item["status"], item["result"]
            if status == "success":
                try:
                    processed_result = set(tuple(map(str, row)) for row in result)
                    result_key = str(sorted(list(processed_result)))
                    if result_key not in execution_groups:
                        execution_groups[result_key] = []
                    execution_groups[result_key].append(sql)
                except Exception as exc:
                    logger.warning(f"Step2 consistency processing failed at idx={idx}: {exc}")

        sorted_groups = sorted(execution_groups.items(), key=lambda item: len(item[1]), reverse=True)
        final_sql = "SELECT 'ERROR: No valid SQL was generated or executed successfully';"
        consistency_result = None
        if sorted_groups:
            top_sql_group = sorted_groups[0][1]
            final_sql = random.choice(top_sql_group)
            _, consistency_result = sql_to_raw_result_map.get(final_sql, ("error", None))
        else:
            logger.error(f"Step2 no valid SQL execution for idx={idx}")

        gold_status = dp_data["gold_status"]
        gold_result = dp_data["gold_sql_result"]
        gold_sql = dp_data.get("gold_sql", "")

        is_correct = False
        recalled_correctly = False
        if gold_status == "success":
            try:
                gold_result_set = set(gold_result)
                if consistency_result is not None:
                    is_correct = set(consistency_result) == gold_result_set
                for item in raw_results_list:
                    if item["status"] == "success" and set(item["result"]) == gold_result_set:
                        recalled_correctly = True
                        break
            except Exception as exc:
                logger.warning(f"Step2 cannot compare result sets for db_id={dp_data.get('db_id')}: {exc}")

        gold_in_top1_bucket = False
        gold_in_top2_buckets = False
        gold_in_top3_buckets = False
        if gold_status == "success" and sorted_groups:
            try:
                gold_result_as_set = set(tuple(map(str, row)) for row in gold_result)
                gold_result_key = str(sorted(list(gold_result_as_set)))
                top_bucket_keys = [group[0] for group in sorted_groups]
                gold_in_top1_bucket = gold_result_key in top_bucket_keys[:1]
                gold_in_top2_buckets = gold_result_key in top_bucket_keys[:2]
                gold_in_top3_buckets = gold_result_key in top_bucket_keys[:3]
            except Exception as exc:
                logger.warning(f"Step2 cannot compute top-k hit for db_id={dp_data.get('db_id')}: {exc}")

        execution_buckets: List[Dict[str, Any]] = []
        for _, sql_list in sorted_groups:
            unique_sqls_in_group = sorted(list(set(sql_list)))
            sample_sql = unique_sqls_in_group[0]
            status, raw_result = sql_to_raw_result_map.get(sample_sql, ("error", None))
            if status == "success":
                execution_buckets.append({"sqls": unique_sqls_in_group, "execution_result": raw_result})

        dp_data.update(
            {
                "candidate_sql_sources": {sql: list(sources) for sql, sources in dp_data["candidate_sql_sources"].items()},
                "num_execution_groups": len(execution_groups),
                "final_sql": final_sql,
                "gold_sql": gold_sql,
                "self_consistency_result": str(consistency_result),
                "gold_sql_result": gold_result,
                "execution_correct": is_correct,
                "recalled_correctly": recalled_correctly,
                "gold_in_top1_bucket": gold_in_top1_bucket,
                "gold_in_top2_buckets": gold_in_top2_buckets,
                "gold_in_top3_buckets": gold_in_top3_buckets,
                "execution_buckets": execution_buckets,
            }
        )
        dp_data.pop("raw_execution_results", None)
        final_results.append(dp_data)

    latest_record_path = args.output_dir / "step2_latest_run.json"
    dump_json(latest_record_path, final_results)

    best_record_path = args.output_dir / "best_record.json"
    if best_record_path.exists():
        best_record = load_json(best_record_path).get("best_record", 0.0)
    else:
        best_record = 0.0

    total_samples = len(final_results)
    correct_predictions = sum(1 for result in final_results if result["execution_correct"])
    exec_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0.0

    # Keep a best-record snapshot compatible with existing workflow.
    if exec_accuracy > best_record:
        best_record = exec_accuracy
        dump_json(best_record_path, {"best_record": best_record})
        step2_best_record_path = args.output_dir / "step2_best_record.json"
        dump_json(step2_best_record_path, final_results)
    else:
        step2_best_record_path = args.output_dir / "step2_best_record.json"
        if not step2_best_record_path.exists():
            dump_json(step2_best_record_path, final_results)

    recalled_samples = sum(1 for result in final_results if result["recalled_correctly"])
    recall_rate = (recalled_samples / total_samples) * 100 if total_samples > 0 else 0.0
    analyzable_samples = [result for result in final_results if result.get("gold_status") == "success"]
    total_analyzable_samples = len(analyzable_samples)
    hits_in_top1 = sum(1 for result in analyzable_samples if result["gold_in_top1_bucket"])
    hits_in_top2 = sum(1 for result in analyzable_samples if result["gold_in_top2_buckets"])
    hits_in_top3 = sum(1 for result in analyzable_samples if result["gold_in_top3_buckets"])
    top1_hit_rate = (hits_in_top1 / total_analyzable_samples) * 100 if total_analyzable_samples > 0 else 0.0
    top2_hit_rate = (hits_in_top2 / total_analyzable_samples) * 100 if total_analyzable_samples > 0 else 0.0
    top3_hit_rate = (hits_in_top3 / total_analyzable_samples) * 100 if total_analyzable_samples > 0 else 0.0

    report = f"""
    ============================================================
    SQL Generation and Self-Consistency Report
    ============================================================
    Total samples: {total_samples}
    Execution-correct samples: {correct_predictions}
    Execution Accuracy (EX): {exec_accuracy:.2f}%
    Execution Recall: {recall_rate:.2f}%

    Gold-executable samples: {total_analyzable_samples}
    Top-1 hit rate: {top1_hit_rate:.2f}%
    Top-2 hit rate: {top2_hit_rate:.2f}%
    Top-3 hit rate: {top3_hit_rate:.2f}%
    ============================================================
    """
    print(report)
    report_path = args.output_dir / f"report-sql-generation-{time.strftime('%Y%m%d-%H%M%S')}-{exec_accuracy:.2f}.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Step2 finished. Output: {latest_record_path}")

    release_vllm_model(llm, stage_name="step2")
    return final_results


# =============================================================================
# Stage 3: Pairwise rerank tournament
# =============================================================================
def run_step3(args: argparse.Namespace, step2_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run global-round pairwise rerank tournament on execution buckets."""
    logger.info("Step3 started: pairwise rerank.")

    llm = LLM(
        model=args.selector_model_path,
        tokenizer=args.selector_model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.step3_max_num_seqs,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.selector_model_path, local_files_only=args.local_files_only)
    sampling_params = SamplingParams(n=1, temperature=0.0, max_tokens=args.step3_max_new_tokens, stop=[tokenizer.eos_token])

    # Initialize one tournament state per data point.
    tournament_states: List[Dict[str, Any]] = []
    for dp_index, dp in enumerate(step2_results):
        candidates: List[Dict[str, Any]] = []
        if "execution_buckets" in dp and dp["execution_buckets"]:
            for bucket in dp["execution_buckets"]:
                for sql in bucket.get("sqls", []):
                    candidates.append({"sql": sql, "result": bucket.get("execution_result")})

        random.shuffle(candidates)
        tournament_states.append(
            {
                "dp_index": dp_index,
                "db_id": dp["db_id"],
                "base_info": dp,
                "candidates": candidates,
                "is_finished": len(candidates) <= 1,
                "next_round_candidates": [],
                "final_sql": candidates[0] if len(candidates) == 1 else None,
            }
        )

    global_round_num = 1
    # Global synchronized tournament rounds.
    while True:
        if all(state["is_finished"] for state in tournament_states):
            logger.info("Step3 all rerank tournaments finished.")
            break

        # Collect all pairwise matches for this global round.
        prompts_for_this_global_round: List[str] = []
        pairs_metadata: List[Dict[str, Any]] = []

        active_tournaments = sum(1 for state in tournament_states if not state["is_finished"])
        pbar = tqdm(total=active_tournaments, desc=f"Step3-Round {global_round_num} collecting")

        weird_count = 0
        equal_count = 0
        other_count = 0

        for state in tournament_states:
            if state["is_finished"]:
                continue

            current_candidates = state["candidates"]
            state["next_round_candidates"] = []
            random.shuffle(current_candidates)

            for i in range(0, len(current_candidates), 2):
                if i + 1 >= len(current_candidates):
                    state["next_round_candidates"].append(current_candidates[i])
                    continue

                pair = (current_candidates[i], current_candidates[i + 1])
                sql1_info, sql2_info = pair

                try:
                    res1 = [tuple(row) for row in sql1_info["result"]]
                    res2 = [tuple(row) for row in sql2_info["result"]]
                    if isinstance(res1, list) and isinstance(res2, list) and set(res1) == set(res2):
                        equal_count += 1
                        state["next_round_candidates"].append(random.choice(pair))
                        continue
                    other_count += 1
                except Exception as exc:
                    print("Exception:", exc)
                    pass

                dp_info = state["base_info"]
                question = get_question(dp_info)
                prompt = generate_pairwise_prompt(
                    schema=dp_info["prepared_schema_str"],
                    question=question,
                    sql1=sql1_info["sql"],
                    result1=format_sql_results_for_llm(sql1_info["result"]),
                    sql2=sql2_info["sql"],
                    result2=format_sql_results_for_llm(sql2_info["result"]),
                )
                messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": RERANK_RESPONSE_TEMPLATE}]
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
                prompts_for_this_global_round.append(formatted_prompt)
                pairs_metadata.append({"dp_index": state["dp_index"], "pair": pair})

            pbar.update(1)
        pbar.close()

        logger.info(f"Step3 round {global_round_num} skip stats: weird={weird_count}, equal={equal_count}, compared={other_count}, prompts={len(prompts_for_this_global_round)}")

        if prompts_for_this_global_round:
            # One batched inference call for all matches in the round.
            outputs = llm.generate(prompts_for_this_global_round, sampling_params)
            for i, output in enumerate(tqdm(outputs, desc=f"Step3-Round {global_round_num} dispatching")):
                response_text = output.outputs[0].text
                metadata = pairs_metadata[i]
                dp_index = metadata["dp_index"]
                pair = metadata["pair"]

                chose_sql1 = r"\box{SQL1}" in response_text
                chose_sql2 = r"\box{SQL2}" in response_text
                if chose_sql1 and not chose_sql2:
                    winner = pair[0]
                elif chose_sql2 and not chose_sql1:
                    winner = pair[1]
                else:
                    logger.warning(f"Model made no clear selection for db_id={tournament_states[dp_index]['db_id']}; choosing randomly.")
                    winner = random.choice(pair)
                tournament_states[dp_index]["next_round_candidates"].append(winner)

        # Advance each tournament to the next round.
        for state in tournament_states:
            if state["is_finished"]:
                continue
            state["candidates"] = state["next_round_candidates"]
            if len(state["candidates"]) <= 1:
                state["is_finished"] = True
                if state["candidates"]:
                    state["final_sql"] = state["candidates"][0]
                else:
                    state["final_sql"] = {"sql": "SELECT 'ERROR: No candidates left'", "result": "[]"}

        global_round_num += 1

    # Build final output and compute rerank accuracy against gold execution result.
    final_selection_results: List[Dict[str, Any]] = []
    for state in tournament_states:
        dp = state["base_info"]
        question = get_question(dp)

        selected_sql = "SELECT 'ERROR: No final SQL selected'"
        selected_result: Any = []
        if state["final_sql"]:
            selected_sql = state["final_sql"]["sql"]
            selected_result = state["final_sql"]["result"]

        is_correct = False
        try:
            gold_result = dp["gold_sql_result"]
            selected_result = [tuple(item) for item in selected_result]
            gold_result = [tuple(item) for item in gold_result]
            is_correct = set(selected_result) == set(gold_result)
        except Exception as exc:
            logger.error(f"Step3 comparison failed for db_id={dp.get('db_id')}: {exc}")

        gold_sql = dp["query"] if "query" in dp else dp["sql"]
        final_selection_results.append(
            {
                "db_id": dp["db_id"],
                "question": question,
                "gold_sql": gold_sql,
                "gold_sql_result": dp.get("gold_sql_result"),
                "selected_sql_after_rerank": selected_sql,
                "selected_sql_result": selected_result,
                "rerank_is_correct": is_correct,
            }
        )

    total_samples = len(final_selection_results)
    correct_count = sum(1 for result in final_selection_results if result["rerank_is_correct"])
    accuracy = (correct_count / total_samples) * 100 if total_samples > 0 else 0.0

    step3_output_path = args.output_dir / "step3_results.json"
    dump_json(step3_output_path, final_selection_results)

    report = f"""
    ============================================================
    Step3 Pairwise CoT Rerank Report
    ============================================================
    Rerank model: {args.selector_model_path}
    ------------------------------------------------------------
    Total samples: {total_samples}
    Correct after rerank: {correct_count}
    Final execution accuracy: {accuracy:.2f}%
    ============================================================
    """
    print(report)
    report_path = args.output_dir / f"step3_rerank_report-{time.strftime('%Y%m%d-%H%M')}-{accuracy:.2f}.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Step3 finished. Output: {step3_output_path}")

    release_vllm_model(llm, stage_name="step3")
    return final_selection_results


# =============================================================================
# Entrypoint
# =============================================================================
def main() -> int:
    """Run the full four-stage pipeline in a single process."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)

    logger.info(f"Unified pipeline started for benchmark={args.evaluation_benchmark}")

    step0_results = run_step0(args)
    step1_results = run_step1(args, step0_results)
    step2_results = run_step2(args, step1_results)
    run_step3(args, step2_results)

    logger.success("pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
