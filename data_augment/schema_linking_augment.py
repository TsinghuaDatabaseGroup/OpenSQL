#!/usr/bin/env python3
# data_augment/schema_linking_augment.py
# End-to-end pipeline: schema linking -> noise augmentation -> SFT/DPO training data generation.
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import sqlparse
from loguru import logger
from sqlglot import exp, parse_one
from sqlglot.optimizer import qualify, traverse_scope
from sqlglot.schema import MappingSchema

# ── Project paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schema_utils import IR2Schema  # noqa: E402

DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "schema_link"

SUPPORTED_BENCHES = ("Spider_train", "BIRD_train")

BENCH_DATASET_PATHS: dict[str, Path] = {
    "Spider_train": DATASET_ROOT / "spider_data" / "train.json",
    "BIRD_train": DATASET_ROOT / "BIRD_train" / "train.json",
}

BENCH_DB_ROOTS: dict[str, Path] = {
    "Spider_train": DATASET_ROOT / "spider_data" / "database",
    "BIRD_train": DATASET_ROOT / "BIRD_train" / "train_databases",
}

BENCH_IR_PATHS: dict[str, Path] = {
    "Spider_train": PROJECT_ROOT / "artifacts" / "ir" / "Spider_train.json",
    "BIRD_train": PROJECT_ROOT / "artifacts" / "ir" / "BIRD_train.json",
}

# ── DPO defaults ─────────────────────────────────────────────────────
DEFAULT_DPO_RATIO = 0.3  # 30% of db_ids are assigned to DPO; the rest become SFT
DEFAULT_DPO_TOLERANCE = 0.02  # Accept actual DPO datapoint ratio within ±2% of target
DEFAULT_NUM_COL_SAMPLES = 3  # Generate 3 column-deletion negative samples per DPO datapoint

# ── Local schema linking defaults ─────────────────────────────────────
DEFAULT_LOCAL_FALSE_TO_TRUE_RATIO = 3.0

# ── Noise defaults ───────────────────────────────────────────────────
# Noise tables: randomly pick from this range to decide how many distractor tables to add
NOISE_TABLES_RANGE = (0, 1, 2)
# For each noise table, include its PKs plus 2-4 random non-PK columns
NOISE_TABLE_COLS_RANGE = (2, 4)
# Each GT table has a 50% chance of receiving extra noise columns
NOISE_COL_PROB = 0.5
# When triggered, add 1-2 random non-PK/FK columns to the GT table
NOISE_COLS_RANGE = (1, 2)

# =====================================================================
# Phase 1 — Schema Linking
# =====================================================================

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


def build_schema_cache(
    db_root: Path,
) -> tuple[dict[str, dict], dict[str, MappingSchema]]:
    """Walk all SQLite databases under *db_root*, return (schema_dict, mapping_schema_cache)."""
    schema_dict: dict[str, dict] = {}
    mapping_cache: dict[str, MappingSchema] = {}

    for db_dir in sorted(db_root.iterdir()):
        if not db_dir.is_dir() or db_dir.name == ".DS_Store":
            continue
        db_id = db_dir.stem
        db_filepath = next(db_dir.glob("*.sqlite"), None)
        if not db_filepath:
            continue
        try:
            conn = sqlite3.connect(f"file:{db_filepath}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            schema: dict[str, dict] = {}
            for (original_table_name,) in cursor.fetchall():
                table_lower = original_table_name.lower()
                try:
                    cursor.execute(f'PRAGMA table_info("{original_table_name}")')
                    columns_info = cursor.fetchall()
                except sqlite3.OperationalError:
                    logger.warning(f"Cannot read table '{original_table_name}' info (DB: {db_id})")
                    continue

                columns_map: dict[str, str] = {}
                pk_columns: list[str] = []
                columns_ordered: list[str] = []
                for col_info in columns_info:
                    col_name = col_info[1]
                    col_lower = col_name.lower()
                    columns_map[col_lower] = col_name
                    columns_ordered.append(col_lower)
                    if col_info[5] > 0:
                        pk_columns.append(col_lower)

                schema[table_lower] = {
                    "original_case": original_table_name,
                    "columns": columns_map,
                    "pk_columns": pk_columns,
                    "columns_ordered": columns_ordered,
                }

            schema_dict[db_id] = schema
            mapping_cache[db_id] = MappingSchema({tbl: {col: "text" for col in info["columns"]} for tbl, info in schema.items()})
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Database connection/query failed: {db_filepath}, error: {e}")
    return schema_dict, mapping_cache


# ── SQL analysis ─────────────────────────────────────────────────────


def _resolve_table_alias(scope_obj, alias_lower: str) -> str:
    """Resolve alias up the scope chain: return real table name for physical tables, alias itself for subqueries."""
    cur = scope_obj
    while cur:
        if alias_lower in cur.sources:
            source = cur.sources[alias_lower]
            if isinstance(source, exp.Table):
                return source.name.lower()
            return alias_lower
        cur = cur.parent
    return alias_lower


def analyze_query(
    sql: str,
    mapping_schema: MappingSchema,
) -> dict[str, list[str]] | None:
    """Qualify and traverse scopes of *sql*, return {table_lower: [col_lower, ...]} or None on failure."""

    def try_qualify(text: str):
        parsed_expr = parse_one(text, dialect="sqlite")
        return qualify.qualify(
            parsed_expr,
            schema=mapping_schema,
            dialect="sqlite",
            quote_identifiers=False,
            identify=False,
            expand_stars=True,
            qualify_columns=True,
            allow_partial_qualification=True,
        )

    standardized = sqlparse.format(sql, keyword_case="upper", identifier_case="upper", strip_comments=True)
    standardized = " ".join(standardized.split())

    # Try parsing
    try:
        parse_one(standardized, dialect="sqlite")
    except Exception:
        return None

    qualified = None
    try:
        qualified = try_qualify(standardized)
    except Exception:
        alt = standardized.replace('"', "'")
        if alt != standardized:
            try:
                qualified = try_qualify(alt)
            except Exception:
                pass
    if qualified is None:
        return None

    appeared_tables: dict[str, list[str]] = {}
    tables_in_query: set[str] = set()

    for scope in traverse_scope(qualified):
        alias_to_table: dict[str, str] = {}
        for alias, source in scope.sources.items():
            if isinstance(source, exp.Table):
                base = source.name.lower()
                alias_to_table[alias.lower()] = base
                tables_in_query.add(base)
        for table_expr in scope.tables:
            tables_in_query.add(table_expr.name.lower())
        for column in scope.columns:
            if not column.table:
                continue
            col_table = column.table.lower()
            base_table = alias_to_table.get(col_table, _resolve_table_alias(scope, col_table))
            col_lower = column.name.lower()
            appeared_tables.setdefault(base_table, [])
            if col_lower not in appeared_tables[base_table]:
                appeared_tables[base_table].append(col_lower)
            tables_in_query.add(base_table)

    for tbl in tables_in_query:
        appeared_tables.setdefault(tbl, [])

    return appeared_tables


# ── Formatting & validation ──────────────────────────────────────────


def build_schema_link(
    appeared_tables: dict[str, list[str]],
    schema: dict[str, dict],
) -> tuple[dict[str, list[str]], dict[str, list[str]]] | None:
    """Convert analyze_query results into final schema_link + primary_keys. Return None on validation failure."""
    final_link: dict[str, list[str]] = {}
    primary_keys: dict[str, list[str]] = {}

    for table_lower, cols_lower in appeared_tables.items():
        if table_lower not in schema:
            continue
        tbl_info = schema[table_lower]
        table_original = tbl_info["original_case"]
        pk_cols = tbl_info.get("pk_columns", [])

        linked = set(cols_lower) | set(pk_cols)

        # Validate column existence
        for c in linked:
            if c not in tbl_info["columns"]:
                return None

        ordered = [col for col in tbl_info["columns_ordered"] if col in linked]
        if not ordered:
            continue

        cols_original = [tbl_info["columns"][c] for c in ordered]
        final_link[table_original] = cols_original

        pk_original = [tbl_info["columns"][pk] for pk in pk_cols if pk in tbl_info["columns"]]
        if pk_original:
            primary_keys[table_original] = pk_original

    # Fallback: when tables appear but no columns (e.g. COUNT(*)), use PKs or all columns
    if not final_link:
        for tbl_lower in appeared_tables:
            if tbl_lower not in schema:
                continue
            tbl_info = schema[tbl_lower]
            pk_cols = tbl_info.get("pk_columns", [])
            cols = pk_cols if pk_cols else list(tbl_info["columns"].keys())
            cols_original = [tbl_info["columns"][c] for c in cols if c in tbl_info["columns"]]
            if cols_original:
                table_original = tbl_info["original_case"]
                final_link[table_original] = cols_original
                if pk_cols:
                    primary_keys[table_original] = [tbl_info["columns"][pk] for pk in pk_cols if pk in tbl_info["columns"]]

    return (final_link, primary_keys) if final_link else None


# ── Phase 1 entry point ─────────────────────────────────────────────


def _linearize(schema_link: dict) -> str:
    return "```json\n" + json.dumps(schema_link, indent=2, ensure_ascii=False) + "\n```"


def run_schema_linking(bench: str) -> list[dict]:
    """Run schema linking for *bench*, return list of annotated datapoints."""
    dataset_path = BENCH_DATASET_PATHS[bench]
    db_root = BENCH_DB_ROOTS[bench]

    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        return []
    if not db_root.exists():
        logger.error(f"Database directory not found: {db_root}")
        return []

    dataset: list[dict] = json.loads(dataset_path.read_text("utf-8"))
    logger.info(f"[{bench}] Dataset contains {len(dataset)} records")

    logger.info(f"[{bench}] Parsing database schemas...")
    schema_dict, mapping_cache = build_schema_cache(db_root)
    logger.info(f"[{bench}] Schema parsing done, {len(schema_dict)} databases")

    is_spider = "Spider" in bench
    sql_key = "query" if is_spider else "SQL"
    results: list[dict] = []

    for i, dp in enumerate(dataset):
        original_query = dp[sql_key]
        db_id = dp["db_id"]

        # Filter
        if "ref_company_types" in original_query.lower():
            continue
        if bench == "BIRD_train" and db_id == "retail_world":
            continue
        if " join " in original_query.lower() and "on" not in original_query.lower() and "using" not in original_query.lower():
            logger.warning(f"[{bench}] Skipping JOIN without ON/USING (Index: {i}, DB: {db_id})")
            continue

        if db_id not in schema_dict or db_id not in mapping_cache:
            continue

        appeared = analyze_query(original_query, mapping_cache[db_id])
        if appeared is None:
            logger.warning(f"[{bench}] SQL parsing failed (Index: {i}, DB: {db_id})")
            continue

        link_result = build_schema_link(appeared, schema_dict[db_id])
        if link_result is None:
            logger.warning(f"[{bench}] Schema validation failed (Index: {i}, DB: {db_id})")
            continue

        final_link, primary_keys = link_result

        evidence = "" if is_spider else dp.get("evidence", "")
        question = dp["question"]
        question_with_evidence = question + (f"\n(hint: {evidence.strip()})" if len(evidence.strip()) >= 5 else "")

        formatted_query = original_query.strip()
        if not formatted_query.endswith(";"):
            formatted_query += ";"
        formatted_query = sqlparse.format(formatted_query, keyword_case="upper", reindent=False, strip_whitespace=True)
        formatted_query = " ".join(formatted_query.split())

        results.append(
            {
                "benchmark": bench,
                "db_id": db_id,
                "question": question,
                "evidence": evidence,
                "question_with_evidence": question_with_evidence,
                "original_query": original_query,
                "query": formatted_query,
                "gt_schema_link": final_link,
                "related_primary_keys": primary_keys,
                "gt_schema_link_linearized": _linearize(final_link),
            }
        )

    logger.success(f"[{bench}] Schema linking done, {len(results)} valid datapoints")
    return results


def _get_heuristic_columns(table_ir: dict) -> list[str]:
    """Select PK columns + a few random non-PK columns for a noise table."""
    all_cols = table_ir.get("columns", [])
    pk_indices = table_ir.get("primary_keys", [])
    pk_names = {c["col_name"] for c in all_cols if c["col_idx"] in pk_indices}
    non_pk = [c["col_name"] for c in all_cols if c["col_idx"] not in pk_indices]
    n = min(random.randint(*NOISE_TABLE_COLS_RANGE), len(non_pk))
    return list(pk_names) + random.sample(non_pk, n)


def _add_noise_cols_to_gt_tables(noised: dict[str, list[str]], db_ir: dict, gt_tables: set[str]) -> None:
    """In-place: randomly add non-PK/FK columns to GT tables."""
    for table_name in gt_tables:
        if random.random() > NOISE_COL_PROB:
            continue
        table_ir = next((t for t in db_ir["tables"] if t["table_name"] == table_name), None)
        if not table_ir:
            continue
        all_cols_info = table_ir.get("columns", [])
        pk_indices = table_ir.get("primary_keys", [])
        pk_names = {c["col_name"] for c in all_cols_info if c["col_idx"] in pk_indices}
        fk_names = {fk["column"].strip('"') for fk in table_ir.get("foreign_keys", [])}
        current = set(noised.get(table_name, []))
        forbidden = pk_names | fk_names | current
        candidates = [c["col_name"] for c in all_cols_info if c["col_name"] not in forbidden]
        if not candidates:
            continue
        n = min(random.randint(*NOISE_COLS_RANGE), len(candidates))
        noised[table_name] = sorted(set(noised.get(table_name, [])) | set(random.sample(candidates, n)))


def _add_noise_to_datapoint(datapoint: dict, ir_db_map: dict) -> dict:
    """Add ``noised_schema_linking`` field to *datapoint*."""
    db_id = datapoint["db_id"]
    db_ir = ir_db_map.get(db_id)
    if not db_ir:
        datapoint["noised_schema_linking"] = datapoint["gt_schema_link"]
        return datapoint

    gt_sl = datapoint["gt_schema_link"]
    gt_tables = set(gt_sl.keys())

    # Start with a copy of GT schema link
    noised: dict[str, list[str]] = {t: cols.copy() for t, cols in gt_sl.items()}

    # Stage 1: add distractor tables
    all_tables = {t["table_name"] for t in db_ir.get("tables", [])}
    distractors = list(all_tables - gt_tables)
    if distractors:
        n = min(random.choice(NOISE_TABLES_RANGE), len(distractors))
        chosen = random.sample(distractors, n)
        for tbl_name in chosen:
            tbl_ir = next((t for t in db_ir["tables"] if t["table_name"] == tbl_name), None)
            if tbl_ir:
                noised[tbl_name] = _get_heuristic_columns(tbl_ir)
        # FK enhancement: if GT tables have FKs pointing to newly added noise tables, add those FK cols
        if chosen:
            added_set = set(chosen)
            for src_ir in db_ir.get("tables", []):
                if src_ir["table_name"] not in gt_tables:
                    continue
                for fk in src_ir.get("foreign_keys", []):
                    if fk["referenced_table"].strip('"') in added_set:
                        fk_col = fk["column"].strip('"')
                        src_name = src_ir["table_name"]
                        if fk_col not in noised.get(src_name, []):
                            noised.setdefault(src_name, []).append(fk_col)
                            noised[src_name] = sorted(set(noised[src_name]))

    # Stage 2: add noise columns to GT tables
    _add_noise_cols_to_gt_tables(noised, db_ir, gt_tables)

    datapoint["noised_schema_linking"] = noised if noised else gt_sl
    return datapoint


def run_noise_generation(schema_links: list[dict], ir_set: list[dict]) -> None:
    """In-place: add ``noised_schema_linking`` to every datapoint in *schema_links*."""
    logger.info("Phase 1.5: adding noise augmentation...")
    ir_db_map = {ir["db_id"]: ir for ir in ir_set}
    for dp in schema_links:
        _add_noise_to_datapoint(dp, ir_db_map)
    logger.success(f"Phase 1.5 done: noised {len(schema_links)} datapoints")


# =====================================================================
# Phase 1.4 — Local Schema Linking Training Data
# =====================================================================


def _extract_non_key_columns(table_ir: dict) -> list[str]:
    """Return non-PK/FK columns in original order for a table in IR."""
    columns = table_ir.get("columns", [])
    pk_indices = set(table_ir.get("primary_keys", []))
    pk_names = {c["col_name"].lower() for c in columns if c["col_idx"] in pk_indices}
    fk_names = {fk["column"].strip('"').lower() for fk in table_ir.get("foreign_keys", [])}
    forbidden = pk_names | fk_names
    return [c["col_name"] for c in columns if c["col_name"].lower() not in forbidden]


def _make_local_prompt(converter: IR2Schema, question: str, table_name: str, column_name: str) -> str | None:
    """Render one local-classification prompt for (table, column)."""
    try:
        table_statement, column_value_examples = converter.render_table_and_column_examples(table_name, column_name)
    except Exception as exc:
        logger.warning(f"Failed to render local prompt for {table_name}.{column_name}: {exc}")
        return None
    return LOCAL_CLASSIFICATION_TEMPLATE.format(
        table_schema=table_statement,
        question_with_evidence=question,
        column_value_examples_prompt=column_value_examples,
        column_name=column_name,
    )


def _build_local_samples_for_datapoint(
    datapoint: dict,
    db_ir: dict,
    false_to_true_ratio: float,
) -> list[dict]:
    """
    Build local schema-linking classification samples for one datapoint.

    Strategy:
    - Positives: non-PK/FK columns inside GT tables that appear in ``gt_schema_link``.
    - Negatives: prioritize non-PK/FK columns from GT tables, then backfill from non-GT tables.
    """
    gt_schema_link = datapoint.get("gt_schema_link", {})
    if not gt_schema_link:
        return []

    question = datapoint.get("question_with_evidence") or datapoint.get("question", "")
    if not question:
        return []

    tables_by_lower = {t["table_name"].lower(): t for t in db_ir.get("tables", [])}

    positive_pairs: set[tuple[str, str]] = set()
    hard_negative_pool: set[tuple[str, str]] = set()
    selected_tables_lower: set[str] = set()

    # Build positives and hard negatives from GT tables.
    for gt_table_name, linked_cols in gt_schema_link.items():
        table_ir = tables_by_lower.get(gt_table_name.lower())
        if table_ir is None:
            continue

        table_name = table_ir["table_name"]
        table_lower = table_name.lower()
        selected_tables_lower.add(table_lower)

        non_key_cols = _extract_non_key_columns(table_ir)
        linked_cols_lower = {c.lower() for c in linked_cols}

        table_positive: set[str] = {c for c in non_key_cols if c.lower() in linked_cols_lower}
        for col in table_positive:
            positive_pairs.add((table_name, col))
        for col in non_key_cols:
            if col not in table_positive:
                hard_negative_pool.add((table_name, col))

    # If no non-key positive columns, skip this datapoint to avoid all-negative data.
    if not positive_pairs:
        return []

    # Build easy negative pool from non-GT tables.
    easy_negative_pool: set[tuple[str, str]] = set()
    for table_ir in db_ir.get("tables", []):
        table_name = table_ir["table_name"]
        if table_name.lower() in selected_tables_lower:
            continue
        for col in _extract_non_key_columns(table_ir):
            easy_negative_pool.add((table_name, col))

    positive_list = sorted(positive_pairs)
    hard_negatives = sorted(hard_negative_pool - positive_pairs)
    easy_negatives = sorted(easy_negative_pool - positive_pairs)

    target_negatives = int(math.ceil(len(positive_list) * false_to_true_ratio))
    target_negatives = max(target_negatives, 0)

    sampled_negatives: list[tuple[str, str, str]] = []
    if target_negatives > 0:
        n_hard = min(target_negatives, len(hard_negatives))
        if n_hard > 0:
            sampled_negatives.extend([(t, c, "hard") for t, c in random.sample(hard_negatives, n_hard)])
        remaining = target_negatives - n_hard
        if remaining > 0 and easy_negatives:
            n_easy = min(remaining, len(easy_negatives))
            sampled_negatives.extend([(t, c, "easy") for t, c in random.sample(easy_negatives, n_easy)])

    converter = IR2Schema(
        ir=db_ir,
        chosen=None,
        tindex=None,
        question=question,
        emb_model=None,
        print_contain_null=False,
    )

    local_samples: list[dict] = []
    for table_name, column_name in positive_list:
        prompt = _make_local_prompt(converter, question, table_name, column_name)
        if prompt is None:
            continue
        local_samples.append(
            {
                "benchmark": datapoint["benchmark"],
                "db_id": datapoint["db_id"],
                "table_name": table_name,
                "column_name": column_name,
                "label": "True",
                "neg_type": "",
                "prompt": prompt,
            }
        )

    for table_name, column_name, neg_type in sampled_negatives:
        prompt = _make_local_prompt(converter, question, table_name, column_name)
        if prompt is None:
            continue
        local_samples.append(
            {
                "benchmark": datapoint["benchmark"],
                "db_id": datapoint["db_id"],
                "table_name": table_name,
                "column_name": column_name,
                "label": "False",
                "neg_type": neg_type,
                "prompt": prompt,
            }
        )

    random.shuffle(local_samples)
    return local_samples


def run_local_schema_linking_data_generation(
    schema_links: list[dict],
    ir_set: list[dict],
    *,
    false_to_true_ratio: float,
) -> list[dict]:
    """Generate local schema-linking train samples compatible with Local/schema_local.py."""
    logger.info("Phase 1.4: building local schema-linking training data...")
    ir_db_map = {ir["db_id"]: ir for ir in ir_set}

    all_local_samples: list[dict] = []
    skipped_no_ir = 0
    skipped_no_positive = 0

    for dp in schema_links:
        db_ir = ir_db_map.get(dp["db_id"])
        if db_ir is None:
            skipped_no_ir += 1
            continue
        one_dp_samples = _build_local_samples_for_datapoint(
            datapoint=dp,
            db_ir=db_ir,
            false_to_true_ratio=false_to_true_ratio,
        )
        if not one_dp_samples:
            skipped_no_positive += 1
            continue
        all_local_samples.extend(one_dp_samples)

    logger.success(
        "Phase 1.4 done: generated "
        f"{len(all_local_samples)} local samples "
        f"(skipped_no_ir={skipped_no_ir}, skipped_no_positive={skipped_no_positive})"
    )
    return all_local_samples


# =====================================================================
# Phase 1.6 — Render Noised Schemas
# =====================================================================


def run_schema_rendering(schema_links: list[dict], ir_set: list[dict]) -> None:
    """In-place: render ``noised_schema_linking`` into ``dynamic_noised_schema`` text via IR2Schema."""
    logger.info("Phase 1.6: rendering noised schemas...")
    ir_db_map = {ir["db_id"]: ir for ir in ir_set}
    rendered = 0
    for dp in schema_links:
        db_ir = ir_db_map.get(dp["db_id"])
        chosen = dp.get("noised_schema_linking", dp.get("gt_schema_link"))
        if not db_ir or not chosen:
            dp["dynamic_noised_schema"] = ""
            continue
        converter = IR2Schema(
            ir=db_ir,
            chosen=chosen,
            tindex=None,
            question=None,
            emb_model=None,
            print_contain_null=False,
        )
        schema_text, _ = converter.render_schema()
        dp["dynamic_noised_schema"] = schema_text
        rendered += 1
    logger.success(f"Phase 1.6 done: rendered {rendered} schemas")


# =====================================================================
# Phase 2 — DPO / SFT Training Data Generation
# =====================================================================


def _prepare_dpo_base(datapoint: dict, mess_type: str) -> dict:
    """Prepare a DPO sample skeleton via deep copy."""
    dpo_sample = copy.deepcopy(datapoint)
    dpo_sample["mess_type"] = mess_type
    dpo_sample["train_type"] = "DPO"
    return dpo_sample


def _synthesize_column_deletion_sample(datapoint: dict, ir_db_map: dict) -> dict | None:
    """Create a negative sample by randomly deleting 1-2 non-PK/FK columns."""
    db_id = datapoint["db_id"]
    db_ir = ir_db_map.get(db_id)
    if not db_ir:
        return None

    dpo_sample = _prepare_dpo_base(datapoint, "DELETE_COLUMN")
    win_sl_dict = datapoint["gt_schema_link"]

    # Identify non-deletable columns (PKs and FKs)
    non_deletable: dict[str, set[str]] = defaultdict(set)
    for table_name in win_sl_dict:
        table_ir = next((t for t in db_ir.get("tables", []) if t["table_name"] == table_name), None)
        if not table_ir:
            continue
        pk_indices = table_ir.get("primary_keys", [])
        for col in table_ir.get("columns", []):
            if col["col_idx"] in pk_indices:
                non_deletable[table_name].add(col["col_name"])
        for fk in table_ir.get("foreign_keys", []):
            non_deletable[table_name].add(fk["column"].strip('"'))

    # Build deletable column pool
    pool = []
    for table_name, cols in win_sl_dict.items():
        for col_name in cols:
            if col_name not in non_deletable[table_name]:
                pool.append((table_name, col_name))

    if not pool:
        return None

    num_to_delete = 1 if len(pool) == 1 or random.random() < 0.7 else 2
    num_to_delete = min(num_to_delete, len(pool))
    selected = random.sample(pool, num_to_delete)

    loss_sl = copy.deepcopy(win_sl_dict)
    for table, col in selected:
        if col in loss_sl.get(table, []):
            loss_sl[table].remove(col)

    loss_sl = {k: v for k, v in loss_sl.items() if v}
    if not loss_sl or loss_sl == win_sl_dict:
        return None

    dpo_sample["loss_schema_link"] = loss_sl
    dpo_sample["loss_schema_link_linearized"] = _linearize(loss_sl)
    return dpo_sample


def _synthesize_table_deletion_sample(datapoint: dict, ir_db_map: dict, table_to_delete: str) -> dict | None:
    """Create a negative sample by deleting a table and cleaning up dangling FK columns."""
    db_id = datapoint["db_id"]
    db_ir = ir_db_map.get(db_id)
    if not db_ir:
        return None

    dpo_sample = _prepare_dpo_base(datapoint, "DELETE_TABLE")
    win_sl_dict = datapoint["gt_schema_link"]
    loss_sl = copy.deepcopy(win_sl_dict)

    if table_to_delete not in loss_sl:
        return None
    del loss_sl[table_to_delete]

    # Clean up FK columns that referenced the deleted table
    cleanup: dict[str, set[str]] = defaultdict(set)
    for src_table in loss_sl:
        src_ir = next((t for t in db_ir.get("tables", []) if t["table_name"] == src_table), None)
        if not src_ir:
            continue
        for fk in src_ir.get("foreign_keys", []):
            if fk["referenced_table"].strip('"') == table_to_delete:
                cleanup[src_table].add(fk["column"].strip('"'))

    for table_name, cols_to_remove in cleanup.items():
        if table_name in loss_sl:
            loss_sl[table_name] = list(set(loss_sl[table_name]) - cols_to_remove)

    loss_sl = {k: v for k, v in loss_sl.items() if v}
    if not loss_sl or loss_sl == win_sl_dict:
        return None

    dpo_sample["loss_schema_link"] = loss_sl
    dpo_sample["loss_schema_link_linearized"] = _linearize(loss_sl)
    return dpo_sample


def run_dpo_generation(
    schema_links: list[dict],
    ir_set: list[dict],
    *,
    dpo_ratio: float,
    dpo_tolerance: float,
    num_col_samples: int,
    generate_table_deletion: bool,
) -> tuple[list[dict], list[dict]]:
    """Split schema_links into SFT / DPO sets and synthesize negative samples. Return (sft_data, dpo_data)."""
    logger.info("Phase 2: generating SFT/DPO training data...")

    ir_db_map = {ir["db_id"]: ir for ir in ir_set}
    all_db_ids = sorted({sl["db_id"] for sl in schema_links})

    # Count datapoints per db_id for ratio-aware splitting
    db_counts: dict[str, int] = defaultdict(int)
    for sl in schema_links:
        db_counts[sl["db_id"]] += 1

    total = len(schema_links)
    target = total * dpo_ratio
    margin = total * dpo_tolerance
    lo, hi = target - margin, target + margin
    logger.info(f"Total datapoints: {total}, target DPO range: [{int(lo)}, {int(hi)}]")

    # Shuffle db_ids until DPO datapoint count falls within tolerance
    dpo_db_ids: set[str] = set()
    sft_db_ids: set[str] = set()
    for attempt in range(1, 1002):
        random.shuffle(all_db_ids)
        n = int(len(all_db_ids) * dpo_ratio)
        candidate = set(all_db_ids[:n])
        count = sum(db_counts[d] for d in candidate)
        if lo <= count <= hi:
            dpo_db_ids = candidate
            sft_db_ids = set(all_db_ids) - dpo_db_ids
            logger.success(f"Found valid split after {attempt} attempts (DPO: {count}/{total} = {count / total:.2%})")
            break
        if attempt == 1001:
            logger.warning("Exceeded 1000 attempts, using last split")
            dpo_db_ids = candidate
            sft_db_ids = set(all_db_ids) - dpo_db_ids

    logger.info(f"DBs: {len(dpo_db_ids)} DPO, {len(sft_db_ids)} SFT | col_samples={num_col_samples}, table_deletion={generate_table_deletion}")

    sft_data: list[dict] = []
    dpo_data: list[dict] = []

    for dp in schema_links:
        db_id = dp["db_id"]
        if db_id in sft_db_ids:
            sft_sample = copy.deepcopy(dp)
            sft_sample["train_type"] = "SFT"
            sft_data.append(sft_sample)
        elif db_id in dpo_db_ids:
            for _ in range(num_col_samples):
                sample = _synthesize_column_deletion_sample(dp, ir_db_map)
                if sample:
                    dpo_data.append(sample)
            if generate_table_deletion:
                tables = list(dp.get("gt_schema_link", {}).keys())
                if len(tables) > 1:
                    for tbl in tables:
                        sample = _synthesize_table_deletion_sample(dp, ir_db_map, tbl)
                        if sample:
                            dpo_data.append(sample)

    logger.success(f"Phase 2 done: {len(sft_data)} SFT, {len(dpo_data)} DPO samples")
    return sft_data, dpo_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end pipeline: schema linking -> local/global training data generation.",
    )
    grp = p.add_mutually_exclusive_group(required=False)
    grp.add_argument("--bench", choices=SUPPORTED_BENCHES, help="Process one benchmark.")
    grp.add_argument("--all", action="store_true", help="Process all supported benchmarks.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    p.add_argument("--dpo-ratio", type=float, default=DEFAULT_DPO_RATIO, help="Fraction of db_ids used for DPO.")
    p.add_argument("--dpo-tolerance", type=float, default=DEFAULT_DPO_TOLERANCE, help="Tolerance around dpo-ratio.")
    p.add_argument("--num-col-samples", type=int, default=DEFAULT_NUM_COL_SAMPLES, help="Column-deletion samples per DPO datapoint.")
    p.add_argument("--no-table-deletion", action="store_true", help="Disable table-deletion negative samples.")
    p.add_argument("--no-noise", action="store_true", help="Skip noise augmentation (Phase 1.5).")
    p.add_argument(
        "--local-false-to-true-ratio",
        type=float,
        default=DEFAULT_LOCAL_FALSE_TO_TRUE_RATIO,
        help="Local schema-linking negative/positive ratio; negatives prioritize GT tables first.",
    )
    p.add_argument(
        "--full-output",
        action="store_true",
        help="Write additional intermediate files (schema_link + per-benchmark split files).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.all and args.bench is None:
        args.all = True

    if args.local_false_to_true_ratio < 0:
        logger.error("--local-false-to-true-ratio must be >= 0")
        return 1

    random.seed(args.seed)
    benches = SUPPORTED_BENCHES if args.all else (args.bench,)

    if args.all:
        output_dir = args.output_dir
    else:
        output_dir = args.output_dir / args.bench
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    all_local: list[dict] = []
    all_sft: list[dict] = []
    all_dpo: list[dict] = []

    for bench in benches:
        logger.info(f"{'=' * 60}")
        logger.info(f"Processing benchmark: {bench}")

        # Phase 1: schema linking
        schema_links = run_schema_linking(bench)
        if not schema_links:
            logger.warning(f"[{bench}] No schema links produced, skipping")
            continue

        # Load IR (needed by Phase 1.5 and Phase 2)
        ir_path = BENCH_IR_PATHS[bench]
        if not ir_path.exists():
            logger.error(f"[{bench}] IR file not found: {ir_path}. Skipping Phase 1.5 & 2.")
            continue
        ir_set: list[dict] = json.loads(ir_path.read_text("utf-8"))

        # Phase 1.4: local schema-linking train data
        local_data = run_local_schema_linking_data_generation(
            schema_links,
            ir_set,
            false_to_true_ratio=args.local_false_to_true_ratio,
        )

        # Phase 1.5: noise augmentation
        if not args.no_noise:
            run_noise_generation(schema_links, ir_set)

        # Phase 1.6: render noised schemas into text
        run_schema_rendering(schema_links, ir_set)

        # Phase 2: DPO/SFT generation
        sft_data, dpo_data = run_dpo_generation(
            schema_links,
            ir_set,
            dpo_ratio=args.dpo_ratio,
            dpo_tolerance=args.dpo_tolerance,
            num_col_samples=args.num_col_samples,
            generate_table_deletion=not args.no_table_deletion,
        )

        if args.full_output:
            (output_dir / f"{bench}_local_schema_linking.json").write_text(
                json.dumps(local_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (output_dir / f"{bench}_schema_link.json").write_text(
                json.dumps(schema_links, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (output_dir / f"{bench}_sft.json").write_text(
                json.dumps(sft_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (output_dir / f"{bench}_dpo.json").write_text(
                json.dumps(dpo_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(f"[{bench}] Saved full outputs ({len(local_data)} local, {len(sft_data)} SFT, {len(dpo_data)} DPO)")

        all_local.extend(local_data)
        all_sft.extend(sft_data)
        all_dpo.extend(dpo_data)

    # Save required training files + config
    if all_local or all_sft or all_dpo:
        # Default output names are aligned with training stage-to-file mapping.
        (output_dir / "local_schema_linking.json").write_text(
            json.dumps(all_local, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "global_schema_linking_sft.json").write_text(
            json.dumps(all_sft, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "global_schema_linking_dpo.json").write_text(
            json.dumps(all_dpo, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if args.full_output and args.all:
            (output_dir / "mixed_local_schema_linking.json").write_text(
                json.dumps(all_local, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (output_dir / "mixed_sft.json").write_text(
                json.dumps(all_sft, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (output_dir / "mixed_dpo.json").write_text(
                json.dumps(all_dpo, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        config = {
            "benchmarks": list(benches),
            "all_mode": args.all,
            "output_dir": str(output_dir),
            "local_false_to_true_ratio": args.local_false_to_true_ratio,
            "dpo_ratio": args.dpo_ratio,
            "dpo_tolerance": args.dpo_tolerance,
            "num_col_samples": args.num_col_samples,
            "table_deletion": not args.no_table_deletion,
            "noise": not args.no_noise,
            "full_output": args.full_output,
            "seed": args.seed,
        }
        (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    logger.success(
        f"All done: {len(all_local)} local + {len(all_sft)} SFT + {len(all_dpo)} DPO total -> {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
