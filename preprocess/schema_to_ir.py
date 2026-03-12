import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chardet
import pandas as pd
from tqdm import tqdm

TYPE_MAPPING = {
    "int": "integer",
    "real": "real",
    "double": "real",
    "float": "real",
    "numeric": "numeric",
    "decimal": "numeric",
    "num": "numeric",
    "year": "integer",
    "bit": "integer",
    "bool": "boolean",
    "boolean": "boolean",
    "date": "time",
    "time": "time",
    "char": "text",
    "nchar": "text",
    "varchar": "text",
    "nvarchar": "text",
    "text": "text",
    "clob": "text",
    "blob": "blob",
    "string": "text",
    "jsonb": "jsonb",
    "json": "jsonb",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_DRSPIDER_ROOT = DEFAULT_DATASET_ROOT / "diagnostic-robustness-text-to-sql" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "ir"

SUPPORTED_BENCHES = (
    "Spider_train",
    "Spider_dev",
    "Spider_test",
    "BIRD_train",
    "BIRD_dev",
)


@dataclass(frozen=True)
class BenchmarkPaths:
    db_base: Path
    table_path: Path


def normalize_type(sql_type: str) -> str:
    if not sql_type:
        return "others"
    if "character" in sql_type.lower():
        return "text"
    sql_type = sql_type.lower()
    for key, mapped in TYPE_MAPPING.items():
        if key in sql_type:
            return mapped

    assert False, sql_type
    return "others"


def string_equivalent(str1: str, str2: str) -> bool:
    """
    Compare two strings for "equality".
    Complete equality or differences only in spaces and underscores are considered equal.

    :param str1: The first string
    :param str2: The second string
    :return: Return True if the strings are considered equal, otherwise return False
    """
    if str1 == str2:
        return True

    normalized_str1 = "".join(char.lower() for char in str1 if char not in " _`")
    normalized_str2 = "".join(char.lower() for char in str2 if char not in " _`")
    return normalized_str1 == normalized_str2


def _normalize_description_string(description: str) -> str:
    """
    Normalize the description string.
    """
    description = description.strip().replace("\r", "").replace("\n", " ").replace("commonsense evidence:", "").replace("Commonsense evidence:", "").strip()
    while "  " in description:
        description = description.replace("  ", " ")
    return description


class Schema2IR:
    """
    Input information of a database schema, convert it to a dict representation of the schema.
    Thie intermediate representation (IR) can then be used for schema linking.

    IR format:
    - db_id
    - db_dir
    - db_json
    - bench
    - tables
        - table_id
        - table_name
    """

    ir: dict[str, Any]

    def __init__(self, db_id: str, db_dir: Path, db_json: dict[str, Any], bench: str):
        self.db_id = db_id
        self.db_dir = db_dir
        self.db_json = db_json
        self.bench = bench

        self.ir = {"db_id": db_id, "bench": bench}
        self._original_to_display_table_idx = self._build_original_to_display_table_idx()
        self._original_col_idx_to_display_name = self._build_original_col_idx_to_display_name()
        self._parse_tables()

    def _build_original_to_display_table_idx(self) -> dict[int, int]:
        """
        Build a best-effort mapping from table_names_original index -> table_names index.
        """
        display_table_names = self.db_json["table_names"]
        original_table_names = self.db_json["table_names_original"]

        mapping: dict[int, int] = {}
        used_display_indices: set[int] = set()

        for original_idx, original_name in enumerate(original_table_names):
            candidates = [
                display_idx
                for display_idx, display_name in enumerate(display_table_names)
                if display_idx not in used_display_indices and string_equivalent(display_name, original_name)
            ]
            if len(candidates) == 1:
                mapping[original_idx] = candidates[0]
                used_display_indices.add(candidates[0])

        for original_idx in range(len(original_table_names)):
            if original_idx not in mapping and original_idx < len(display_table_names):
                mapping[original_idx] = original_idx

        return mapping

    def _build_original_col_idx_to_display_name(self) -> dict[int, str]:
        """
        Build a best-effort mapping from column_names_original index -> column_names text.
        Structural parsing always uses *_original fields; this map is only for comments.
        """
        display_columns = self.db_json["column_names"]
        original_columns = self.db_json["column_names_original"]

        if len(display_columns) != len(original_columns):
            return {}

        if all(display_columns[idx][0] == original_columns[idx][0] for idx in range(len(display_columns))):
            return {idx: display_columns[idx][1] for idx in range(len(display_columns))}

        mapping: dict[int, str] = {}
        total_original_tables = len(self.db_json["table_names_original"])

        for original_table_idx in range(total_original_tables):
            display_table_idx = self._original_to_display_table_idx.get(original_table_idx)
            if display_table_idx is None:
                continue

            original_col_indices = [idx for idx, (table_idx, _) in enumerate(original_columns) if table_idx == original_table_idx]
            display_col_names = [col_name for table_idx, col_name in display_columns if table_idx == display_table_idx]

            if len(original_col_indices) != len(display_col_names):
                continue

            for original_col_idx, display_col_name in zip(original_col_indices, display_col_names):
                mapping[original_col_idx] = display_col_name

        return mapping

    def _get_display_table_name(self, original_table_idx: int, original_table_name: str) -> str:
        display_table_idx = self._original_to_display_table_idx.get(original_table_idx)
        if display_table_idx is None:
            return original_table_name

        display_table_names = self.db_json["table_names"]
        if 0 <= display_table_idx < len(display_table_names):
            return display_table_names[display_table_idx]

        return original_table_name

    def _parse_tables(self):
        self.ir["tables"] = []
        for original_table_idx, original_table_name in enumerate(self.db_json["table_names_original"]):
            table_name = self._get_display_table_name(original_table_idx, original_table_name)

            table = {
                "table_name": original_table_name,
                "table_comment": f" -- Table Description: {table_name}" if not string_equivalent(table_name, original_table_name) else "",
            }

            columns, primary_keys = self._parse_columns(original_table_idx, original_table_name)
            table["columns"] = columns
            table["primary_keys"] = primary_keys
            table["foreign_keys"] = self._parse_foreign_keys(original_table_idx, original_table_name)
            table["value_examples"] = self._parse_value_examples(original_table_name)
            self.ir["tables"].append(table)

    def _parse_columns(self, table_idx: int, table_name: str) -> list[dict[str, Any]]:
        # connect to db
        db_path = self.db_dir / f"{self.db_id}.sqlite"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        column_types = {}
        column_null_able = {}
        column_is_primary_key = {}
        column_contain_null = {}
        cursor.execute(f'PRAGMA table_info("{table_name}")')

        # get (type | nullable | primary key)
        for col in cursor.fetchall():
            column_name = col[1]
            column_type = col[2]
            column_types[column_name] = column_type
            column_null_able[column_name] = col[3]
            if col[5] >= 1:
                column_is_primary_key[column_name] = True

            # Check if the column contains NULL
            sql = f'SELECT COUNT(*) FROM "{table_name}" WHERE "{column_name}" IS NULL'
            cursor.execute(sql)
            column_contain_null[column_name] = cursor.fetchone()[0] > 0

        # get UNIQUE columns
        column_unique = {}
        cursor.execute(f'PRAGMA index_list("{table_name}")')
        for index in cursor.fetchall():
            if index[2] == 1:
                for column in cursor.execute(f'PRAGMA index_info("{index[1]}")').fetchall():
                    column_name = column[2]
                    column_unique[column_name] = True
        conn.close()

        # Prepare description for BIRD
        if self.bench == "BIRD_train" or self.bench == "BIRD_dev":
            desc_path = self.db_dir / "database_description" / f"{table_name}.csv"
            if not desc_path.exists():
                raise ValueError(f"Description file {desc_path} does not exist.")
            with open(desc_path, "rb") as f:
                result = chardet.detect(f.read())
            desc_df = pd.read_csv(desc_path, encoding=result["encoding"])

        # make column list and primary key list
        columns = []
        primary_keys = []

        for col_idx, (col_table_idx, original_col_name) in enumerate(self.db_json["column_names_original"]):
            if col_table_idx == table_idx:
                col_name = self._original_col_idx_to_display_name.get(col_idx, original_col_name)

                # Type
                try:
                    # col_type = column_types[original_col_name].upper()
                    col_type = normalize_type(column_types[original_col_name]).upper()
                except KeyError:
                    print(f"column {original_col_name} not found in table {table_name}.")
                    raise
                column_def = f'    "{original_col_name}" {col_type}'
                column_def_plain = column_def

                # Column comment and description
                descriptions = []
                if "Spider" in self.bench or self.bench.startswith("DB_"):
                    if not string_equivalent(col_name, original_col_name):
                        column_def_plain += f" -- Column Meaning: {col_name}"
                        descriptions.append("Column Meaning: " + col_name)

                    descriptions.append("Contains NULL: " + str(column_contain_null[original_col_name]))
                    column_def += " -- " + " | ".join(descriptions)

                else:
                    assert self.bench in ["BIRD_train", "BIRD_dev"]
                    if not string_equivalent(col_name, original_col_name):
                        descriptions.append("Column Meaning: " + col_name)

                    desc_line = desc_df[desc_df["original_column_name"].str.strip() == original_col_name]
                    assert not desc_line.empty, f"Description for column {original_col_name} not found in {desc_path}"

                    if not desc_line.empty:
                        # If the corresponding column in "nan" in pandas, it means no information about it
                        col_desc_val = desc_line["column_description"].values[0]
                        if pd.notna(col_desc_val):
                            col_desc = _normalize_description_string(col_desc_val)
                            if col_desc and not string_equivalent(original_col_name, col_desc) and not (col_name and string_equivalent(col_name, col_desc)):
                                descriptions.append("Column Description: " + col_desc)

                        value_desc_val = desc_line["value_description"].values[0]
                        if pd.notna(value_desc_val):
                            value_desc = _normalize_description_string(value_desc_val)
                            if (
                                value_desc
                                and not string_equivalent(original_col_name, value_desc)
                                and not (col_name and string_equivalent(col_name, value_desc))
                                and not (pd.notna(col_desc_val) and col_desc and string_equivalent(col_desc, value_desc))
                            ):
                                descriptions.append("Value Description: " + value_desc)

                    if descriptions:
                        column_def_plain += " -- " + " | ".join(descriptions)

                    descriptions.append("Contains NULL: " + str(column_contain_null[original_col_name]))
                    column_def += " -- " + " | ".join(descriptions)

                if original_col_name in column_is_primary_key:
                    primary_keys.append(len(columns))

                columns.append(
                    {
                        "col_idx": len(columns),
                        "col_name": original_col_name,
                        "col_defination": column_def,
                        "col_defination_plain": column_def_plain,
                    }
                )

        return columns, primary_keys

    def _parse_foreign_keys(self, table_idx: int, table_name: str) -> list[dict[str, Any]]:
        """Get foreign key information for a specific table."""
        foreign_keys = []
        db = self.db_json
        original_columns = db["column_names_original"]
        for fk in db["foreign_keys"]:
            from_col_idx, to_col_idx = fk
            if not (0 <= from_col_idx < len(original_columns) and 0 <= to_col_idx < len(original_columns)):
                continue
            if original_columns[from_col_idx][0] != table_idx:
                continue

            referenced_table_idx = original_columns[to_col_idx][0]
            if not (0 <= referenced_table_idx < len(db["table_names_original"])):
                continue

            foreign_keys.append(
                {
                    "table": f'"{table_name}"',
                    "column": f'"{original_columns[from_col_idx][1]}"',
                    "referenced_table": db["table_names_original"][referenced_table_idx],
                    "referenced_column": f'"{original_columns[to_col_idx][1]}"',
                }
            )
        return foreign_keys

    def _parse_value_examples(self, table_name: str) -> dict[str, Any]:
        db_path = self.db_dir / f"{self.db_id}.sqlite"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(f"SELECT name FROM pragma_table_info('{table_name}');")
        column_names = [row[0] for row in cursor.fetchall()]

        column_data: dict[str, Any] = {}
        for col_name in column_names:
            cursor.execute(f'SELECT DISTINCT "{col_name}" FROM "{table_name}" LIMIT 5')
            rows = cursor.fetchall()
            if not rows:
                continue

            column_values = [row[0] for row in rows]

            # Not too long
            column_values = [value for value in column_values if len(str(value)) < 80 and value]
            if not column_values:
                continue
            column_data[col_name] = column_values

        conn.close()
        return column_data

    def to_dict(self):
        return self.ir


def is_supported_benchmark(bench: str) -> bool:
    return bench in SUPPORTED_BENCHES or bench.startswith("DB_")


def resolve_benchmark_paths(bench: str, dataset_root: Path, drspider_root: Path) -> BenchmarkPaths:
    spider_base = dataset_root / "spider_data"
    if bench in {"Spider_train", "Spider_dev"}:
        return BenchmarkPaths(
            db_base=spider_base / "database",
            table_path=spider_base / "tables.json",
        )
    if bench == "Spider_test":
        return BenchmarkPaths(
            db_base=spider_base / "test_database",
            table_path=spider_base / "test_tables.json",
        )
    if bench == "BIRD_train":
        return BenchmarkPaths(
            db_base=dataset_root / "BIRD_train" / "train_databases",
            table_path=dataset_root / "BIRD_train" / "train_tables.json",
        )
    if bench == "BIRD_dev":
        return BenchmarkPaths(
            db_base=dataset_root / "BIRD_dev" / "dev_databases",
            table_path=dataset_root / "BIRD_dev" / "dev_tables.json",
        )
    if bench.startswith("DB_"):
        return BenchmarkPaths(
            db_base=drspider_root / bench / "database_post_perturbation",
            table_path=drspider_root / bench / "tables_post_perturbation.json",
        )
    raise ValueError(f"Unsupported benchmark: {bench}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate schema IR from benchmark metadata and SQLite files.")
    parser.add_argument(
        "--bench",
        action="append",
        default=[],
        help="Benchmark name. Repeat this flag for multiple benchmarks. Supports Spider_train/Spider_dev/Spider_test/BIRD_train/BIRD_dev and DB_*.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all fixed benchmarks: Spider_train, Spider_dev, Spider_test, BIRD_train, BIRD_dev.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Dataset root directory.")
    parser.add_argument("--drspider-root", type=Path, default=DEFAULT_DRSPIDER_ROOT, help="Dr.Spider dataset root directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for IR files.")
    parser.add_argument(
        "--legacy-ir-name",
        action="store_true",
        help="Use legacy output name format: <bench>_ir.json. Default is <bench>.json.",
    )
    parser.add_argument(
        "--db-id",
        action="append",
        default=[],
        help="Only process specified db_id. Repeat this flag for multiple db_id values.",
    )
    return parser.parse_args()


def resolve_benchmarks(args: argparse.Namespace) -> list[str]:
    benches: list[str] = []
    if args.all:
        benches.extend(SUPPORTED_BENCHES)
    benches.extend(args.bench)

    if not benches:
        raise ValueError("No benchmark specified. Use --bench <name> or --all.")

    unique_benches: list[str] = []
    seen: set[str] = set()
    for bench in benches:
        if not is_supported_benchmark(bench):
            raise ValueError(f"Unsupported benchmark: {bench}. Supported fixed benchmarks: Spider_train, Spider_dev, Spider_test, BIRD_train, BIRD_dev; plus DB_* for Dr.Spider.")
        if bench not in seen:
            unique_benches.append(bench)
            seen.add(bench)

    return unique_benches


def build_output_path(bench: str, output_dir: Path, legacy_ir_name: bool) -> Path:
    filename = f"{bench}_ir.json" if legacy_ir_name else f"{bench}.json"
    return output_dir / filename


def generate_ir_set(bench: str, paths: BenchmarkPaths, selected_db_ids: set[str] | None = None) -> list[dict[str, Any]]:
    if not paths.db_base.exists():
        raise FileNotFoundError(f"Database base directory not found: {paths.db_base}")
    if not paths.table_path.exists():
        raise FileNotFoundError(f"Table metadata file not found: {paths.table_path}")

    with paths.table_path.open("r", encoding="utf-8") as file:
        tables: list[dict[str, Any]] = json.load(file)

    tables_by_db_id: dict[str, dict[str, Any]] = {table["db_id"]: table for table in tables}
    db_ids = sorted(tables_by_db_id)

    if selected_db_ids:
        db_ids = [db_id for db_id in db_ids if db_id in selected_db_ids]

    ir_set: list[dict[str, Any]] = []
    for db_id in tqdm(db_ids, desc=f"Processing {bench}", total=len(db_ids)):
        db_dir = paths.db_base / db_id
        db_json = tables_by_db_id[db_id]
        print(f"### {bench}: {db_id}")
        ir_set.append(Schema2IR(db_id, db_dir, db_json, bench).to_dict())

    return ir_set


def main() -> None:
    args = parse_args()
    benches = resolve_benchmarks(args)
    selected_db_ids = set(args.db_id) if args.db_id else None

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for bench in benches:
        bench_paths = resolve_benchmark_paths(bench, dataset_root=args.dataset_root, drspider_root=args.drspider_root)
        ir_set = generate_ir_set(bench, bench_paths, selected_db_ids=selected_db_ids)

        output_path = build_output_path(bench, output_dir=args.output_dir, legacy_ir_name=args.legacy_ir_name)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(ir_set, file, ensure_ascii=False, indent=2)

        print(f"[{bench}] Total DBs processed: {len(ir_set)}")
        print(f"[{bench}] IR saved to: {output_path}")


if __name__ == "__main__":
    main()
