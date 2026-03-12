# value_index/build_index.py

import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .vector_index import ColumnVectorIndex


def embed_values_in_db(
    bench: str,
    db_base_path: Path,
    db_id: str,
    embed_model: SentenceTransformer,
    output_root: Path = Path("artifacts/value_index"),
) -> Path:
    """Embed the values of the TEXT-like columns of the tables in the database.
    Store the embeddings in a FAISS index and save it in the end.
    Vector Index (FAISS) design:
        - each column --> a set of vectors (embeddings)
        - query embeddings --> retrieve some TEXT values in this column
        - which means every (table, TEXT-like column) has a vector index
        - we have a dict[table_name][column_name] = FAISS index
    """

    def custom_text_factory(byte_string: bytes):
        """
        Try to decode the byte string with UTF-8, if failed, return None.
        """
        try:
            return byte_string.decode("utf-8")
        except UnicodeDecodeError:
            # When encountering a decoding error, return None, instead of raising an exception
            return None

    print(f"Start embedding {bench}: {db_id}")
    TEXT_COLUMN_TYPES = ["TEXT", "VARCHAR", "CHAR", "DATE", "DATETIME"]
    all_indexes: dict[tuple[str, str], ColumnVectorIndex] = {}

    # Connect to the database
    conn = sqlite3.connect(db_base_path / db_id / f"{db_id}.sqlite")
    conn.text_factory = custom_text_factory
    cursor = conn.cursor()

    # Get all the tables in the database, we will embed the values of each table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_names = [table[0] for table in tables if table[0] != "sqlite_sequence"]

    for table_name in tqdm(table_names, total=len(table_names), desc=f"Processing tables in {db_id}"):
        # Get the columns of the table
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns = cursor.fetchall()

        for column in tqdm(columns, total=len(columns), desc=f"Embedding values in {table_name}"):
            col_name = column[1]
            col_type = column[2]
            if any(TEXT_COLUMN_TYPE in col_type.upper() for TEXT_COLUMN_TYPE in TEXT_COLUMN_TYPES):
                print(f"Processing column: {column}")
                cursor.execute(f'SELECT DISTINCT "{col_name}" FROM "{table_name}" LIMIT 30000')
                values = []
                while True:
                    try:
                        row = cursor.fetchone()
                        if row is None:
                            break
                        value = row[0]
                        # skip overly long strings to keep embeddings manageable
                        if isinstance(value, str) and (value is not None) and str(value).strip() and len(value) < 200:
                            values.append(value)
                    except (UnicodeDecodeError, sqlite3.OperationalError) as e:
                        print(f"{bench}: {db_id}: Warning: Could not fetch values in {table_name}.{col_name}: {e}")
                        continue

                if len(values) == 0:
                    continue

                # Embed the values with FAISS
                try:
                    embeddings: np.ndarray = embed_model.encode(values)
                except Exception as e:
                    print(f"{bench}: {db_id}: Warning: Could not embed values in {table_name}.{col_name}: {e}")
                    print(f"Type = {type(values[0])}, TEXT_COLUMN_TYPE = {col_type.upper()}")
                    continue

                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
                all_indexes[(table_name, col_name)] = ColumnVectorIndex(index, values)

    conn.close()

    output_dir = output_root / bench
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{db_id}.pkl"

    with open(out_path, "wb") as f:
        pickle.dump(all_indexes, f)

    return out_path
