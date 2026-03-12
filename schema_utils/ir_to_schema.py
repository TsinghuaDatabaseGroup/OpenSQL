import copy
import datetime
import decimal
import re
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
else:
    SentenceTransformer = Any

from value_index import ColumnVectorIndex


class IR2Schema:
    """
    Render an IR (intermediate representation) of a database into a SQL DDL-like schema string,
    optionally augmented with column value examples (static examples + dynamic retrieval).

    This class is typically used to:
    1) Produce a compact, LLM-friendly schema prompt (tables/columns/PK/FK + examples).
    2) Produce a localized context for a specific table/column to support column selection
       or explainability.

    Parameters
    ----------
    ir:
        Database IR dictionary. Expected keys:
        - "db_id": str
        - "tables": list[dict], each table contains:
            - "table_name": str
            - "table_comment": str
            - "columns": list[dict] with "col_idx", "col_name",
              "col_defination", "col_defination_plain"
            - "primary_keys": list[int]
            - "foreign_keys": list[dict] with "column", "referenced_table", "referenced_column"
            - "value_examples": dict[str, list[Any]]
        Optional keys:
        - "db_overview": str
    chosen:
        Optional selection mapping {table_name: [column_name, ...]}.
        If None, all tables/columns are considered selected.
    tindex:
        Optional mapping {(table_name, column_name): ColumnVectorIndex} for retrieval-based examples.
    question:
        Optional user question text. Required for retrieval-based examples.
    emb_model:
        Optional embedding model used by ColumnVectorIndex.get_similar_strings.
    print_contain_null:
        Whether to use the full column definition (may include NULL constraints) or a plain variant.

    Notes
    -----
    - The rendered schema is intended for prompting / inspection, not for direct execution.
    - Value examples may contain sensitive strings; consider filtering/limiting in upstream code.
    """

    def __init__(
        self,
        ir: dict,
        chosen: Optional[dict[str, list[str]]],
        tindex: Optional[dict[tuple[str, str], ColumnVectorIndex]],
        question: Optional[str],
        emb_model: Optional[SentenceTransformer],
        print_contain_null: bool,
    ):
        self.ir = ir
        self.chosen = chosen
        self.tindex = tindex
        self.question = question
        self.emb_model = emb_model
        self.print_contain_null = print_contain_null

    def _optimize_value_examples(self, examples: list[Any]) -> list[Any]:
        if not examples:
            return []

        def is_email(text: str) -> bool:
            pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
            return re.match(pattern, text) is not None

        def is_url(text: str) -> bool:
            return text.startswith("http://") or text.startswith("https://")

        str_examples = [str(e) for e in examples if e is not None]
        if not str_examples:
            return []

        first_example_str = str_examples[0]

        if is_email(first_example_str) or is_url(first_example_str):
            return [examples[0]]

        return examples

    def _is_table_chosen(self, table_name: str) -> bool:
        if not self.chosen:
            return True
        chosen_table_names = [name.lower() for name in self.chosen]
        return table_name.lower() in chosen_table_names

    def _is_column_chosen(self, random_table_name: str, column_name: str) -> bool:
        if not self.chosen:
            return True
        if not self._is_table_chosen(random_table_name):
            return False

        column_name_lower = column_name.lower()

        for table_name in self.chosen:
            if table_name.lower() == random_table_name.lower():
                chosen_column_names = [name.lower() for name in self.chosen[table_name]]
                if column_name_lower in chosen_column_names:
                    return True
                break

        return False

    def _get_column_value_examples(self, table: dict[str, Any], column_name: str) -> str:
        default_value_examples = []
        if column_name in table["value_examples"]:
            default_value_examples = table["value_examples"][column_name]

        dynamic_value_examples = []
        key = (table["table_name"], column_name)
        if (self.tindex is not None) and (self.emb_model is not None) and (self.question is not None) and (key in self.tindex):
            column_vector_index = self.tindex[key]
            retrieved_values = column_vector_index.get_similar_strings(self.emb_model, self.question)
            dynamic_value_examples = [val for val in retrieved_values if len(val) < 100]

        all_value_examples = list(dict.fromkeys(dynamic_value_examples + default_value_examples))
        if len(all_value_examples) == 0:
            return ""
        else:
            first_val = all_value_examples[0]
            if isinstance(first_val, str) or isinstance(first_val, datetime.date) or isinstance(first_val, datetime.datetime):
                formatted_values = ["'" + str(val).strip() + "'" for val in all_value_examples]
            elif isinstance(first_val, decimal.Decimal):
                formatted_values = [str(float(v)) for v in all_value_examples]
            else:
                formatted_values = [str(val) for val in all_value_examples]

        template = "[Column Value Examples]\nWe want to know if the column named '{column_name}' is useful to answer the question, here are some example values of this column:\n{value_examples}"
        return template.format(column_name=column_name, value_examples=", ".join(formatted_values))

    def _table_statement(self, table: dict[str, Any], pred_link, add_value: Optional[bool] = True) -> str:
        have_primary_keys = False
        have_foreign_keys = False
        statement_latter = ""

        primary_keys = table["primary_keys"]
        if primary_keys:
            primary_names = ['"' + table["columns"][idx]["col_name"] + '"' for idx in primary_keys]
            primary_statement = "    PRIMARY KEY (" + ", ".join(primary_names) + ")"
            statement_latter += primary_statement
            have_primary_keys = True

        extra_foreign_keys = set()
        for i, fk in enumerate(table["foreign_keys"]):
            from_column = fk["column"]
            to_table = fk["referenced_table"]
            to_column = fk["referenced_column"]

            if (not self._is_table_chosen(to_table)) and (not self._is_column_chosen(table["table_name"], from_column)):
                continue
            extra_foreign_keys.add(from_column.strip('"'))

            if (have_primary_keys) or (have_foreign_keys):
                statement_latter += ",\n"
            statement_latter += f"    FOREIGN KEY ({from_column}) REFERENCES {to_table}({to_column})"
            have_foreign_keys = True

        if statement_latter:
            statement_latter += "\n);\n"
        else:
            statement_latter = ");\n"

        former_statement = f'CREATE TABLE "{table["table_name"]}" ({table["table_comment"]}\n'
        chosen_columns = []

        for column in table["columns"]:
            if self.chosen:
                valid = column["col_idx"] in primary_keys
                valid = valid or self._is_column_chosen(table["table_name"], column["col_name"])
                valid = valid or (column["col_name"] in extra_foreign_keys)

                if not valid:
                    continue
                elif pred_link is not None:
                    try:
                        if column["col_name"] not in pred_link[table["table_name"]]:
                            pred_link[table["table_name"]].append(column["col_name"])
                    except Exception as e:
                        print(e)
                        print("table['table_name']: ", table["table_name"])
                        print("column['col_name']: ", column["col_name"])
                        print("pred_link: ", pred_link)
                        raise e
            chosen_columns.append(column)

        column_parts_list = []
        for column in chosen_columns:
            if self.print_contain_null:
                col_def_full = column["col_defination"].replace("\n", " ").strip()
            else:
                col_def_full = column["col_defination_plain"].replace("\n", " ").strip()
            code_part = col_def_full
            existing_comment = ""
            if " --" in col_def_full:
                parts = col_def_full.split(" --", 1)
                code_part = parts[0].strip()
                existing_comment = parts[1].strip()

            value_example = []
            col_name = column["col_name"]
            key = (table["table_name"], col_name)

            if add_value:
                if self.tindex is not None and self.emb_model is not None and self.question is not None and key in self.tindex:
                    column_vector_index = self.tindex[key]
                    retrieved_values = column_vector_index.get_similar_strings(self.emb_model, self.question)
                    value_example.extend([val for val in retrieved_values if len(val) < 100])

                if col_name in table["value_examples"]:
                    default_values = table["value_examples"][col_name]
                    for val in default_values:
                        if val not in value_example and len(value_example) < 4:
                            value_example.append(val)

                value_example = self._optimize_value_examples(value_example)

            final_comment = ""
            if value_example:
                formatted_values = []
                first_val = value_example[0]
                if isinstance(first_val, str) or isinstance(first_val, datetime.date) or isinstance(first_val, datetime.datetime):
                    formatted_values = ["'" + str(val).strip() + "'" for val in value_example]
                elif isinstance(first_val, decimal.Decimal):
                    formatted_values = [str(float(v)) for v in value_example]
                else:
                    formatted_values = [str(val) for val in value_example]

                value_str = f"Value Examples: [{', '.join(formatted_values)}]"

                if existing_comment:
                    final_comment = f"-- {existing_comment} | {value_str}"
                else:
                    final_comment = f"-- {value_str}"

            elif existing_comment:
                final_comment = f"-- {existing_comment}"

            column_parts_list.append((code_part, final_comment))

        column_str = ""
        num_columns = len(column_parts_list)
        for i, (code_part, final_comment) in enumerate(column_parts_list):
            line_builder = f"    {code_part}"
            is_last_column = i == num_columns - 1
            if not is_last_column or (is_last_column and (have_primary_keys or have_foreign_keys)):
                line_builder += ","
            if final_comment:
                line_builder += f" {final_comment}"
            column_str += line_builder + "\n"

        former_statement += column_str
        return former_statement + statement_latter

    def render_schema(self) -> tuple[str, Optional[dict[str, list[str]]]]:
        if self.chosen is not None:
            pred_link = copy.deepcopy(self.chosen)
        else:
            pred_link = None

        schema = f"-- Database name: {self.ir['db_id']}\n"

        if "db_overview" in self.ir:
            schema += f"-- Database overview: {self.ir['db_overview']}\n"

        schema += "-- Database schema:\n"
        for table in self.ir["tables"]:
            table_name = table["table_name"]
            if not self._is_table_chosen(table_name):
                continue
            schema += self._table_statement(table, pred_link)
        return schema, pred_link

    def render_table_and_column_examples(self, table_name: str, local_column: str) -> tuple[str, str]:
        table = [t for t in self.ir["tables"] if t["table_name"] == table_name][0]
        pred_link = None
        table_statement = self._table_statement(table, pred_link, add_value=False)
        column_value_examples = self._get_column_value_examples(table, local_column)
        return table_statement, column_value_examples
