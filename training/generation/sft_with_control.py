import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import (
    DataCollatorForCompletionOnlyLM,
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
)

# A single, unified prompt template for the SQL generation task.
# The {question} will be dynamically formatted to include a control token.
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
REASONING_PATHS = ["Normal", "CTE", "Subquery"]
CONTROL_TOKENS = [f"[{path.upper()}]" for path in REASONING_PATHS]
WRONG_SQL_MARKERS = ["<|wrong SQL|>", "<|unsuited|>"]
NA_TOKEN = "[N/A]"


@dataclass
class CustomConfig:
    cache_dir: str = field(metadata={"help": "Directory to cache pretrained models."})
    model_storage_dir: str = field(metadata={"help": "Directory to save the fine-tuned models."})
    finetune_data_dir: str = field(metadata={"help": "Path to the fine-tuning data in JSON format."})


def create_training_instances(raw_data: List[Dict]) -> List[Dict]:
    training_instances = []
    for dp in raw_data:
        schema = dp.get("dynamic_noised_schema")
        question = dp.get("question_with_evidence")
        if not schema or not question:
            continue

        for path in REASONING_PATHS:
            sql_or_marker = dp.get(path)

            if not sql_or_marker:
                continue

            target_output = NA_TOKEN if sql_or_marker in WRONG_SQL_MARKERS else sql_or_marker

            training_instances.append(
                {
                    "dynamic_noised_schema": schema,
                    "question_with_evidence": question,
                    "reasoning_path": path,
                    "query": target_output,
                }
            )
    return training_instances


def formatting_prompts_func(training_dataset):
    texts = []
    dataset_size = len(training_dataset["dynamic_noised_schema"])
    for i in range(dataset_size):
        path = training_dataset["reasoning_path"][i]
        question = training_dataset["question_with_evidence"][i]
        controlled_question = f"[{path.upper()}] {question}"

        # Format the full input prompt
        input_prompt: str = NL2SQL_TEMPLATE.format(
            schema=training_dataset["dynamic_noised_schema"][i],
            question=controlled_question,
        )

        # The expected model output (either a SQL query or "N/A")
        output = SQL_RESPONSE_TEMPLATE + training_dataset["query"][i]

        # Apply the chat template to create the final training text
        messages = [{"role": "user", "content": input_prompt}, {"role": "assistant", "content": output}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)
    return texts


if __name__ == "__main__":
    # --- 1. Argument Parsing ---
    parser = TrlParser((CustomConfig, ModelConfig, SFTConfig))
    (custom_config, model_config, training_args) = parser.parse_args_and_config()
    logger.info(f"Using random seed for this run: {training_args.seed}")

    # --- 2. Model and Tokenizer Initialization ---
    logger.info(f"Loading base model from: {model_config.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        cache_dir=custom_config.cache_dir,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        cache_dir=custom_config.cache_dir,
        local_files_only=True,
    )
    tokenizer.padding_side = "right"

    # Add special tokens for control and padding
    special_tokens_to_add = []
    if tokenizer.pad_token is None:
        special_tokens_to_add.append("[PAD]")

    if NA_TOKEN not in tokenizer.get_vocab():
        special_tokens_to_add.append(NA_TOKEN)

    for token in CONTROL_TOKENS:
        if token not in tokenizer.get_vocab():
            special_tokens_to_add.append(token)

    if special_tokens_to_add:
        tokenizer.add_special_tokens({"pad_token": "[PAD]", "additional_special_tokens": special_tokens_to_add})
        model.resize_token_embeddings(len(tokenizer))

    logger.info(f"Tokenizer vocabulary updated. Control tokens: {CONTROL_TOKENS}")
    logger.info("Model and tokenizer loaded successfully.")

    # --- 3. Dataset Loading and Preparation ---
    with open(custom_config.finetune_data_dir, "r") as f:
        finetune_data_json = json.load(f)

    training_instances = create_training_instances(finetune_data_json)

    final_instances = []
    for dp in training_instances:
        controlled_question = f"[{dp['reasoning_path'].upper()}] {dp['question_with_evidence']}"
        input_prompt = NL2SQL_TEMPLATE.format(schema=dp["dynamic_noised_schema"], question=controlled_question)
        full_output = SQL_RESPONSE_TEMPLATE + dp["query"]
        if len(tokenizer.encode(input_prompt + full_output)) <= training_args.max_seq_length:
            final_instances.append(dp)

    logger.info(f"Dataset size after filtering long sequences: {len(final_instances)}")
    sft_dataset = Dataset.from_list(final_instances).shuffle(seed=training_args.seed)

    # --- 4. Trainer Setup and Training ---
    collator = DataCollatorForCompletionOnlyLM(response_template=SQL_RESPONSE_TEMPLATE, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=sft_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    trainer.train()

    # --- 5. Save the Final Model ---
    model_name_part = model_config.model_name_or_path.split("/")[-1]
    config_name = f"lr_{training_args.learning_rate}_epoch_{training_args.num_train_epochs}"
    storage_path = Path(custom_config.model_storage_dir) / f"{model_name_part}_{config_name}"

    storage_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(storage_path))
    tokenizer.save_pretrained(str(storage_path))
