import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from datasets import Dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, ModelConfig, TrlParser

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


@dataclass
class CustomConfig:
    cache_dir: str = field(metadata={"help": "Cache directory for pretrained models."})
    sft_model_path: str = field(metadata={"help": "Path to the SFT-finetuned base model."})
    dpo_model_storage_dir: str = field(metadata={"help": "Directory to save the DPO-tuned model."})
    finetune_data_dir: str = field(metadata={"help": "Path to the finetuning data file (containing DPO pairs)."})


def create_dpo_dataset(dpo_raw_data: list[dict], tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    """
    Formats the raw data into a Dataset for DPOTrainer.
    Each entry must have 'prompt', 'chosen', and 'rejected' keys.
    """
    dpo_data_list = []

    for dp in dpo_raw_data:
        # Construct the user part of the prompt
        prompt = SCHEMA_LINK_TEMPLATE.format(
            schema=dp["dynamic_schema"],
            question=dp["question_with_evidence"],
        )

        chosen_response = dp["gt_schema_link_linearized"]
        rejected_response = dp["loss_schema_link_linearized"]

        # Apply chat template to create the full prompt that the model will see
        # DPOTrainer needs the prompt and responses separately
        prompt_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": SCHEMA_LINK_RESPONSE_TEMPLATE}]
        formatted_prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, continue_final_message=True)

        # Filter out long sequences
        if len(tokenizer.encode(formatted_prompt + chosen_response)) < max_length and len(tokenizer.encode(formatted_prompt + rejected_response)) < max_length:
            dpo_data_list.append(
                {
                    "prompt": formatted_prompt,
                    "chosen": chosen_response,
                    "rejected": rejected_response,
                }
            )

    logger.info(f"DPO Datasize From {len(dpo_raw_data)} --> {len(dpo_data_list)}")
    return Dataset.from_list(dpo_data_list)


if __name__ == "__main__":
    parser = TrlParser((CustomConfig, ModelConfig, DPOConfig))
    (custom_config, model_config, dpo_args) = parser.parse_args_and_config()

    logger.info(f"Using random seed for this run: {dpo_args.seed}")
    logger.info(f"Loading base SFT model from: {custom_config.sft_model_path}")

    # 1. Load the SFT-tuned model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        custom_config.sft_model_path,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        cache_dir=custom_config.cache_dir,
        local_files_only=True,
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        custom_config.sft_model_path,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        cache_dir=custom_config.cache_dir,
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        custom_config.sft_model_path,
        cache_dir=custom_config.cache_dir,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    # 2. Load and prepare the DPO dataset
    finetune_data_json = json.load(open(custom_config.finetune_data_dir, "r"))
    dpo_finetune_data = finetune_data_json

    dpo_dataset = create_dpo_dataset(
        dpo_raw_data=dpo_finetune_data,
        tokenizer=tokenizer,
        max_length=dpo_args.max_length,
    )
    dpo_dataset = dpo_dataset.shuffle(seed=dpo_args.seed)

    # 3. Initialize DPOTrainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
    )

    # 4. Start DPO training
    dpo_trainer.train()

    # 5. Save the final model
    base_model_name = custom_config.sft_model_path.split("/")[-1].split("_lr")[0]
    config_name = f"{base_model_name}_lr_{dpo_args.learning_rate}_beta_{dpo_args.beta}_alpha_{dpo_args.rpo_alpha}_epoch_{dpo_args.num_train_epochs}"
    storage_path = Path(custom_config.dpo_model_storage_dir) / config_name

    dpo_trainer.save_model(storage_path)
    tokenizer.save_pretrained(storage_path)
