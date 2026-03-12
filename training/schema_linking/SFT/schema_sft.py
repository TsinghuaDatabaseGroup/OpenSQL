import json
from dataclasses import dataclass, field
from pathlib import Path

from datasets import Dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, ModelConfig, SFTConfig, SFTTrainer, TrlParser

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
    cache_dir: str = field()
    model_storage_dir: str = field()
    finetune_data_dir: str = field()


def formatting_prompts_func(training_dataset, tokenizer: AutoTokenizer):
    texts = []
    dataset_size = len(training_dataset["dynamic_schema"])

    for i in range(dataset_size):
        input_prompt: str = SCHEMA_LINK_TEMPLATE.format(
            schema=training_dataset["dynamic_schema"][i],
            question=training_dataset["question_with_evidence"][i],
        )

        output = SCHEMA_LINK_RESPONSE_TEMPLATE + training_dataset["gt_schema_link_linearized"][i]
        messages = [{"role": "user", "content": input_prompt}, {"role": "assistant", "content": output}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)
    return texts


if __name__ == "__main__":
    parser = TrlParser((CustomConfig, ModelConfig, SFTConfig))
    (custom_config, model_config, training_args) = parser.parse_args_and_config()
    custom_config: CustomConfig
    model_config: ModelConfig
    training_args: SFTConfig
    training_args_dict = training_args.to_dict()

    ####################################
    # 1. Model Init kwargs & Tokenizer #
    ####################################
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        cache_dir=custom_config.cache_dir,
        local_files_only=True,
    )

    #################################
    # 2. Load & Tweak the tokenizer #
    #################################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        cache_dir=custom_config.cache_dir,
        local_files_only=True,
    )
    tokenizer.padding_side = "right"

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

    ##############
    # 3. Dataset #
    ##############
    finetune_data_json = json.load(open(custom_config.finetune_data_dir, "r"))
    sft_finetune_data = [d for d in finetune_data_json if d["train_type"] == "SFT"]

    # Regularize SFT data
    logger.info(f"SFT Datasize Before Cleaning: {len(sft_finetune_data)}")
    reg_sft = []
    for dp in sft_finetune_data:
        full_text = SCHEMA_LINK_TEMPLATE.format(schema=dp["dynamic_schema"], question=dp["question_with_evidence"]) + dp["gt_schema_link_linearized"]
        if len(tokenizer.encode(full_text)) <= training_args.max_seq_length:
            reg_sft.append(dp)
    logger.info(f"SFT Datasize From {len(sft_finetune_data)} --> {len(reg_sft)}")

    data_dir = Path(custom_config.finetune_data_dir).parent
    with open(data_dir / "reg_sft_8192.json", "w") as f:
        json.dump(reg_sft, f, indent=2)

    sft_dataset = Dataset.from_list(reg_sft).shuffle(seed=training_args.seed)

    ###############
    # 4. Training #
    ###############
    collator = DataCollatorForCompletionOnlyLM(SCHEMA_LINK_RESPONSE_TEMPLATE, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        data_collator=collator,
        formatting_func=lambda dataset: formatting_prompts_func(dataset, tokenizer),
        train_dataset=sft_dataset,
    )
    trainer.train()

    ####################################
    # 5. Validation for the first time #
    ####################################

    # format model name and storage name
    model_name = model_config.model_name_or_path.split("/")[-1]
    config_name = f"lr_{training_args.learning_rate}_epoch_{training_args.num_train_epochs}"
    storage_path = Path(custom_config.model_storage_dir) / f"{model_name}_{config_name}"

    logger.info(f"Storage Path: {storage_path}")

    if not Path(custom_config.model_storage_dir).exists():
        Path(custom_config.model_storage_dir).mkdir(parents=True, exist_ok=True)

    trainer.save_model(storage_path)
    tokenizer.save_pretrained(storage_path)
