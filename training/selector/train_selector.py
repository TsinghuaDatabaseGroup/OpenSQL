import json
from dataclasses import dataclass, field
from pathlib import Path

from datasets import Dataset
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, ModelConfig, SFTConfig, SFTTrainer, TrlParser

RERANK_RESPONSE_TEMPLATE = "[Correct Group Judgement]\n"


@dataclass
class CustomConfig:
    cache_dir: str = field()
    model_storage_dir: str = field()
    finetune_data_dir: str = field()


def formatting_prompts_func(training_dataset, tokenizer: AutoTokenizer):
    texts = []
    for i in range(len(training_dataset["input_prompt"])):
        input_prompt: str = training_dataset["input_prompt"][i]
        label: str = RERANK_RESPONSE_TEMPLATE + training_dataset["output_label"][i]
        messages = [{"role": "user", "content": input_prompt}, {"role": "assistant", "content": label}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)
    return texts


def normalize_selector_sample(item: dict) -> dict | None:
    """Normalize different selector-data schemas into one training sample."""
    # Preferred schema (from updated compare_augment.py)
    if item.get("instruction_content") and item.get("output_content"):
        return {
            "input_prompt": item["instruction_content"],
            "output_label": item["output_content"],
        }

    # Legacy schema: instruction + analysis + final_choice
    if item.get("instruction") and item.get("analysis") and item.get("final_choice"):
        final_choice = str(item["final_choice"]).strip().upper()
        if final_choice not in {"SQL1", "SQL2"}:
            return None
        output_content = f"{item['analysis'].rstrip()}\n\\box{{{final_choice}}}"
        return {
            "input_prompt": item["instruction"],
            "output_label": output_content,
        }

    return None


if __name__ == "__main__":
    parser = TrlParser((CustomConfig, ModelConfig, SFTConfig))
    (custom_config, model_config, training_args) = parser.parse_args_and_config()
    custom_config: CustomConfig
    model_config: ModelConfig
    training_args: SFTConfig

    # --- 1. Load model and tokenizer ---
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        cache_dir=custom_config.cache_dir,
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        cache_dir=custom_config.cache_dir,
        local_files_only=True,
    )

    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    # --- 2. Load and prepare dataset ---
    finetune_data_json = json.load(open(custom_config.finetune_data_dir, "r"))
    normalized_samples = []
    for item in finetune_data_json:
        normalized = normalize_selector_sample(item)
        if normalized is not None:
            normalized_samples.append(normalized)
        else:
            logger.warning("Skip one selector sample due to missing/invalid fields.")

    if len(normalized_samples) == 0:
        raise ValueError("No valid selector training samples found after normalization.")

    finetune_data_json = normalized_samples
    finetune_dataset = Dataset.from_list(finetune_data_json).shuffle(seed=training_args.seed)

    # --- 3. Initialize trainer ---
    collator = DataCollatorForCompletionOnlyLM(RERANK_RESPONSE_TEMPLATE, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        data_collator=collator,
        formatting_func=lambda dataset: formatting_prompts_func(dataset, tokenizer),
        train_dataset=finetune_dataset,
    )

    # --- 4. Start training ---
    trainer.train()

    # --- 5. Save final model ---
    model_name_prefix = model_config.model_name_or_path.split("/")[-1]
    config_name = f"lr_{training_args.learning_rate}_epoch_{training_args.num_train_epochs}"
    storage_path = Path(custom_config.model_storage_dir) / f"{model_name_prefix}_{config_name}"

    if not Path(custom_config.model_storage_dir).exists():
        Path(custom_config.model_storage_dir).mkdir(parents=True, exist_ok=True)

    trainer.save_model(storage_path)
    tokenizer.save_pretrained(storage_path)
