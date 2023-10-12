import logging

from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from tinyllama_breakdown.dataset import prepare_train_data, prepare_train_data_dummy

LOG = logging.getLogger(__name__)


def get_model_and_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.main.base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(**cfg.bnb)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.main.base_model_id, quantization_config=bnb_config, **cfg.hf_model
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer


def finetune_tinyllama(cfg):
    if cfg.main.dummy:
        data = prepare_train_data_dummy(cfg)
    else:
        data = prepare_train_data(cfg)
    model, tokenizer = get_model_and_tokenizer(cfg)

    peft_config = LoraConfig(**cfg.lora)
    training_arguments = TrainingArguments(
        output_dir=cfg.main.model_id_bactrian_lora, **cfg.training_arguments
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        train_dataset=data["train"],
        peft_config=peft_config,
        packing=False,
        max_seq_length=1024,
        dataset_text_field="text",
    )
    LOG.info("Start training")
    trainer.train()
    LOG.info("Finished training")

    LOG.info("Pushing model to hub")
    trainer.push_to_hub()
