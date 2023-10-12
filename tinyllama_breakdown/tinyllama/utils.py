import torch
from peft import PeftModel, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments


def merge_lora_model(
    pretrained_model_name_or_path: str, adapter_name_or_path: str, path_to_save: str
):
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model_to_merge = PeftModel.from_pretrained(base_model, adapter_name_or_path)

    model_to_merge = model_to_merge.merge_and_unload()
    model_to_merge.save_pretrained(path_to_save)
