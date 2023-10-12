import pandas as pd

from datasets import Dataset, DatasetDict, load_dataset
from tinyllama_breakdown.templates.prompt_format import (
    GENERATE_DATA_ANKI_CARDS_JSONL_TEMPLATE,
)


def formatted_train(user: str, assistant: str, instruction: str | None) -> str:
    if instruction:
        return f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{user}\
\n<|im_start|>assistant\n{assistant}"
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}\
<|im_end|>\n"


def prepare_train_data(cfg) -> DatasetDict:
    data = load_dataset(cfg.data.dataset_id, name=cfg.data.dataset_subset_id)
    prepared_data = DatasetDict()

    if isinstance(data, DatasetDict):
        for subset in data.keys():
            data_subset = data[subset]
            data_df = data_subset.to_pandas()
            # For now only works on Bactrian-X dataset
            data_df["text"] = data_df[["instruction", "input", "output"]].apply(
                lambda x: formatted_train(*x), axis=1
            )
            prepared_data[subset] = Dataset.from_pandas(data_df)

    return prepared_data


def preprocess_data_dummy(examples: dict, templates: str) -> dict:
    examples["text"] = templates.format(
        input_user=examples["user"], response=examples["assistant"]
    )
    return examples


def prepare_train_data_dummy(cfg) -> DatasetDict:
    df = pd.read_csv(cfg.data.csv_path)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(
        lambda x: preprocess_data_dummy(x, GENERATE_DATA_ANKI_CARDS_JSONL_TEMPLATE)
    )
    dataset_dict = DatasetDict({"train": dataset})

    return dataset_dict
