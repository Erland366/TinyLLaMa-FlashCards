import contextlib
import json
import logging
import os
from time import perf_counter

import huggingface_hub
import yaml
from addict import Dict
from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.utils import LocalTokenNotFoundError
from transformers import AutoTokenizer

from tinyllama_breakdown.bench import gpu_memory_usage_all

LOG = logging.getLogger("tinyllama_breakdown")
load_dotenv()


def read_yaml_file(file_path):
    with open(file_path, "r") as f:
        try:
            LOG.info(f"Reading {file_path} config.")
            return DictDefault(yaml.safe_load(f))
        except yaml.YAMLError as e:
            print(e)


def check_user_token():
    api = HfApi()
    try:
        user_info = api.whoami()
        return bool(user_info)
    except LocalTokenNotFoundError:
        LOG.warning(
            "No user token found. Please run `huggingface-cli login` \
to log in or put HF_TOKEN_API on .env."
        )
        return False


def parse_output(text: str):
    # Split the text into sections
    sections = text.split("### ")

    # Find the last Response section
    last_response_section = None
    for section in reversed(sections):
        if "Response:" in section:
            last_response_section = section
            break

    # Check if a Response section was found
    if last_response_section is not None:
        # Split the section into lines
        lines = last_response_section.split("\n")

        # Parse the valid JSON lines
        valid_responses = []
        for line in lines:
            try:
                response = json.loads(line)
                valid_responses.append(response)
            except json.JSONDecodeError:
                pass  # Ignore lines that are not valid JSON
        return valid_responses
    else:
        raise ValueError("No Response section can be parsed.")


def load_user_token() -> None:
    try:
        LOG.info("Loading HF_TOKEN_API from .env.")
        huggingface_hub.login(os.environ["HF_TOKEN_API"])
    except KeyError:
        LOG.warning("No HF_TOKEN_API found on .env.")


def list_of_json_to_jsonl(list_of_json: list[dict], output_path: str) -> None:
    with open(output_path, "w") as f:
        for json_object in list_of_json:
            json_str = json.dumps(json_object)
            f.write(json_str + "\n")


class TokenCounter:
    @classmethod
    def count(self, tokenizer_name_or_path: str, text: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        tokenized = tokenizer(text, return_tensors="pt")
        print(f"Number of tokens: {tokenized.input_ids.shape[1]}")


class Timeit:
    def __init__(self, name: str):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, *args):
        self.end = perf_counter()
        self.elapsed = self.end - self.start
        LOG.info(f"{self.name} took {self.elapsed:.2f} seconds.")
        print(f"{self.name} took {self.elapsed:.2f} seconds.")


class DictDefault(Dict):
    """
    A Dict that returns None instead of returning empty Dict for missing keys.

    Taken from : https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/utils/dict.py
    """

    def __missing__(self, key):
        return None

    def __or__(self, other):
        return DictDefault(super().__or__(other))


class GPUMemoryTracker:
    def __init__(self, device=0):
        self.device = device
        self.start_memory = None
        self.end_memory = None
        self.difference = None

    def __enter__(self):
        self.start_memory = gpu_memory_usage_all(self.device)
        return self

    def __exit__(self, *args):
        self.end_memory = gpu_memory_usage_all(self.device)
        self.difference = tuple(
            end - start for end, start in zip(self.end_memory, self.start_memory)
        )


@contextlib.contextmanager
def gpu_memory_tracker(device=0):
    start_memory = gpu_memory_usage_all(device)
    yield
    end_memory = gpu_memory_usage_all(device)
    difference = tuple(end - start for start, end in zip(start_memory, end_memory))
    print(f"Memory usage before inference: {start_memory}")
    print(f"Memory usage after inference: {end_memory}")
    print(f"Memory usage difference: {difference}")
