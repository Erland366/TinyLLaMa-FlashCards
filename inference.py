import argparse
import logging
import os

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline
from vllm import LLM, SamplingParams

from tinyllama_breakdown.templates.prompt_format import (
    GENERATE_DATA_ANKI_CARDS_JSONL_TEMPLATE,
)
from tinyllama_breakdown.utils import GPUMemoryTracker, Timeit, read_yaml_file

LOG = logging.getLogger("tinyllama_inference")
INFERENCE_SERVICE = ["awq", "hf"]
INFERENCE_TYPE = ["pipeline", "streaming", "regular"]
TEMPLATE_CONFIG = {
    "anki_jsonl": GENERATE_DATA_ANKI_CARDS_JSONL_TEMPLATE,
}


class TinyLLaMaInference:
    def __init__(self, cfg):
        self.model_id = cfg.main.model_name_or_path
        self.cfg = self.check_config(cfg)
        self.model, self.tokenizer = self.initialize_model(cfg)
        self.streamer = TextStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

    def initialize_model(self, cfg):
        model_id = self.model_id
        inference_service = cfg.extra.inference_service

        if cfg.extra.vllm:
            model = self.initialize_vllm_model(cfg)
        elif (
            inference_service == "awq" or self.extract_inference_type(model_id) == "awq"
        ):
            model = self.initialize_awq_model(cfg)
        elif inference_service == "hf":
            model = self.initialize_hf_model(cfg)
        else:
            raise ValueError(
                f"Inference service not supported, currently only supported {INFERENCE_SERVICE}"  # noqa: E501
            )
        return model

    def initialize_hf_model(self, cfg):
        try:
            model = AutoModelForCausalLM.from_pretrained(**cfg.main)
            tokenizer = AutoTokenizer.from_pretrained(**cfg.tokenizer)
        except TypeError:
            raise TypeError(
                "Arguments not found for HF model. Are you sure you are using HF config?"  # noqa: E501
            )

        return model, tokenizer

    def initialize_vllm_model(self, cfg):
        try:
            model = LLM(**cfg.main)
        except TypeError:
            raise TypeError(
                "Arguments not found for vLLM model. Are you sure you are using vLLM config?"  # noqa: E501
            )
        print("vLLM doesn't require tokenizer. Put None instead.")
        return model, None

    def initialize_awq_model(self, cfg) -> AutoAWQForCausalLM:
        try:
            model = AutoAWQForCausalLM.from_quantized(**cfg.main)
            tokenizer = AutoTokenizer.from_pretrained(**cfg.tokenizer)
        except TypeError:
            raise TypeError(
                "Arguments not found for AWQ model. Are you sure you are using AWQ config?"  # noqa: E501
            )
        return model, tokenizer

    def inference_pipeline(self, prompt: str, verbose: bool = False) -> str | None:
        pipe = pipeline(
            "text-generation",
            model=self.model_id,
            torch_dtype=self.cfg.main.torch_dtype,
            device_map=self.cfg.main.device_map,
        )
        prompt = self.formatted_prompt(prompt)

        with Timeit("Inference Pipeline"):
            sequences = pipe(
                prompt, eos_token_id=self.tokenizer.eos_token_id, **self.cfg.pipeline
            )
            if verbose:
                for sequence in sequences:
                    print(f"Result : {sequence['generated_text']}")

        return sequences

    def inference_streaming(self, prompt: str, verbose: bool = False) -> str | None:
        prompt = self.formatted_prompt(prompt)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        streamer = TextStreamer(self.tokenizer)
        _ = self.model.generate(
            **inputs,
            eos_token_id=[self.tokenizer.eos_token_id],
            streamer=streamer,
            **self.cfg.streaming,
        )

    def inference_regular(self, prompt: str, verbose: bool = False) -> str | None:
        prompt = self.formatted_prompt(prompt)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        print("Start Inference")
        with Timeit("Regular Inference") as timer, GPUMemoryTracker() as tracker:
            outputs = self.model.generate(
                **inputs,
                streamer=self.streamer,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.cfg.regular,
            )
            sequences = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if verbose:
            self.print_verbose(timer, tracker, outputs)
        return sequences

    def inference_vllm(self, prompt: str, verbose: bool = False) -> str | None:
        sampling_params = SamplingParams(**self.cfg.sampling)
        with Timeit("vLLM Inference") as timer, GPUMemoryTracker() as tracker:
            outputs = self.model.generate(prompt, sampling_params=sampling_params)
        if verbose:
            self.print_verbose(timer, tracker, outputs)

    def print_verbose(
        self, timer: Timeit, tracker: GPUMemoryTracker, outputs: torch.Tensor
    ):
        total_tokens = int(outputs.shape[1])
        print(f"Total tokens : {total_tokens}")
        print(f"Total tokens per second : {total_tokens / timer.elapsed:.2f}")
        print(f"Total memory usage : {str(tracker.difference)} GB")
        print(f"Before inference : {str(tracker.start_memory)} GB")
        print(f"After inference : {str(tracker.end_memory)} GB")

    def formatted_prompt(self, input_user: str) -> str:
        template = self.cfg.extra.prompt_template
        return template.format(input_user=input_user, response="")

    @staticmethod
    def extract_inference_type(model_id: str) -> str:
        return model_id.split("-")[-1]

    def check_config(self, cfg):
        if cfg.extra.inference_service not in INFERENCE_SERVICE:
            raise ValueError(
                f"Inference service not supported, currently only supported {INFERENCE_SERVICE}"  # noqa: E501
            )

        if cfg.extra.inference_type not in INFERENCE_TYPE:
            raise ValueError(
                f"Inference type not supported, currently only supported {INFERENCE_TYPE}"  # noqa: E501
            )

        if (
            cfg.extra.inference_service == "awq"
            and cfg.extra.inference_type != "regular"
        ):
            raise ValueError(
                "Inference type not supported, awq inference service only support regular inference type"  # noqa: E501
            )

        cfg.tokenizer.pretrained_model_name_or_path = (
            cfg.tokenizer.pretrained_model_name_or_path or cfg.main.model_name_or_path
        )

        if cfg.extra.template:
            cfg.extra.prompt_template = TEMPLATE_CONFIG[cfg.extra.template]

        if cfg.extra.vllm:
            cfg.main.model = cfg.main.model_name_or_path
            del cfg.main.model_name_or_path
        elif cfg.extra.inference_service == "awq":
            cfg.main.quant_path = cfg.main.model_name_or_path
            del cfg.main.model_name_or_path
        elif cfg.extra.inference_service == "hf":
            cfg.main.pretrained_model_name_or_path = cfg.main.model_name_or_path
            cfg.main.torch_dtype = getattr(torch, "float16")
            del cfg.main.dtype
            del cfg.main.model_name_or_path

        return cfg


def main(cfg) -> None:
    inference_object = TinyLLaMaInference(cfg)
    example_prompt = """Sekolah menengah atas negeri dan swasta tersebar di berbagai wilayah di tanah air. SMA merupakan salah satu pilihan yang banyak dipilih pelajar selepas lulus SMP. Melanjutkan sekolah SMA lebih berpeluang untuk masuk ke perguruan tinggi negeri dengan memilih jurusan kuliah yang diminatinya.

Pemilihan jurusan bukan hanya dilakukan pada jenjang pendidikan pada SMK. Namun juga berlaku pada siswa siswi yang memilih masuk ke SMA. Jurusan yang terdapat pada SMA, antara lain: jurusan IPA, IPS dan jurusan bahasa."""  # noqa: E501
    if cfg.extra.vllm:
        sequences = inference_object.inference_vllm(example_prompt, verbose=True)
    elif cfg.extra.inference_type == "pipeline":
        sequences = inference_object.inference_pipeline(example_prompt, verbose=False)
    elif cfg.extra.inference_type == "streaming":
        sequences = inference_object.inference_streaming(example_prompt, verbose=False)
    elif cfg.extra.inference_type == "regular":
        sequences = inference_object.inference_regular(example_prompt, verbose=True)
    print(f"Result : {sequences}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        choices=["hf", "awq", "hf_dummy"],
        help="Which experiment to run",
    )
    args = parser.parse_args()
    if args.config == "hf":
        cfg = read_yaml_file(os.path.join("config", "inference_hf.yaml"))
    elif args.config == "awq":
        cfg = read_yaml_file(os.path.join("config", "inference_awq.yaml"))
    elif args.config == "hf_dummy":
        cfg = read_yaml_file(os.path.join("config", "inference_hf_dummy.yaml"))

    main(cfg)
