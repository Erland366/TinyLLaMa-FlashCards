from vllm import LLM, SamplingParams

from tinyllama_breakdown.templates.prompt_format import (
    GENERATE_DATA_ANKI_CARDS_JSONL_TEMPLATE,
)

# Sample prompts.
example_prompt = """Sekolah menengah atas negeri dan swasta tersebar di berbagai wilayah di tanah air. SMA merupakan salah satu pilihan yang banyak dipilih pelajar selepas lulus SMP. Melanjutkan sekolah SMA lebih berpeluang untuk masuk ke perguruan tinggi negeri dengan memilih jurusan kuliah yang diminatinya.

Pemilihan jurusan bukan hanya dilakukan pada jenjang pendidikan pada SMK. Namun juga berlaku pada siswa siswi yang memilih masuk ke SMA. Jurusan yang terdapat pada SMA, antara lain: jurusan IPA, IPS dan jurusan bahasa.
"""  # noqa: E50
the_thing = GENERATE_DATA_ANKI_CARDS_JSONL_TEMPLATE.format(
    input_user=example_prompt, response=""
)
prompts = [the_thing]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)

# Create an LLM.
llm = LLM(
    model="Erland/tinyllama-1.1B-chat-v0.3-dummy-AWQ",
    quantization="awq",
    dtype="half",
    tokenizer="hf-internal-testing/llama-tokenizer",
)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
