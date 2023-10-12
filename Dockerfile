FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN apt update && apt install -y python3 python3-pip git nano

RUN pip install transformers==4.33.2
RUN pip install vllm==0.2.0
RUN pip install fschat

ENV HOST=0.0.0.0
ENV PORT=5000
ENV SWAP_SPACE=2
ENV MODEL=Erland/tinyllama-1.1B-chat-v0.3-dummy-AWQ
ENV NUM_GPUS=1
ENV TOKENIZER=hf-internal-testing/llama-tokenizer
ENV QUANTIZATION=awq
ENV DTYPE=float16

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["/entrypoint.sh"]