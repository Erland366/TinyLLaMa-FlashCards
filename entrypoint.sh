#!/bin/bash
huggingface-cli login --token $HUGGINGFACE_KEY # login to huggingface for private model access

echo "Model name: $MODEL"

python3 -m vllm.entrypoints.openai.api_server --host $HOST --port $PORT --swap-space $SWAP_SPACE --model $MODEL --dtype $DTYPE --quantization $QUANTIZATION --tokenizer $TOKENIZER --tensor-parallel-size $NUM_GPUS 