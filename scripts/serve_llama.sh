#!/bin/bash

# export VLLM_ROCM_USE_AITER=1
# export VLLM_USE_AITER_UNIFIED_ATTENTION=1
# export VLLM_ROCM_USE_AITER_MHA=0
# export VLLM_SERVER_DEV_MODE=1
# export TP=8 # both model can run with TP1
# export DP=1
# # export VLLM_MOREH_ALL2ALL_BACKEND="mori"
# # DP8-EP
# export VLLM_SERVER_DEV_MODE=1
# export VLLM_ROCM_USE_AITER_MOE=0

export VLLM_ROCM_USE_AITER=0
export VLLM_USE_AITER_UNIFIED_ATTENTION=0
export VLLM_ROCM_USER_AITER_MHA=0
export VLLM_USE_V1=1
export MODEL_PATH=/home/tester/data/meta-llama/Meta-Llama-3-8B
export VLLM_MOREH_USE_DUAL_MOE=1

# Change moreh-mxfp4 -> mxfp4 to run original mxfp4
vllm serve $MODEL_PATH \
    --max-model-len 1024 \
    --enforce-eager \
    --trust-remote-code
    # --enable-prefix-caching \