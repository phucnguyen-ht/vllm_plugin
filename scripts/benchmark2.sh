#!/usr/bin/env bash
set -euo pipefail
shopt -s dotglob

#############################################
# Configuration
#############################################

# Scale factor: num_prompts = max_concurrency × SCALE
SCALE=3

export MODEL_DIR=/home/tester/data/openai/gpt-oss-20b

# (input_len  output_len  max_concurrency)
TEST_CASES=(
  # "512 512 1"
  # "512 512 8"
  # "512 512 64"
  # "512 512 256"
  "4096 1024 1"
  "4096 1024 8"
  "4096 1024 64"
  "4096 1024 256"
  "32768 1024 1"
  "32768 1024 8"
  "32768 1024 32"
)

PORT=8000


#############################################
# Main loop
#############################################

for test_case in "${TEST_CASES[@]}"; do
  # Split the triple into individual variables
  read -r INPUT_LEN OUTPUT_LEN MAX_CONCURRENCY <<< "$test_case"

  # Compute num_prompts = max_concurrency × SCALE
  NUM_PROMPTS=$(( MAX_CONCURRENCY * SCALE ))

  echo "Running: input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN}, " \
       "max_concurrency=${MAX_CONCURRENCY}, num_prompts=${NUM_PROMPTS}"

  vllm bench serve \
    --endpoint /v1/completions \
    --port "${PORT}" \
    --model $MODEL_DIR \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --ignore-eos \

done