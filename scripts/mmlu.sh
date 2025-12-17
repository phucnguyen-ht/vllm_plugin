mkdir -p /root/.cache/huggingface/datasets
cp -r /home/tester/phucnguyen/hygon-test/dev/vllm_plugin_vllmm_0.8.2/datasets/longnguyen/.cache/huggingface/datasets/cais___mmlu /root/.cache/huggingface/datasets/

export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1

lm_eval --model local-completions \
  --tasks mmlu \
  --model_args model=/home/tester/data/openai/gpt-oss-20b,base_url=http://localhost:8000/v1/completions,num_concurrent=8,max_retries=3,tokenized_requests=False,timeout=10

# vllm serve /workspace/models/google/gemma-3-12b-it --disable-log-requests  --port 3434 
# lm_eval --model local-completions  --tasks mmlu  --model_args model=/workspace/models/tencent/Hunyuan-7B-Instruct,base_url=http://localhost:3434/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False,timeout=10