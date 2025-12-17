
## Setup and run:
### Clone repository and serve the container
```
git clone --recursive https://github.com/phucnguyen-ht/vllm_plugin.git
cd docker
docker compose -f image.yml up
docker exec -ti gptoss-plugin-vllm0.8.2 bash
```
It's important to note that it's required to include the following line in `mount` section:
```
../3rdparty/vllm:/usr/local/lib/python3.10/dist-packages/vllm
```
This is because we will use our customized vLLM on default version `0.8.2+e3e4b26`. The customized section details will be told in the `Appendix`
### Setup packages
```
cd scripts      # note that you need to change the $WORKING_DIR in this file
bash setup.sh
```
### Serve gpt-oss
```
bash serve.sh
```
### Run MMLU
```
bash mmlu.sh
```
### Run benchmark
```
bash benchmark2.sh
```

## Appendix:
### vLLM on modification:
All the patched diff listed in `assets/vllm_patch.diff`:
- attention backend: gpt-oss uses attention sink for its attention layer, so I `tham khảo` from latest vLLM to add sink in triton attention backend interface and triton kernels:
    - backend interface:
        - `v1/attention/backends/triton_attn.py`
    - triton kernels:
        - `attention/ops/chunked_prefill_paged_decode.py`
        - `attention/ops/prefix_prefill.py`
- rmsnorm: using hygon-rmsnorm return wrong result compared to MI250-rmsnorm, so I use native torch call in `model_executor/layers/layernorm.py`
- `vllm/v1/worker/gpu_model_runner.py:get_kv_cache_spec`: There is a change in this method which using in Attention module. The reason for this modification is that I use custom `MorehFusedMoE` class, this one must inherit from `FusedMoe` class. And this `FusedMoE` needs to be patched very much between version `0.8.2` and `0.10.1` (placed in `vllm_patch`). And because `vllm_patch/FusedMoE` is outside of vLLM project, so if I keep the original code, the error occurs here because `FusedMoE` in this `get_kv_cache_spec` refering to original `FusedMoE` in `v0.8.2`, not patched one.
    ```
        if isinstance(attn_module, FusedMoE):
            continue
    ```
    change to:
    ```
        if "FusedMoE" in attn_module.__class__.__name__:
            continue
    ```
- We must set `"torch_dtype": "bfloat16"` in `/openai/gpt-oss-20b/config.json` due to the following code in `vllm/config.py:_get_and_verify_dtype`. This method returns the type of model: 
    ```
    def _get_and_verify_dtype(
        config: PretrainedConfig,
        dtype: Union[str, torch.dtype],
    ) -> torch.dtype:
        # NOTE: getattr(config, "torch_dtype", torch.float32) is not correct
        # because config.torch_dtype can be None.
        config_dtype = getattr(config, "torch_dtype", None)
        
        ................

        if config_dtype is None:
            config_dtype = torch.float32

        if isinstance(dtype, str):
            dtype = dtype.lower()
            if dtype == "auto":
                if config_dtype == torch.float32:
                    # Following common practice, we use float16 for float32 models
                    torch_dtype = torch.float16
                else:
                    torch_dtype = config_dtype
    ```
    If we dont set `torch_dtype` as `bfloat16`, the type of model will be torch.float16. (The later vLLM `v0.10.1` change this method to automatically cast to `bfloat16` instead of `float16`)

- Logit Processor: The forward of this class wrongs with GPT-OSS. Here is the flow:
    ```
    model_executor.layers.logit_processor.py:LogitsProcessor.forward(...)
    model_executor.layers.logit_processor.py:LogitsProcessor._get_logits(...) -> lm_head.quant_method.apply(...)
    model_executor.layers.vocab_parallel_embedding.py:UnquantizedEmbeddingMethod.apply(...) -> F.linear(x, layer.weight, bias)
    ```
    After debugging by some print statemments, I found out that the output of this class is wrong when all returns to zero.
    ```
    name='logits'                  | dtype=torch.bfloat16  | shape=(1, 201088)          | 
    [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
    ```
    So, I have replaced it by a simple triton kernel in `vllm/external/matmul.py:triton_matmul`


### Execution flow to modify the logic of custom mxfp4 quantization:
- `src/vllm_plugin/quantization/mxfp4.py:MorehMxfp4MoEMethod.create_weights`: This method creates weight Tensor object to store checkpoint. Note that hidden_size and intermediate_size has been rounded up by factor of 256 (note: `v0.10.1` rounds hidden_size by 128 while latest version rounds by 256)
- `src/vllm_plugin/models/gpt_oss.py:MorehGptOssForCausalLM._load_weights_mxfp4`: This method loads weight from checkpoint to the weight object created by `create_weights` method above.
- `src/vllm_plugin/quantization/mxfp4.py:MorehMxfp4MoEMethod.process_weights_after_loading`: This method preprocesses the weight Tensor before applying to forward process
- `src/vllm_plugin/quantization/mxfp4.py:MorehMxfp4MoEMethod.apply`: This method implements the forward pass of MoE layer: fused_topk (to get `topk_weights` and `topk_ids`) -> pass to our custom kernel function: `rocm_gptoss_moreh_moe_1stage`
- `src/vllm_plugin/fused_moe/rocm_moreh_fused_moe.py:rocm_gptoss_moreh_moe_1stage`: The implementation of Moe after having `topk` metadata: `moe_sorting` (to accquire the token indexes sorted by expert - note that: the original `moe_sorting` using aiter/CK isnot supported so I have to use torch native function (`error: RuntimeError: HIP Function Failed … invalid device function`))
- `src/vllm_plugin/fused_moe/rocm_moreh_fused_moe.py:gptoss_moe_1stage`: This method calls the pybind function of our kernel implemented in `3rdparty/fused_moe/csrc/gfx928/a8w4_mxfp4`