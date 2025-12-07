1. Clone repository:
```
git clone --recursive https://github.com/phucnguyen-ht/vllm_plugin.git
```
2. On Hygon-device:
```
cd docker
docker compose -f image.yml up -d test-gptoss-plugin-vllm-0.8.2
```
Note that: If you run original 'mxfp4' quantization using triton_kernels, it is required to get modified triton/triton_kernels version before running docker:
```
    cp -r /home/tester/phucnguyen/hygon-test/dev/vllm_plugin/3rdparty/triton 3rdparty/
    cp -r /home/tester/phucnguyen/hygon-test/dev/vllm_plugin/3rdparty/triton_kernels 3rdparty/

    # Uncomment two following lines in image.yml:
    # - ../3rdparty/triton:/usr/local/lib/python3.10/dist-packages/triton
    # - ../3rdparty/triton_kernels:/usr/local/lib/python3.10/dist-packages/triton_kernels
```
3. Setup fused_moe (just placeholder):
```
    cd scripts
    bash setup.sh
```
4. Serve gpt-oss:
Note that: To change from original mxfp4 quantization, simply replacing `--quantization moreh-mxfp4` to `--quantization mxfp4`
```
    bash serve.sh
```

5. This main branch develops the gpt-oss plugin on original vllm image v0.8.2. To serve in the upstream vllm (v0.10.1), checkout to branch:
```
    git checkout dev/vllm_upstream_v0.10.1
    docker compose -f image.yml up -d test-gptoss-plugin-vllm-0.10.1
```

6. Execution flow to modify the logic of custom mxfp4 quantization:
- src/vllm_plugin/quantization/mxfp4.py:MorehMxfp4MoEMethod.create_weights: This method creates weight Tensor object to store checkpoint. Note that hidden_size and intermediate_size has been rounded up by factor of 256 (note: vllm0.10.1 rounds hidden_size by 128 while latest version rounds by 256)
- src/vllm_plugin/models/gpt_oss.py:MorehGptOssForCausalLM._load_weights_mxfp4: This method loads weight from checkpoint to the weight object created by `create_weights` method above.
- src/vllm_plugin/quantization/mxfp4.py:MorehMxfp4MoEMethod.process_weights_after_loading: This method preprocesses the weight Tensor before applying to forward process
- src/vllm_plugin/quantization/mxfp4.py:MorehMxfp4MoEMethod.apply: This method implements the forward pass of MoE layer: fused_topk (to get topk_weights and topk_ids) -> pass to our custom kernel function: `rocm_gptoss_moreh_moe_1stage`
- src/vllm_plugin/fused_moe/rocm_moreh_fused_moe.py:rocm_gptoss_moreh_moe_1stage: The implementation of Moe after having `topk` metadata: `moe_sorting` (to accquire the token indexes sorted by expert - note that: the original moe_sorting using aiter/CK isnot supported so I have to use torch native function (error: RuntimeError: HIP Function Failed … invalid device function))
- src/vllm_plugin/fused_moe/rocm_moreh_fused_moe.py:gptoss_moe_1stage: This method calls the pybind function of our kernel implemented in `3rdparty/fused_moe/csrc/gfx928/a8w4_mxfp4`