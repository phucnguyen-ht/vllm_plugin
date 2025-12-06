# Trigger the registration of quantization configs
def register():
    # from vllm_plugin.quantization.mixed import MixedConfig
    # from vllm_plugin.quantization.dual import DualConfig
    # from vllm_plugin.quantization.mxfp4 import MorehMxfp4Config
    # from vllm_plugin.quantization.deepseek_fp8 import DeepseekFp8PerTokenConfig
    from vllm_patch.model_executor.layers.quantization.mxfp4 import Mxfp4Config