from vllm import ModelRegistry
from vllm.logger import init_logger

logger = init_logger(__file__)

def _register_moreh_custom_model(arch_name , arch_cls):
    logger.debug(f"Register {arch_name}: {arch_cls}")
    ModelRegistry.register_model(arch_name, arch_cls)


def register():

    if "MorehGptOssForCausalLM" not in ModelRegistry.get_supported_archs():
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_plugin.models.gpt_oss_config import GptOssConfig
        
        # HACK: _CONFIG_REGISTRY doesn't expose a register API
        _CONFIG_REGISTRY["gpt_oss"] = GptOssConfig
    
        # Override both names to point to Moreh implementation
        _register_moreh_custom_model(
            "MorehGptOssForCausalLM",
            "vllm_plugin.models.gpt_oss:MorehGptOssForCausalLM",
        )
        _register_moreh_custom_model(
            "GptOssForCausalLM",
            "vllm_plugin.models.gpt_oss:MorehGptOssForCausalLM",
        )
        # _register_moreh_custom_model(
        #     "MorehGptOssForCausalLM",
        #     "vllm_plugin.models.gpt_oss:MorehGptOssForCausalLM",
        # )
