from typing import Optional

from .moreh_rocm_platform import MorehRocmPlatform

def moreh_platform_plugin() -> Optional[str]:
    return "vllm_plugin.platform.MorehRocmPlatform"
