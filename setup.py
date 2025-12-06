import os

import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import ROCM_HOME


def is_rocm():
    return os.path.exists(ROCM_HOME) and torch.version.hip

assert is_rocm()

setup(
    name="vllm_plugin",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),    
    python_requires=">=3.10.0",
    include_package_data=True,
    package_data={"vllm_plugin": ["**/*.csv", "**/*.json"]},
    entry_points={
        "vllm.general_plugins": [
            "register_custom_model = vllm_plugin.models:register",
            "register_quantization_config = vllm_plugin.quantization:register",
            "register_post_parsed_function = vllm_plugin.moreh_config:register_post_parsed_function",
        ],
        # "vllm.platform_plugins": [
        #     "moreh_platform_plugin = vllm_plugin.platform:moreh_platform_plugin",
        # ],
    },
)
