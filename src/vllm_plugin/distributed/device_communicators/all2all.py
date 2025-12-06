import json
import os
from pathlib import Path
from typing import Any

import psutil
import torch

from vllm.logger import init_logger
from vllm_moreh.utils import has_mori
from vllm.distributed.device_communicators.base_device_communicator import All2AllManagerBase, Cache

import vllm_moreh.envs as envs

from vllm.logger import init_logger
logger = init_logger(__name__)

class MoriAll2AllManager(All2AllManagerBase):
    """
    All2All communication based on mori kernels.
    """

    def __init__(self, cpu_group):
        assert has_mori(), "Please install mori from ROCm/mori github."

        super().__init__(cpu_group)
        self.handle_cache = Cache()
        self.config = None
        self._shmem_initialized = False

        self.json_config = None
        config_path = envs.VLLM_MOREH_MORI_CONFIG_PATH
        if config_path:
            self.json_config = self._load_mori_config_from_json(config_path)

        # Delay mori shmem initialization until first use
        logger.debug("[rank %s] MoriAll2AllManager created", self.rank)

    def _ensure_shmem_initialized(self):
        """Initialize mori's shared memory system lazily"""
        if self._shmem_initialized:
            return

        import torch.distributed as dist
        from mori.shmem import shmem_torch_process_group_init

        try:
            # Check if we have a valid backend
            backend = dist.get_backend()
            if backend is None:
                raise RuntimeError("No valid distributed backend found")

            logger.debug(
                "[rank %s] PyTorch distributed ready with backend: %s",
                self.rank,
                backend,
            )

            assert self.cpu_group is not None, "No CPU group is given to mori"
            ppid = psutil.Process(os.getpid()).ppid()
            group_name = f"mori_shmem_group_{ppid}"

            try:
                import torch._C._distributed_c10d as c10d

                # Register the process group
                c10d._register_process_group(group_name, self.cpu_group)
                logger.debug(
                    "[rank %s] Registered proc group %s", self.rank, group_name
                )

                # Initialize mori shmem with the registered group
                shmem_torch_process_group_init(group_name)
                logger.debug("[rank %s] torch proc group shmem init success", self.rank)
                self._shmem_initialized = True
                return

            except Exception as torch_error:
                raise RuntimeError(
                    "torch process group initialization failed"
                ) from torch_error

        except Exception as e:
            raise RuntimeError("mori shmem initialization failed") from e

    def _load_mori_config_from_json(self, json_path: str) -> dict | None:
        """
        Load mori configuration parameters from JSON file.

        Supports both flat and hierarchical schema:

        Flat schema:
        {
            "warp_num_per_block": 8,
            "block_num": 80,
        }

        Hierarchical schema (dispatch/combine specific):
        {
            "global": {
                "warp_num_per_block": 8,
                "block_num": 80,
            },
            "dispatch": {
                "warp_num_per_block": 16,
                "block_num": 160
            },
            "combine": {
                "warp_num_per_block": 4,
                "block_num": 40
            }
        }

        Args:
            json_path: Path to JSON configuration file

        Returns:
            Dictionary of configuration parameters, or None if file doesn't exist

        Raises:
            ValueError: If JSON is invalid or contains unsupported parameters
        """
        if not json_path:
            return None

        json_file = Path(json_path)
        if not json_file.exists():
            logger.warning(
                "[rank %d] Mori config file not found: %s", self.rank, json_path
            )
            return None

        try:
            with open(json_file) as f:
                config = json.load(f)

            # Valid parameter keys
            valid_param_keys = {
                "warp_num_per_block",
                "block_num",
            }

            is_hierarchical = any(
                key in config for key in ["global", "dispatch", "combine"]
            )

            if is_hierarchical:
                valid_top_keys = {"global", "dispatch", "combine"}
                invalid_keys = set(config.keys()) - valid_top_keys
                if invalid_keys:
                    raise ValueError(
                        f"Invalid top-level keys: {invalid_keys}. "
                        f"Valid keys: {valid_top_keys}"
                    )

                # Validate each section
                for section in ["global", "dispatch", "combine"]:
                    if section in config:
                        section_config = config[section]
                        if not isinstance(section_config, dict):
                            raise ValueError(f"'{section}' must be a dictionary")

                        invalid_keys = set(section_config.keys()) - valid_param_keys
                        if invalid_keys:
                            raise ValueError(
                                f"Invalid keys in '{section}': {invalid_keys}. "
                                f"Valid keys: {valid_param_keys}"
                            )
            else:
                invalid_keys = set(config.keys()) - valid_param_keys
                if invalid_keys:
                    raise ValueError(
                        f"Invalid config keys: {invalid_keys}. "
                        f"Valid keys: {valid_param_keys}"
                    )

            return config

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in mori config file {json_path}") from e
        except Exception as e:
            raise ValueError(f"Error loading mori config from {json_path}") from e

    def _make_mori_config(
        self,
        max_num_tokens: int,
        num_local_experts: int,
        experts_per_token: int,
        hidden_dim: int,
        scale_dim: int,
        scale_type_size: int,
        data_type: torch.dtype = torch.bfloat16,
        quant_dtype: torch.dtype | None = None,
    ):
        """
        Create mori EpDispatchCombineConfig.

        Args:
            max_num_tokens: Maximum number of tokens per DP rank
            num_local_experts: Number of local experts
            experts_per_token: Number of experts per token (topk)
            hidden_dim: Hidden dimension size
            scale_dim: Scale dimension for quantization
            scale_type_size: Scale type size for quantization
            data_type: Tensor data type
            quant_dtype: Quantization data type (optional)
        """
        from mori.ops import EpDispatchCombineConfig
        from mori.ops.dispatch_combine import EpDispatchCombineKernelType

        from vllm.platforms import current_platform

        assert quant_dtype is None or quant_dtype == current_platform.fp8_dtype()

        # Default values (can be overridden by JSON)
        warp_num_per_block = 8
        block_num = 80

        # Override with JSON config if provided
        if self.json_config is not None:
            is_hierarchical = any(
                key in self.json_config for key in ["global", "dispatch", "combine"]
            )

            global_config = self.json_config
            if is_hierarchical and "global" in global_config:
                global_config = self.json_config["global"]

            warp_num_per_block = global_config.get(
                "warp_num_per_block", warp_num_per_block
            )
            block_num = global_config.get("block_num", block_num)

        config = EpDispatchCombineConfig(
            data_type=data_type if quant_dtype is None else quant_dtype,
            rank=self.rank,
            world_size=self.world_size,
            hidden_dim=hidden_dim,
            max_num_inp_token_per_rank=max_num_tokens,
            num_experts_per_rank=num_local_experts,
            num_experts_per_token=experts_per_token,
            max_token_type_size=data_type.itemsize,
            # Performance tuning parameters
            warp_num_per_block=warp_num_per_block,
            block_num=block_num,
            # Quantization support
            scale_dim=scale_dim,
            scale_type_size=scale_type_size,
            # Determine kernel type based on topology
            kernel_type=(
                EpDispatchCombineKernelType.InterNode
                if self.internode
                else EpDispatchCombineKernelType.IntraNode
            ),
        )

        return config

    def get_handle(self, kwargs):
        """
        Get or create mori operation handle.
        Args:
            kwargs: Dictionary with keys:
                - max_num_tokens: Maximum tokens per DP rank
                - num_local_experts: Number of local experts
                - experts_per_token: Number of experts per token (topk)
                - hidden_dim: Hidden dimension size
                - data_type: Tensor data type (optional, default bfloat16)
                - scale_dim: Scale dimension (optional)
                - scale_type_size: Scale type size (optional)
                - ubatch_id: Microbatch ID (optional)
        """
        # Ensure shmem is initialized before creating handles
        self._ensure_shmem_initialized()

        def create_mori_handle(
            max_num_tokens: int,
            num_local_experts: int,
            experts_per_token: int,
            hidden_dim: int,
            scale_dim: int,
            scale_type_size: int,
            data_type: torch.dtype = torch.bfloat16,
            quant_dtype: torch.dtype | None = None,
        ):
            from mori.ops import EpDispatchCombineOp

            config = self._make_mori_config(
                max_num_tokens=max_num_tokens,
                num_local_experts=num_local_experts,
                experts_per_token=experts_per_token,
                hidden_dim=hidden_dim,
                scale_dim=scale_dim,
                scale_type_size=scale_type_size,
                data_type=data_type,
                quant_dtype=quant_dtype,
            )
            op = EpDispatchCombineOp(config)
            logger.debug(
                "[rank %s] Created mori handle with config: tokens=%d, experts=%d,"
                " topk=%d, hidden_dim=%d",
                self.dp_rank,
                max_num_tokens,
                num_local_experts,
                experts_per_token,
                hidden_dim,
            )
            return op

        return self.handle_cache.get_or_create(kwargs, create_mori_handle)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
    ):
        raise NotImplementedError

    def combine(
        self,
        hidden_states: torch.Tensor,
        is_sequence_parallel: bool = False,
    ):
        raise NotImplementedError

    def destroy(self):
        """Clean up mori resources"""
        try:
            # Clear operation handle cache
            with self.handle_cache._lock:
                for _, handle in self.handle_cache._cache.items():
                    handle.destroy()

            # finalize mori shared memory if it was initialized
            if self._shmem_initialized:
                try:
                    from mori.shmem import shmem_finalize

                    # Check if shmem is actually active before finalizing
                    shmem_finalize()
                    logger.debug("[rank %s] mori shmem finalize", self.dp_rank)
                except Exception as shmem_error:
                    logger.debug(
                        "[rank %s] shmem finalize failed "
                        "(may not have been active): %s",
                        self.dp_rank,
                        shmem_error,
                    )

            logger.debug("[rank %s] mori resources cleaned up", self.dp_rank)

        except Exception as e:
            logger.warning("[rank %s] mori cleanup fail: %s", self.dp_rank, e)