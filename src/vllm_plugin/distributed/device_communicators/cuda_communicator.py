# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.device_communicators.cuda_communicator import *

class MorehCudaCommunicator(CudaCommunicator):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        DeviceCommunicatorBase.__init__(self, cpu_group, device, device_group, unique_name)
        if "tp" not in unique_name:
            # custom allreduce or torch symm mem can be used only by tp
            use_custom_allreduce = False
            use_torch_symm_mem = False
        else:
            from vllm.distributed.parallel_state import _ENABLE_CUSTOM_ALL_REDUCE

            use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE
            use_torch_symm_mem = envs.VLLM_ALLREDUCE_USE_SYMM_MEM

        self.use_custom_allreduce = use_custom_allreduce
        self.use_torch_symm_mem = use_torch_symm_mem

        # lazy import to avoid documentation build error
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )
        from vllm.distributed.device_communicators.symm_mem import SymmMemCommunicator

        self.pynccl_comm: PyNcclCommunicator | None = None
        if self.world_size > 1:
            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group,
                device=self.device,
            )
            if is_symmetric_memory_enabled():
                register_nccl_symmetric_ops(self.pynccl_comm)

        self.ca_comm: CustomAllreduce | None = None
        self.qr_comm: QuickAllReduce | None = None
        self.symm_mem_comm: SymmMemCommunicator | None = None
        if use_torch_symm_mem and current_platform.is_cuda():
            self.symm_mem_comm = SymmMemCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

        if use_custom_allreduce and self.world_size > 1:
            # Initialize a custom fast all-reduce implementation.
            self.ca_comm = CustomAllreduce(
                group=self.cpu_group,
                device=self.device,
                symm_mem_enabled=(
                    self.symm_mem_comm is not None and not self.symm_mem_comm.disabled
                ),
            )

            if current_platform.is_rocm():
                # Initialize a custom quick all-reduce implementation for AMD.
                # Quick reduce is designed as a complement to custom allreduce.
                # Based on quickreduce (https://github.com/mk1-project/quickreduce).
                # If it's a rocm, 'use_custom_allreduce==True' means it must
                # currently be an MI300 series.
                self.qr_comm = QuickAllReduce(group=self.cpu_group, device=self.device)

        if self.use_all2all:
            if self.all2all_backend == "naive":
                from vllm.distributed.device_communicators.all2all import NaiveAll2AllManager

                self.all2all_manager = NaiveAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "allgather_reducescatter":
                from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager

                self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "pplx":
                from vllm.distributed.device_communicators.all2all import PPLXAll2AllManager

                self.all2all_manager = PPLXAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "deepep_high_throughput":
                from vllm.distributed.device_communicators.all2all import DeepEPHTAll2AllManager

                self.all2all_manager = DeepEPHTAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "deepep_low_latency":
                from vllm.distributed.device_communicators.all2all import DeepEPLLAll2AllManager

                self.all2all_manager = DeepEPLLAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "flashinfer_all2allv":
                from vllm.distributed.device_communicators.all2all import FlashInferAllToAllManager

                self.all2all_manager = FlashInferAllToAllManager(self.cpu_group)
            elif self.all2all_backend == "mori": # [Moreh] Adding mori backend
                from vllm_moreh.distributed.device_communicators.all2all import MoriAll2AllManager

                self.all2all_manager = MoriAll2AllManager(self.cpu_group)
            else:
                raise ValueError(f"Unknown all2all backend: {self.all2all_backend}")

            if is_global_first_rank():
                logger.info(
                    "Using %s all2all manager.",
                    self.all2all_manager.__class__.__name__,
                )

    def prepare_communication_buffer_for_model(self, model: torch.nn.Module) -> None:
        """
        Prepare the communication buffer for the model.
        """
        if not self.is_ep_communicator:
            return

        moe_modules = [
            module
            for module in model.modules()
            # TODO(bnell): Should use isinstance but can't.  Maybe search for
            # presence of quant_method.init_prepare_finalize?
            if (
                module.__class__.__name__ == "FusedMoE"
                or module.__class__.__name__ == "SharedFusedMoE"
                or module.__class__.__name__ == "MorehFusedMoE" # [Moreh] Adding for MorehFusedMoE
            )
        ]

        for module in moe_modules:
            module.quant_method.init_prepare_finalize(module)