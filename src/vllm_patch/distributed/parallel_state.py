from vllm.distributed.parallel_state import init_model_parallel_group, GroupCoordinator

from functools import cache

_EP: GroupCoordinator | None = None

@cache
def get_ep_group() -> GroupCoordinator:
    # assert _EP is not None, "expert parallel group is not initialized"
    print("-" * 50 + "Call get_ep_group" + "-" * 50)
    _EP = init_model_parallel_group(
        group_ranks=[[0]],
        local_rank=0,
        backend='nccl',
        use_message_queue_broadcaster=False,
        group_name="ep"
    )
    return _EP