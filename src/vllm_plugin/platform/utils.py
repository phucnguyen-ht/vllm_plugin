import torch
from functools import cache


@cache
def on_mi300x_mi308x_mi325x() -> bool:
    GPU_ARCH = torch.cuda.get_device_properties("cuda").gcnArchName
    return "gfx942" in GPU_ARCH

@cache
def on_mi250x() -> bool:
    GPU_ARCH = torch.cuda.get_device_properties("cuda").gcnArchName
    return "gfx90a" in GPU_ARCH