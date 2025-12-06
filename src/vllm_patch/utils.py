import importlib
from functools import cache

import torch

from packaging import version
from packaging.version import Version

def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)

@cache
def _has_module(module_name: str) -> bool:
    """Return True if *module_name* can be found in the current environment.

    The result is cached so that subsequent queries for the same module incur
    no additional overhead.
    """
    return importlib.util.find_spec(module_name) is not None

def has_triton_kernels() -> bool:
    """Whether the optional `triton_kernels` package is available."""

    return _has_module("triton_kernels")

# Helper function used in testing.
def _is_torch_equal_or_newer(torch_version: str, target: str) -> bool:
    torch_version = version.parse(torch_version)
    return torch_version >= version.parse(target)

def is_torch_equal_or_newer(target: str) -> bool:
    """Check if the installed torch version is >= the target version.

    Args:
        target: a version string, like "2.6.0".

    Returns:
        Whether the condition meets.
    """
    try:
        return _is_torch_equal_or_newer(str(torch.__version__), target)
    except Exception:
        # Fallback to PKG-INFO to load the package info, needed by the doc gen.
        return Version(importlib.metadata.version("torch")) >= Version(target)
    
def has_pplx() -> bool:
    """Whether the optional `pplx_kernels` package is available."""

    return _has_module("pplx_kernels")


def has_deep_ep() -> bool:
    """Whether the optional `deep_ep` package is available."""

    return _has_module("deep_ep")