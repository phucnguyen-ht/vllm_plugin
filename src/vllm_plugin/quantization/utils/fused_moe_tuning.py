import functools
import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import torch

logger = logging.getLogger(__name__)

DEFAULT_MOE_TUNE_FILE = os.environ.get(
    "MOREH_MOE_TUNE_FILE",
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "configs",
        "fused_moe_a8w8.csv"))

@functools.lru_cache(maxsize=None)
def get_num_cu():
    props = torch.cuda.get_device_properties(0)
    return props.multi_processor_count


@dataclass
class TuningConfig:
    LOCAL_BLOCK_SIZE_M: int = 32
    PARALLEL_INTER_DIM: int = 1
    SPLIT_K: int = 1
    DIM_SIZE: int = 3072
    USE_NTLOAD_STAGE1: bool = False
    USE_NTLOAD_STAGE2: bool = False


def _get_empty_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "num_cu",
        "num_token",
        "hidden_dim",
        "inter_dim",
        "num_expert",
        "topk",
        "SHUFFLE_IN",
        "SHUFFLE_IK",
        "USE_FUSED_KERNEL",
        "LOCAL_BLOCK_SIZE_M",
        "PARALLEL_INTER_DIM",
        "SPLIT_K",
        "DIM_SIZE",
        "USE_NTLOAD_STAGE1",
        "USE_NTLOAD_STAGE2",
        "latency_us",
    ])


@functools.lru_cache(maxsize=None)
def _load_tuning_table(tune_file: str) -> pd.DataFrame:
    if not os.path.exists(tune_file):
        logger.warning_once(
            "Moreh MoE tuning file does not exist at %s. Using defaults.",
            tune_file)
        return _get_empty_dataframe()
    df = pd.read_csv(tune_file)
    return df


@functools.lru_cache(maxsize=None)
def get_tuning_dataframe(
    tune_file: str,
    hidden_dim: int,
    inter_dim: int,
    num_expert: int,
    topk: int,
    shuffle_in: int,
    shuffle_ik: int,
    use_fused_kernel: bool,
) -> pd.DataFrame:
    df = _load_tuning_table(tune_file)
    if df.empty:
        return df
    df = df[
        (df["num_cu"] == get_num_cu()) &
        (df["hidden_dim"] == hidden_dim) &
        (df["inter_dim"] == inter_dim) &
        (df["num_expert"] == num_expert) &
        (df["topk"] == topk) &
        (df["SHUFFLE_IN"] == shuffle_in) &
        (df["SHUFFLE_IK"] == shuffle_ik) &
        (df["USE_FUSED_KERNEL"] == use_fused_kernel)
    ].copy()
    df.sort_values(by="num_token", inplace=True)
    return df


@functools.lru_cache(maxsize=None)
def get_tuning_config_caching(
    num_token: int,
    hidden_dim: int,
    inter_dim: int,
    num_expert: int,
    topk: int,
    shuffle_in: int,
    shuffle_ik: int,
    use_fused_kernel: bool,
    tune_file: str = DEFAULT_MOE_TUNE_FILE,
) -> Tuple[Optional[TuningConfig], Optional[int], Optional[float]]:
    df = get_tuning_dataframe(
        tune_file,
        hidden_dim,
        inter_dim,
        num_expert,
        topk,
        shuffle_in,
        shuffle_ik,
        use_fused_kernel,
    )
    if df.empty:
        return None, None, None
    df = df.copy()
    df["num_token_diff"] = (df["num_token"] - num_token).abs()
    df_sorted = df.sort_values(by=["num_token_diff", "latency_us"])
    best_config = df_sorted.iloc[0]
    config = TuningConfig(
        LOCAL_BLOCK_SIZE_M=int(best_config.get("LOCAL_BLOCK_SIZE_M", 32)),
        PARALLEL_INTER_DIM=int(best_config.get("PARALLEL_INTER_DIM", 1)),
        SPLIT_K=int(best_config.get("SPLIT_K", 1)),
        DIM_SIZE=int(best_config.get("DIM_SIZE", 3072)),
        USE_NTLOAD_STAGE1=bool(best_config.get("USE_NTLOAD_STAGE1", False)),
        USE_NTLOAD_STAGE2=bool(best_config.get("USE_NTLOAD_STAGE2", False)),
    )
    num_token_diff = int(best_config.get("num_token_diff", 0))
    latency_us = float(best_config.get("latency_us", float("inf")))
    return config, num_token_diff, latency_us


@functools.lru_cache(maxsize=None)
def get_best_tuning_config(
    num_token: int,
    hidden_dim: int,
    inter_dim: int,
    num_expert: int,
    topk: int,
    shuffle_in: int,
    shuffle_ik: int,
    use_fused_kernel: bool,
    tune_file: str = DEFAULT_MOE_TUNE_FILE,
) -> TuningConfig:
    config, _, _ = get_tuning_config_caching(
        num_token=num_token,
        hidden_dim=hidden_dim,
        inter_dim=inter_dim,
        num_expert=num_expert,
        topk=topk,
        shuffle_in=shuffle_in,
        shuffle_ik=shuffle_ik,
        use_fused_kernel=use_fused_kernel,
        tune_file=tune_file,
    )
    if config is None:
        # logger.warning_once(
        #     ("No tuning config found for num_token=%s, hidden_dim=%s, "
        #      "inter_dim=%s, num_expert=%s, topk=%s, SHUFFLE_IN=%s, "
        #      "SHUFFLE_IK=%s, USE_FUSED_KERNEL=%s. Using default."),
        #     num_token, hidden_dim, inter_dim, num_expert, topk, shuffle_in,
        #     shuffle_ik, use_fused_kernel)
        config = TuningConfig()
    return config


@functools.lru_cache(maxsize=None)
def is_fused_moe_1stage_better_than_2stages(
    num_token: int,
    hidden_dim: int,
    inter_dim: int,
    num_expert: int,
    topk: int,
    shuffle_in: int,
    shuffle_ik: int,
    tune_file: str = DEFAULT_MOE_TUNE_FILE,
) -> bool:
    config_1stage, diff_1stage, lat_1stage = get_tuning_config_caching(
        num_token=num_token,
        hidden_dim=hidden_dim,
        inter_dim=inter_dim,
        num_expert=num_expert,
        topk=topk,
        shuffle_in=shuffle_in,
        shuffle_ik=shuffle_ik,
        use_fused_kernel=True,
        tune_file=tune_file,
    )
    config_2stage, diff_2stage, lat_2stage = get_tuning_config_caching(
        num_token=num_token,
        hidden_dim=hidden_dim,
        inter_dim=inter_dim,
        num_expert=num_expert,
        topk=topk,
        shuffle_in=shuffle_in,
        shuffle_ik=shuffle_ik,
        use_fused_kernel=False,
        tune_file=tune_file,
    )
    if config_1stage is None and config_2stage is None:
        return True
    if config_1stage is None:
        return False
    if config_2stage is None:
        return True
    if diff_1stage == diff_2stage:
        return lat_1stage < lat_2stage
    return diff_1stage < diff_2stage
