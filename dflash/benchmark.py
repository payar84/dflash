from __future__ import annotations

import argparse
import os
import random
import re
import statistics
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
from types import SimpleNamespace
from typing import Any, List, Optional

import numpy as np
import requests
import torch
from torch import distributed as torch_dist
from loguru import logger
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .model import DFlashDraftModel, sample, load_and_process_dataset, extract_context_feature


def _dist_init() -> None:
    if "RANK" not in os.environ:
        warnings.warn("RANK not set. Skipping distributed initialization.")
        return
    torch_dist.init_process_group(backend="nccl", init_method="env://")

def _dist_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))

def _dist_rank() -> int:
    return int(os.environ.get("RANK", 0))

def _dist_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))

def _dist_is_main() -> bool:
    return _dist_rank() == 0

def _dist_gather(obj: Any, dst: int = 0) -> Optional[List[Any]]:
    if not torch_dist.is_initialized():
        return [obj]
    if _dist_is_main():
        objs: List[Any] = [None for _ in range(_dist_size())]
        torch_dist.gather_object(obj, objs, dst=dst)
        return objs
    else:
        torch_dist.gather_object(obj, dst=dst)
        return None


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def _dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    # NOTE: Setting verbose=True here helps me track per-step acceptance during local debugging
    verbose: bool = True,  # changed default to True so I can see acceptance rates without extra flags
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    prefill_start = _cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
    # Fix: use prefill_start instead of undefined variable `pr`
    time_to_first_token = _cuda_time() - prefill_start
