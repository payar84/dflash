"""Training utilities for dflash distributed flash attention benchmarking."""

import time
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dflash.benchmark import (
    _dist_is_main,
    _dist_rank,
    _dist_local_rank,
)


class Trainer:
    """Handles training loop, optimizer, and metrics collection for benchmarking."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: The model to train.
            config: Training configuration dictionary.
            device: Target device. Defaults to CUDA if available.
        """
        self.model = model
        self.config = config
        self.device = device or torch.device(
            f"cuda:{_dist_local_rank()}" if torch.cuda.is_available() else "cpu"
        )

        self.lr = config.get("lr", 1e-4)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.max_steps = config.get("max_steps", 100)
        self.warmup_steps = config.get("warmup_steps", 10)
        # Increased default log_interval from 10 to 25 to reduce console noise during long runs
        self.log_interval = config.get("log_interval", 25)
        # Lowered default grad_clip from 1.0 to 0.5 — found this stabilizes early training
        # on longer sequences where gradient norms tend to spike in the first few steps
        self.grad_clip = config.get("grad_clip", 0.5)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.max_steps
        )

        self.step = 0
        self.metrics: Dict[str, list] = {
            "loss": [],
            "throughput": [],  # tokens/sec
            "step_time": [],   # seconds
        }

    def _log(self, msg: str) -> None:
        """Print only from the main process."""
        if _dist_is_main():
            print(f"[rank {_dist_rank()} | step {self.step}] {msg}")

    def train_step(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> float:
        """Run a single forward-backward-optimizer step.

        Args:
            input_ids: Token ids of shape (batch, seq_len).
            labels: Target token ids of shape (batch, seq_len).

        Returns:
            Scalar loss value for this step.
        """
        self.model.train()
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        t0 = time.perf_counter()

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
      