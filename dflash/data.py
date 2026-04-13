"""Data loading and preprocessing utilities for dflash."""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Iterator

import torch
from torch.utils.data import DataLoader, DistributedSampler

from dflash.config import DataConfig

logger = logging.getLogger(__name__)


@dataclass
class DataBatch:
    """A single batch of tokenized data."""
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "DataBatch":
        """Move batch tensors to the specified device."""
        return DataBatch(
            input_ids=self.input_ids.to(device),
            labels=self.labels.to(device),
            attention_mask=(
                self.attention_mask.to(device)
                if self.attention_mask is not None
                else None
            ),
        )


def _build_dataloader(
    dataset,
    config: DataConfig,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = True,
) -> DataLoader:
    """Construct a DataLoader with optional distributed sampling."""
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True,
        )
        shuffle = False  # sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )


def collate_fn(batch: list) -> DataBatch:
    """Collate a list of samples into a DataBatch.

    Expects each sample to be a dict with keys:
      - ``input_ids``  (list[int] or 1-D tensor)
      - ``labels``     (list[int] or 1-D tensor, optional — falls back to input_ids)
      - ``attention_mask`` (list[int] or 1-D tensor, optional)
    """
    input_ids = torch.stack(
        [torch.as_tensor(s["input_ids"], dtype=torch.long) for s in batch]
    )

    if "labels" in batch[0]:
        labels = torch.stack(
            [torch.as_tensor(s["labels"], dtype=torch.long) for s in batch]
        )
    else:
        # Default: shift input_ids by one position for causal LM training
        labels = input_ids.clone()

    attention_mask = None
    if "attention_mask" in batch[0]:
        attention_mask = torch.stack(
            [torch.as_tensor(s["attention_mask"], dtype=torch.long) for s in batch]
        )

    return DataBatch(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def infinite_loader(loader: DataLoader) -> Iterator[DataBatch]:
    """Yield batches from *loader* indefinitely, reshuffling each epoch."""
    epoch = 0
    while True:
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1
        logger.debug("DataLoader completed epoch %d, restarting.", epoch)


def get_data_path(config: DataConfig) -> str:
    """Resolve and validate the dataset path from *config*."""
    path = os.path.expandvars(os.path.expanduser(config.data_path))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset path does not exist: {path!r}. "
            "Set DataConfig.data_path to a valid directory or file."
        )
    return path
