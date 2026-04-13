"""Configuration dataclasses for dflash training runs."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Configuration for the model architecture."""

    model_name_or_path: str = "gpt2"
    """HuggingFace model name or local path."""

    trust_remote_code: bool = False
    """Whether to trust remote code when loading model."""

    torch_dtype: str = "bfloat16"
    """Dtype to load model weights in. One of: float32, float16, bfloat16."""

    attn_implementation: Optional[str] = "flash_attention_2"
    """Attention implementation. Use 'flash_attention_2' for FlashAttention."""

    gradient_checkpointing: bool = True
    """Enable gradient checkpointing to save memory at the cost of speed."""


@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing."""

    dataset_name: str = "tiiuae/falcon-refinedweb"
    """HuggingFace dataset name or local path."""

    dataset_split: str = "train"
    """Dataset split to use for training."""

    text_column: str = "content"
    """Column name containing the text data."""

    max_seq_length: int = 2048
    """Maximum sequence length for tokenization."""

    num_workers: int = 4
    """Number of dataloader worker processes."""

    streaming: bool = True
    """Whether to stream the dataset instead of downloading it fully."""


@dataclass
class TrainConfig:
    """Configuration for the training loop."""

    output_dir: str = "./checkpoints"
    """Directory to save checkpoints and logs."""

    num_train_steps: int = 10_000
    """Total number of gradient update steps."""

    per_device_train_batch_size: int = 4
    """Batch size per GPU device."""

    gradient_accumulation_steps: int = 4
    """Number of steps to accumulate gradients before an optimizer step."""

    learning_rate: float = 3e-4
    """Peak learning rate for the optimizer."""

    weight_decay: float = 0.1
    """AdamW weight decay coefficient."""

    warmup_steps: int = 500
    """Number of linear warmup steps for the LR scheduler."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    logging_steps: int = 10
    """Log metrics every N steps."""

    save_steps: int = 500
    """Save a checkpoint every N steps."""

    seed: int = 42
    """Random seed for reproducibility."""

    bf16: bool = True
    """Use bfloat16 mixed precision training."""

    fsdp: bool = False
    """Enable Fully Sharded Data Parallel (FSDP) for multi-GPU training."""

    fsdp_sharding_strategy: str = "FULL_SHARD"
    """FSDP sharding strategy. One of: FULL_SHARD, SHARD_GRAD_OP, NO_SHARD."""


@dataclass
class DFlashConfig:
    """Top-level configuration for a dflash training run."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        # Ensure nested dataclasses are properly instantiated when
        # plain dicts are passed (e.g. from YAML/JSON deserialization).
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.train, dict):
            self.train = TrainConfig(**self.train)
