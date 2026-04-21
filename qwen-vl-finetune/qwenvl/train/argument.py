import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    data_path: Optional[str] = field(default=None, metadata={"help": "Direct path to training JSON file"})
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = 2


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)

    ## Eval config
    eval_steps_custom: int = field(default=0, metadata={"help": "Run benchmark eval every N steps (0=disabled)"})
    eval_mode: str = field(default="subset", metadata={"help": "Eval mode: 'subset' or 'full'"})
    eval_subset_size: int = field(default=200, metadata={"help": "Number of samples per benchmark in subset mode"})
    eval_benchmarks: str = field(default="dream1k,carebench", metadata={"help": "Comma-separated benchmark names"})

    ## Wandb logging config
    sample_log_steps: int = field(default=500, metadata={"help": "Log mini eval predictions to wandb every N steps (0=disabled)"})
    sample_log_count: int = field(default=3, metadata={"help": "Number of samples per benchmark for mini eval logging"})
    sample_log_workers: int = field(default=4, metadata={"help": "Number of parallel workers for mini eval video decoding"})
