from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    data_dir: str = "data/mitdb"
    fs: int = 360
    lead_index: int = 0
    window_before: int = 128
    window_after: int = 128
    bandpass_low: float = 0.5
    bandpass_high: float = 40.0
    filter_order: int = 4


@dataclass
class EncoderConfig:
    kind: str = "delta"        # "delta" | "rate" | "none"
    delta_threshold: float = 0.1
    rate_num_steps: int = 256


@dataclass
class ModelConfig:
    kind: str = "snn"          # "snn" | "cnn" | "lstm"
    hidden: int = 128
    beta: float = 0.9
    n_classes: int = 5


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 1337
    device: str = "cpu"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config(
        data=DataConfig(**raw.get("data", {})),
        encoder=EncoderConfig(**raw.get("encoder", {})),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
    )
