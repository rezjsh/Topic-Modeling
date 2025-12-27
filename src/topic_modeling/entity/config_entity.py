from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    dataset_name: str
    test_size: float
    val_size: float
    random_state: int
    shuffle: bool
    arxiv_subset: str

@dataclass(frozen=True)
class DataTransformationConfig:
    max_features: int
    min_df: int
    max_df: float
    ngram_range: Tuple[int, int]
    mode: str  # 'aggressive' or 'contextual'
    batch_size: int
    root_dir: Path = "artifacts/data_transformation"
    bow_train_path: Path = f"{root_dir}/bow_train.npy"
    bow_val_path: Path = f"{root_dir}/bow_val.npy"
    bow_test_path: Path = f"{root_dir}/bow_test.npy"
    vocab_path: Path = f"{root_dir}/vocab.npy"
    id2word_path: Path = f"{root_dir}/id2word.npy"