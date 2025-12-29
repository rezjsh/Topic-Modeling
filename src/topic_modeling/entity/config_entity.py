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

@dataclass(frozen=True)
class DataEDAConfig:
    root_dir: Path
    text_col: str
    label_col: str
    top_k_ngrams: int
    wordcloud_width: int
    wordcloud_height: int

@dataclass(frozen=True)
class DatasetConfig:
    pass

@dataclass(frozen=True)
class DataLoadingConfig:
    root_dir: Path
    batch_size: int
    shuffle: bool
    num_workers: int
    pin_memory: bool


@dataclass(frozen=True)
class ClassicModelConfig:
    n_components: int
    random_state: int = 42
    max_iter: int = 10
    learning_method: str = 'online'  # Specific to LDA
    init: str = 'nndsvd'            # For NMF

@dataclass(frozen=True)
class ProdLDANetworkConfig:
    hidden: int = 256
    dropout: float = 0.2

@dataclass(frozen=True)
class NTMNetworkConfig:
    hidden: int = 256


@dataclass(frozen=True)
class NeuralModelConfig:
    prod_lda_network_config: ProdLDANetworkConfig
    ntm_network_config: NTMNetworkConfig
    learning_rate: float = 0.002
    num_epochs: int = 50


@dataclass(frozen=True)
class EmbeddingModelConfig:
    min_topic_size: int
    top_n_words: int
    calculate_probabilities: bool
    language: str
    n_gram_range: Tuple[int, int]
    low_memory: bool
    speed: int
    workers: int
    
@dataclass(frozen=True)
class TopicModelFactoryConfig:
    classic_model_config: ClassicModelConfig
    neural_model_config: NeuralModelConfig
    embedding_model_config: EmbeddingModelConfig
    model_name: str
    num_topics: int
    top_n: int
    
@dataclass(frozen=True)
class EarlyStoppingCallbackConfig:
    monitor: str = 'loss'
    mode: str = 'min'
    patience: int = 5

@dataclass(frozen=True)
class ModelLoggerConfig:
    log_dir: str = "logs"

@dataclass(frozen=True)
class ModelCheckpointCallbackConfig:
    save_path: str = "checkpoints/best_model.pt"
    monitor: str = "val_loss"
    mode: str = "min"

@dataclass(frozen=True)
class CallbacksConfig:
    early_stopping_callback_config: EarlyStoppingCallbackConfig
    model_logger_callback_config: ModelLoggerConfig
    model_checkpoint_callback_config: ModelCheckpointCallbackConfig
    
