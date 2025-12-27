import pandas as pd
from typing import Dict
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from datasets import load_dataset  # HuggingFace Datasets
from topic_modeling.entity.config_entity import DataIngestionConfig
from topic_modeling.utils.logging_setup import logger

class DataIngestion:
    """
    Handles fetching raw data and creating reproducible splits.
    Supports local scikit-learn datasets and remote HuggingFace datasets.
    """
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _load_20newsgroups(self) -> pd.DataFrame:
        """Fetches the 20 Newsgroups dataset from sklearn."""
        logger.info("Fetching 20 Newsgroups dataset...")
        data = fetch_20newsgroups(
            subset='all',
            remove=('headers', 'footers', 'quotes'),
            return_X_y=False
        )
        df = pd.DataFrame({
            'text': data.data,
            'label': data.target,
            'label_name': [data.target_names[i] for i in data.target]
        })
        # Remove empty documents often caused by cleaning headers/quotes
        df = df[df['text'].str.strip().astype(bool)]
        return df

    def _load_arxiv(self) -> pd.DataFrame:
        """Fetches ArXiv NLP dataset (Parquet-based, topic modeling ready)."""
        logger.info("Fetching ArXiv NLP dataset from HuggingFace...")
        ds = load_dataset("maartengr/arxiv_nlp", split="train")
        max_val = min(10000, len(ds))
        ds = ds.select(range(max_val))  # Your slice
        df = pd.DataFrame({
            'text': ds['Abstracts'],  # Or combine: ds['Titles'] + ' ' + ds['Abstracts']
            'label_name': [cat[0] if isinstance(cat, list) else cat for cat in ds['Categories']]
        })
        df['label'] = pd.Categorical(df['label_name']).codes
        return df


    def load_data(self) -> pd.DataFrame:
        """Entry point to load the requested dataset."""
        if self.config.dataset_name.lower() == "20newsgroups":
            return self._load_20newsgroups()
        elif self.config.dataset_name.lower() == "arxiv":
            return self._load_arxiv()
        else:
            raise ValueError(f"Dataset '{self.config.dataset_name}' not supported.")

    def create_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Creates Train, Validation, and Test splits.
        Calculates relative sizes to ensure final proportions match config.
        """
        logger.info(f"Splitting data into Train/Val/Test ({1-self.config.test_size-self.config.val_size:.2f}/{self.config.val_size}/{self.config.test_size})")

        # 1. Separate Test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            shuffle=self.config.shuffle
        )

        # 2. Separate Validation set from the remaining training data
        # Calculate relative val_size: (0.15 / (0.15 + 0.70)) = ~0.176
        relative_val_size = self.config.val_size / (1 - self.config.test_size)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=self.config.random_state,
            shuffle=self.config.shuffle
        )

        logger.info(f"Split results: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        return {
            "train": train_df.reset_index(drop=True),
            "val": val_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True)
        }
