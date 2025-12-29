import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Optional, Union, Dict

from topic_modeling.entity.config_entity import DatasetConfig

class TopicModelingDataset(Dataset):
    """
    A unified PyTorch Dataset for Topic Modeling.
    Supports BoW vectors for neural models and raw text for embedding models.
    """
    def __init__(
        self,
        config: DatasetConfig,
        bow_matrix: Union[np.ndarray, torch.Tensor],
        texts: List[str],
        labels: Optional[Union[np.ndarray, List[int], torch.Tensor]] = None
    ):
        self.config = config
        if isinstance(bow_matrix, np.ndarray):
            self.bow = torch.from_numpy(bow_matrix).float()
        else:
            self.bow = bow_matrix.float()

        self.texts = texts

        if labels is not None:
            if isinstance(labels, (np.ndarray, list)):
                self.labels = torch.tensor(labels, dtype=torch.long)
            else:
                self.labels = labels.clone().detach().to(dtype=torch.long)
        else:
            self.labels = torch.zeros(len(self.texts), dtype=torch.long)

        assert len(self.bow) == len(self.texts) == len(self.labels), \
            "All input arrays must have the same length."

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        return {
            "bow": self.bow[idx],
            "text": self.texts[idx],
            "label": self.labels[idx]
        }
