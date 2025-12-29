import torch
from typing import List, Dict, Any

from topic_modeling.entity.config_entity import TopicModelFactoryConfig
from topic_modeling.models import ClassicModel, EmbeddingModel, NeuralModel
from topic_modeling.utils.helpers import get_device

class TopicModelFactory:
    """
    The central factory for a 8-model Topic Modeling benchmark.
    Supported: 'LDA', 'NMF', 'LSA', 'pLSA', 'NTM', 'ProdLDA', 'BERTopic', 'Top2Vec'
    """

    def __init__(self, config: TopicModelFactoryConfig, vocab: List[str]):
        self.config = config
        self.vocab = vocab
        self.device = get_device()

    def get_model(self):
        """Returns a model instance based on the model name."""
        model_name = model_name.upper()

        # Group 1: Scikit-Learn based (Classic)
        if self.config.model_name in ['LDA', 'NMF', 'LSA', 'PLSA']:
            return ClassicModel(
                config=self.config.classic_model_config,
                model_type=self.config.model_name,
                top_n=self.config.top_n,
                vocab=self.vocab
            )

        # Group 2: PyTorch based (Neural)
        elif self.config.model_name in ['NTM', 'PRODLDA']:
            return NeuralModel(
                config=self.config.neural_model_config,
                model_type=self.config.model_name,
                num_topics=self.config.num_topics,
                top_n=self.config.top_n,
                vocab=self.vocab,
            )

        # Group 3: Embedding based (Modern)
        elif self.config.model_name in ['BERTOPIC', 'TOP2VEC']:
            return EmbeddingModel(
                config=self.config.embedding_model_config,
                model_type=self.config.model_name,
                num_topics=self.config.num_topics,
                top_n=self.config.top_n

            )

        else:
            raise ValueError(f"Model {self.config.model_name} not implemented in Factory.")
