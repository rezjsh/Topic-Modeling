from typing import List
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
import numpy as np
from topic_modeling.utils.logging_setup import logger
from topic_modeling.entity.config_entity import ClassicModelConfig

class ClassicModel:
    def __init__(self, config: ClassicModelConfig, model_type: str = None, top_n: int = 10, vocab: List[str] = None):
        self.config = config
        self.model_type = model_type.upper()
        self.top_n = top_n
        self.vocab = vocab
        self._init_model()
        logger.info(f"Initialized {self.model_type} with {config.n_components} components.")

    def _init_model(self):
        """Initializes the model based on the specified type."""
        if self.model_type == 'LDA':
            self.model = LatentDirichletAllocation(n_components=self.config.n_components,
            max_iter=self.config.max_iter,
            learning_method=self.config.learning_method,
            random_state=self.config.random_state)
        elif self.model_type == 'NMF':
            self.model = NMF(n_components=self.config.n_components,
                            random_state=self.config.random_state,
                            max_iter=self.config.max_iter,
                            init=self.config.init)
        elif self.model_type == 'LSA':
            self.model = TruncatedSVD(n_components=self.config.n_components,
                                    n_iter=self.config.max_iter,
                                    random_state=self.config.random_state)
        elif self.model_type == 'PLSA':
            # pLSA is equivalent to NMF with KL-divergence
            self.model = NMF(n_components=self.config.n_components,
                            beta_loss='kullback-leibler',
                            max_iter=self.config.max_iter,
                            init='random',
                            solver='mu',
                            random_state=self.config.random_state,
                            )
        else:
            raise ValueError(f"Model type {self.model_type} not recognized for ClassicModel.")

    def fit(self, bow_matrix: np.ndarray):
        """Fits the model to the bag-of-words matrix."""

        if bow_matrix.ndim != 2:
            raise ValueError("Input bow_matrix must be a 2D array.")
        logger.info(f"Fitting {self.model_type} on matrix of shape {bow_matrix.shape}...")
        self.model.fit(bow_matrix)
        logger.info(f"Model fitting completed.")

    def get_topics(self) -> List[List[str]]:
        """Extracts top N words for each topic."""
        # Ensure the model has been fitted
        if self.vocab is None:
            raise RuntimeError("Model must be fitted with vocab before calling get_topics.")
        topics = []
        for topic_idx, topic in enumerate(self.model.components_):
            top_features_ind = topic.argsort()[:-(self.top_n + 1):-1]
            topic_words = [self.vocab[i] for i in top_features_ind]
            topics.append(topic_words)
        return topics