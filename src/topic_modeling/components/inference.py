import torch
import joblib
import numpy as np
from typing import List, Dict, Any, Union
from topic_modeling.components.model_factory import TopicModelFactory
from topic_modeling.utils.helpers import get_device
from topic_modeling.utils.logging_setup import logger
from topic_modeling.entity.config_entity import PredictionConfig, TopicModelFactoryConfig
from topic_modeling.components.data_transformation import DataTransformation


class TopicPredictor:
    """
    Handles the end-to-end inference logic for unseen user data.
    Decouples text cleaning, vectorization, and model scoring.
    """
    def __init__(self, config: PredictionConfig, factory_config: TopicModelFactoryConfig, transformation_component: DataTransformation):
        self.config = config
        self.factory_config = factory_config
        self.device = get_device()

        # 1. Load the Preprocessing Engine (already initialized with artifacts)
        self.transformer = transformation_component
        self.transformer.load_artifacts() # New method we added to your DataTransformation
        # Initialize Factory with loaded vocab
        self.factory = TopicModelFactory(
            config=self.factory_config, # This config must map to Factory requirements
            vocab=self.transformer.get_vocab()
        )
        self.model_wrapper = self._load_model()
        logger.info(f"🚀 TopicPredictor initialized for {self.factory_config.model_name} on {self.device}")

    def _load_model(self):
        """Loads the weights into the factory-generated model."""
        wrapper = self.factory.get_model()

        if self.factory_config.model_name in ['LDA', 'NMF', 'LSA', 'PLSA']:
            wrapper.model = joblib.load(self.config.classic_model_path)
        elif self.factory_config.model_name in ['NTM', 'PRODLDA']:
            wrapper.network.load_state_dict(torch.load(self.config.model_path))
        elif self.factory_config.model_name in ['BERTOPIC', 'TOP2VEC']:
            from bertopic import BERTopic
            from top2vec import Top2Vec

            wrapper.model = BERTopic.load(self.config.model_path) if self.factory_config.model_name == 'BERTOPIC' else Top2Vec.load(self.config.model_path)

        return wrapper

    def predict(self, raw_texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """The main entry point for unseen user data."""
        # 1. Preprocess (spaCy lemmatization)
        cleaned_texts = self.transformer.preprocess_corpus(raw_texts)

        # 2. Vectorize (BoW)
        bow = self.transformer.transform_to_bow(cleaned_texts)

        # 3. Predict
        if self.factory_config.model_name in ['BERTOPIC', 'TOP2VEC']:
            # Embedding models use the cleaned text directly
            distributions = self.model_wrapper.predict(cleaned_texts)
        else:
            # Classic/Neural use the BoW matrix
            distributions = self.model_wrapper.predict(bow)

        # 4. Format Results
        results = []
        for i, dist in enumerate(distributions):
            topic_id = int(np.argmax(dist))
            results.append({
                "text": raw_texts[i],
                "topic_id": topic_id,
                "confidence": float(dist[topic_id]),
                "keywords": self.model_wrapper.get_topics()[topic_id][:5]
            })
        return results