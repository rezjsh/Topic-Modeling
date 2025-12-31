from typing import List
from topic_modeling.components.model_factory import TopicModelFactory
from topic_modeling.config.configuration import ConfigurationManager

class ModelFactoryPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config

    def run_pipeline(self, vocab: List[str]):
        """Runs the Model Factory pipeline to create a topic model instance."""
        model_factory_config = self.config.get_model_factory_config()
        model_factory = TopicModelFactory(
            config=model_factory_config,
            vocab=vocab
        )
        model = model_factory.get_model()
        return model