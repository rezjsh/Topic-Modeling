from typing import List
from topic_modeling.components.data_transformation import DataTransformation
from topic_modeling.components.inference import TopicPredictor
from topic_modeling.config.configuration import ConfigurationManager
from topic_modeling.utils.logging_setup import logger
from topic_modeling.core.singleton import SingletonMeta


class InferencePipeline(metaclass=SingletonMeta):
    def __init__(self, config_manager: ConfigurationManager):
        self.config = config_manager
        self.pred_config = config_manager.get_prediction_config()
        self.trans_config = config_manager.get_data_transformation_config()
        self.factory_config = config_manager.get_model_factory_config()
    def run_pipeline(self, user_texts: List[str]) -> List[dict]:
        """
        Entry point for unseen data.
        """
        try:
            transformation = DataTransformation(config = self.trans_config)

            self.predictor = TopicPredictor(self.pred_config,self.factory_config,  transformation)

            predictions = self.predictor.predict(user_texts)
            return predictions
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise e