import numpy as np
from topic_modeling.components.model_evaluation import ModelEvaluation
from topic_modeling.config.configuration import ConfigurationManager


class ModelEvaluationPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config

    def run_pipeline(self, trained_model, vocab, test_clean_texts):
        
        model_evaluation_config = self.config.get_model_evaluation_config()
        model_config = self.config.get_model_factory_config()

        topics = trained_model.get_topics()

        evaluator = ModelEvaluation(model_evaluation_config, vocab, test_clean_texts)
        metrics = evaluator.calculate_metrics(topics, model_config.model_name)
        evaluator.save_results(metrics, topics)
        
        return metrics