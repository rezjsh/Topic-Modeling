from typing import Any, List
from topic_modeling.components.data_loading import DataLoading
from topic_modeling.components.model_trainer import TopicTrainer
from topic_modeling.config.configuration import ConfigurationManager
from topic_modeling.utils.logging_setup import logger

class ModelTrainerPipeline:
    def __init__(self, config: ConfigurationManager, model: Any, callbacks: List = []):
        self.config = config
        self.model = model
        self.callbacks = callbacks

    def run_pipeline(self, transformation_output: dict, train_loader: DataLoading, val_loader: DataLoading) -> List:
        ''' Executes the model training pipeline and returns the discovered topics. '''
        trainer_config = self.config.get_model_trainer_config()
        model_factory_config = self.config.get_model_factory_config()

        # Normalize model_name to uppercase for consistent comparison
        model_name = model_factory_config.model_name.upper()

        trainer = TopicTrainer(config=trainer_config, model=self.model, callbacks=self.callbacks)

        vocab = transformation_output.get('vocab', None)
        if model_name in ['BERTOPIC', 'TOP2VEC']:
            logger.info("Training text-based model...")
            trained_model = trainer.train(transformation_output['train_clean_text'])
            trainer.save_all_artifacts(trained_model, model_name, vocab)

        elif model_name in ['NTM', 'PRODLDA']:
            logger.info("Training neural network-based model...")
            trained_model = trainer.train(train_loader, val_loader)
            trainer.save_all_artifacts(trained_model, model_name, vocab)

        else:
            logger.info("Training classic model...")
            trained_model = trainer.train(transformation_output['train_bow'], transformation_output['val_bow'])
            trainer.save_all_artifacts(trained_model, model_name, vocab)

        logger.info("Extracting topics from the trained model...")
        topics = trainer.model.get_topics()
        for i, topic in enumerate(topics):
            logger.info(f"Topic {i}: {topic}")
        return trained_model
