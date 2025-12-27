
import pandas as pd
import numpy as np
from topic_modeling.components.data_transformation import DataTransformation
from topic_modeling.config.configuration import ConfigurationManager
from topic_modeling.utils.logging_setup import logger


class DataTransformationPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config

    def run_pipeline(self, df: pd.DataFrame, splits: dict) -> dict:
        """Executes the data transformation pipeline."""
        data_transformation_config = self.config.get_data_transformation_config()
        data_transformation = DataTransformation(data_transformation_config)

        # Important: Create a separate column for cleaned text
        df['clean_text'] = data_transformation.preprocess_corpus(df['text'], mode="aggressive")
        # Fit the vectorizer on the training split only
        data_transformation.vectorizer.fit(splits['train']['clean_text'])

        # Transform the training split
        train_bow = data_transformation.transform_to_bow(splits['train']['clean_text'])
        # Transform the validation and test splits
        val_bow = data_transformation.transform_to_bow(splits['val']['clean_text'])
        test_bow = data_transformation.transform_to_bow(splits['test']['clean_text'])

        np.save(data_transformation_config.bow_train_path, train_bow)
        np.save(data_transformation_config.bow_val_path, val_bow)
        np.save(data_transformation_config.bow_test_path, test_bow)

        logger.info(f"Train BoW shape: {train_bow.shape}, Val BoW shape: {val_bow.shape}, Test BoW shape: {test_bow.shape}")

        return {
            'train_bow': train_bow,
            'val_bow': val_bow,
            'test_bow': test_bow,
            'vocab': data_transformation.get_vocab(),
            'id2word': data_transformation.get_id2word(),
            'clean_text': df['clean_text']
        }