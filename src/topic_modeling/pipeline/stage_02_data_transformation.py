import os
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

        # Preprocess text for each split
        logger.info("Preprocessing text for train, validation, and test splits...")
        train_clean_text = data_transformation.preprocess_corpus(splits['train']['text'].tolist())
        val_clean_text = data_transformation.preprocess_corpus(splits['val']['text'].tolist())
        test_clean_text = data_transformation.preprocess_corpus(splits['test']['text'].tolist())

        # Fit the vectorizer on the training split's cleaned text
        logger.info("Fitting CountVectorizer on training data (clean_text)...")
        data_transformation.fit(train_clean_text)

        # Transform the cleaned text of each split into BoW matrices
        train_bow = data_transformation.transform_to_bow(train_clean_text)
        val_bow = data_transformation.transform_to_bow(val_clean_text)
        test_bow = data_transformation.transform_to_bow(test_clean_text)

        # Save the BoW matrices
        np.save(data_transformation_config.bow_train_path, train_bow)
        np.save(data_transformation_config.bow_val_path, val_bow)
        np.save(data_transformation_config.bow_test_path, test_bow)

        logger.info(f"Train BoW shape: {train_bow.shape}, Val BoW shape: {val_bow.shape}, Test BoW shape: {test_bow.shape}")

        # Preprocess the entire original DataFrame's text for the 'clean_text' return value
        full_df_clean_text = data_transformation.preprocess_corpus(df['text'].tolist())

        return {
            'train_clean_text': train_clean_text,
            'val_clean_text': val_clean_text,
            'test_clean_text': test_clean_text,
            'train_bow': train_bow,
            'val_bow': val_bow,
            'test_bow': test_bow,
            'vocab': data_transformation.get_vocab(),
            'id2word': data_transformation.get_id2word(),
            'clean_text': full_df_clean_text # Return the preprocessed text for the entire dataset
        }
