
from topic_modeling.components.data_ingestion import DataIngestion
from topic_modeling.config.configuration import ConfigurationManager
from topic_modeling.utils.logging_setup import logger

class DataIngestionPipeline:
    def __init__(self, config: ConfigurationManager) -> None:
        self.config = config

    def run_pipeline(self):
        ingestion_config = self.config.get_data_ingestion_config()
        data_ingestion = DataIngestion(ingestion_config)
        df_raw = data_ingestion.load_data()
        logger.info(f"Raw data shape: {df_raw.shape}")
        splits = data_ingestion.create_splits(df_raw)
        logger.info(f"Train split shape: {splits['train'].shape}, Val split shape: {splits['val'].shape}, Test split shape: {splits['test'].shape}")

        return df_raw, splits