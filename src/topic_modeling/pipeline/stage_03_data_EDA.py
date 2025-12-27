import pandas as pd
from topic_modeling.components.EDA import TopicEDA
from topic_modeling.config.configuration import ConfigurationManager
from topic_modeling.utils.logging_setup import logger

class TopicEDAPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config

    def run_pipeline(self, df: pd.DataFrame) -> None:
        logger.info("Starting EDA Pipeline")
        data_eda_config = self.config.get_data_eda_config()
        data_eda = TopicEDA(config=data_eda_config)
        data_eda.run_full_analysis(df)
        logger.info("EDA Pipeline completed successfully")