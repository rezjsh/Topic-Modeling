from topic_modeling.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from topic_modeling.core.singleton import SingletonMeta
from topic_modeling.utils.helpers import create_directory, get_num_workers, read_yaml_file
from topic_modeling.utils.logging_setup import logger
from topic_modeling.entity.config_entity import DataIngestionConfig, DataTransformationConfig

class ConfigurationManager(metaclass=SingletonMeta):
    def __init__(self, config_file_path: str = CONFIG_FILE_PATH, params_file_path: str = PARAMS_FILE_PATH):
        self.config = read_yaml_file(config_file_path)
        self.params = read_yaml_file(params_file_path)


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        logger.info(f"DataIngestionConfig loaded: {config}")

        dataset_config = DataIngestionConfig(
            dataset_name=config.dataset_name,
            test_size=config.test_size,
            val_size=config.val_size,
            random_state=config.random_state,
            shuffle=config.shuffle,
            arxiv_subset=config.arxiv_subset
        )

        logger.info(f"DatasetConfig created: {dataset_config}")
        return dataset_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.data_transformation
        logger.info(f"DataTransformationConfig loaded: configs: {config} and params: {params}")
        dirs_to_create = [config.root_dir]
        create_directory(dirs_to_create)
        logger.info(f"Created directories for data transformation artifacts: {dirs_to_create}")

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            vocab_path = config.vocab_path,
            id2word_path = config.id2word_path,
            bow_train_path = config.bow_train_path,
            bow_val_path = config.bow_val_path,
            bow_test_path = config.bow_test_path,
            max_features=params.max_features,
            min_df=params.min_df,
            max_df=params.max_df,
            ngram_range=config.ngram_range,
            mode=config.mode,
            batch_size=params.batch_size,
        )

        logger.info(f"DataTransformationConfig created: {data_transformation_config}")
        return data_transformation_config
        
    def get_data_eda_config(self):
        config = self.config.data_eda
        params = self.params.data_eda
        logger.info(f"DataEDAConfig loaded: configs: {config} and params: {params}")
        dirs_to_create = [config.root_dir]
        create_directory(dirs_to_create)
        logger.info(f"Created directories for data eda artifacts: {dirs_to_create}")

        data_eda_config = {
            "root_dir": config.root_dir,
            "text_col": config.text_col,
            "label_col": config.label_col,
            "top_k_ngrams": params.top_k_ngrams,
            "wordcloud_width": config.wordcloud_width,
            "wordcloud_height": config.wordcloud_height
        }

        logger.info(f"DataEDAConfig created: {data_eda_config}")
        return data_eda_config