import torch
from topic_modeling.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from topic_modeling.core.singleton import SingletonMeta
from topic_modeling.utils.helpers import create_directory, get_num_workers, read_yaml_file
from topic_modeling.utils.logging_setup import logger
from topic_modeling.entity.config_entity import CallbacksConfig, ClassicModelConfig, DataEDAConfig, DataIngestionConfig, DataLoadingConfig, DataTransformationConfig, DatasetConfig, EarlyStoppingCallbackConfig, EmbeddingModelConfig, ModelCheckpointCallbackConfig, ModelEvaluationConfig, ModelLoggerConfig, NTMNetworkConfig, NeuralModelConfig, PredictionConfig, ProdLDANetworkConfig, TopicModelFactoryConfig, TopicTrainerConfig
from pathlib import Path

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
            vectorizer_path = config.vectorizer_path,
            max_features=params.max_features,
            min_df=params.min_df,
            max_df=params.max_df,
            ngram_range=config.ngram_range,
            mode=config.mode,
            batch_size=params.batch_size,
        )

        logger.info(f"DataTransformationConfig created: {data_transformation_config}")
        return data_transformation_config

    def get_data_eda_config(self) -> DataEDAConfig:
        config = self.config.data_eda
        params = self.params.data_eda
        logger.info(f"DataEDAConfig loaded: configs: {config} and params: {params}")
        dirs_to_create = [Path(config.root_dir)]
        create_directory(dirs_to_create)
        logger.info(f"Created directories for data eda artifacts: {dirs_to_create}")

        data_eda_config = DataEDAConfig(
            root_dir=Path(config.root_dir),
            text_col=config.text_col,
            label_col=config.label_col,
            top_k_ngrams=params.top_k_ngrams,
            wordcloud_width=config.wordcloud_width,
            wordcloud_height=config.wordcloud_height
        )

        logger.info(f"DataEDAConfig created: {data_eda_config}")
        return data_eda_config


    def get_dataset_config(self) -> DatasetConfig:
        # Currently empty, but can be extended in the future
        dataset_config = DatasetConfig()
        logger.info(f"DatasetConfig created: {dataset_config}")
        return dataset_config


    def get_data_loading_config(self) -> DataLoadingConfig:
        config = self.config.data_loading
        params = self.params.data_loading
        logger.info(f"DataLoadingConfig loaded: configs: {config} and params: {params}")
        dirs_to_create = [Path(config.root_dir)]
        create_directory(dirs_to_create)
        logger.info(f"Created directories for data loading artifacts: {dirs_to_create}")

        data_loading_config = DataLoadingConfig(
            root_dir=Path(config.root_dir),
            batch_size=params.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )

        logger.info(f"DataLoadingConfig created: {data_loading_config}")
        return data_loading_config


    def get_classic_model_config(self) -> ClassicModelConfig:
        config = self.config.classic_model
        params = self.params.classic_model
        logger.info(f"ClassicModelConfig loaded: configs: {config} and params: {params}")

        classic_model_config = ClassicModelConfig(
            n_components=params.n_components,
            random_state=config.random_state,
            max_iter=params.max_iter,
            learning_method=config.learning_method,
            init=config.init
        )

        logger.info(f"ClassicModelConfig created: {classic_model_config}")
        return classic_model_config


    def get_prod_lda_network_config(self) -> ProdLDANetworkConfig:
        config = self.config.prod_lda_network
        logger.info(f"ProdLDANetworkConfig loaded: {config}")

        prod_lda_network_config = ProdLDANetworkConfig(
            hidden=config.hidden,
            dropout=config.dropout
        )

        logger.info(f"ProdLDANetworkConfig created: {prod_lda_network_config}")
        return prod_lda_network_config

    def get_ntm_network_config(self) -> NTMNetworkConfig:
        config = self.config.ntm_network
        logger.info(f"NTMNetworkConfig loaded: {config}")

        ntm_network_config = NTMNetworkConfig(
            hidden=config.hidden
        )

        logger.info(f"NTMNetworkConfig created: {ntm_network_config}")
        return ntm_network_config

    def get_neural_model_config(self) -> NeuralModelConfig:
        params = self.params.neural_model
        logger.info(f"NeuralModelConfig loaded: params: {params}")

        neural_model_config = NeuralModelConfig(
            prod_lda_network_config=self.get_prod_lda_network_config(),
            ntm_network_config=self.get_ntm_network_config(),
            learning_rate=params.learning_rate,
            num_epochs=params.num_epochs
        )

        logger.info(f"NeuralModelConfig created: {neural_model_config}")
        return neural_model_config

    def get_embedding_model_config(self) -> EmbeddingModelConfig:
        config = self.config.embedding_model
        logger.info(f"EmbeddingModelConfig loaded: {config}")

        embedding_model_config = EmbeddingModelConfig(
            min_topic_size=config.min_topic_size,
            top_n_words=config.top_n_words,
            calculate_probabilities=config.calculate_probabilities,
            language=config.language,
            n_gram_range=tuple(config.n_gram_range), # Convert to tuple here
            low_memory=config.low_memory,
            speed=config.speed,
            workers=config.workers
        )

        logger.info(f"EmbeddingModelConfig created: {embedding_model_config}")
        return embedding_model_config

    def get_model_factory_config(self) -> TopicModelFactoryConfig:
        config = self.config.model_factory

        model_factory_config = TopicModelFactoryConfig(
            classic_model_config=self.get_classic_model_config(),
            neural_model_config=self.get_neural_model_config(),
            embedding_model_config=self.get_embedding_model_config(),
            model_name=config.model_name,
            num_topics=config.num_topics,
            top_n = config.top_n
        )
        logger.info(f"TopicModelFactoryConfig created: {model_factory_config}")
        return model_factory_config


    def get_early_stopping_config(self) -> EarlyStoppingCallbackConfig:
        config = self.config.early_stopping
        logger.info(f"EarlyStoppingConfig loaded: {config}")

        early_stopping_config = EarlyStoppingCallbackConfig(
            mode=config.mode,
            patience=config.patience,
            monitor=config.monitor,
        )

        logger.info(f"EarlyStoppingConfig created: {early_stopping_config}")
        return early_stopping_config


    def get_model_logger_config(self) -> ModelLoggerConfig:
        config = self.config.model_logger
        logger.info(f"ModelLoggerConfig loaded: {config}")

        dirs_to_create = [Path(config.log_dir)]
        create_directory(dirs_to_create)
        logger.info(f"Created directories for model logger artifacts: {dirs_to_create}")

        model_logger_config = ModelLoggerConfig(
            log_dir=Path(config.log_dir)
        )

        logger.info(f"ModelLoggerConfig created: {model_logger_config}")
        return model_logger_config


    def get_model_checkpoint_config(self) -> ModelCheckpointCallbackConfig:
        config = self.config.model_checkpoint
        logger.info(f"ModelCheckpointConfig loaded: {config}")
        dirs_to_create = [Path(config.save_path).parent]
        create_directory(dirs_to_create)
        logger.info(f"Created directories for model checkpoint artifacts: {dirs_to_create}")
        model_checkpoint_config = ModelCheckpointCallbackConfig(
            save_path=config.save_path,
            monitor=config.monitor,
            mode=config.mode
        )

        logger.info(f"ModelCheckpointConfig created: {model_checkpoint_config}")
        return model_checkpoint_config

    def get_callbacks_config(self) -> CallbacksConfig:
        callbacks_config = CallbacksConfig(
            early_stopping_callback_config=self.get_early_stopping_config(),
            model_logger_callback_config=self.get_model_logger_config(),
            model_checkpoint_callback_config=self.get_model_checkpoint_config()

        )

        logger.info(f"CallbacksConfig created: {callbacks_config}")
        return callbacks_config


    def get_model_trainer_config(self) -> TopicTrainerConfig:
        config = self.config.topic_trainer
        params = self.params.topic_trainer

        logger.info(f"TopicTrainerConfig loaded: {config}")

        # Create the root directory for the model trainer artifacts
        dirs_to_create = [Path(config.root_dir)]
        create_directory(dirs_to_create)
        logger.info(f"Created directories for model trainer artifacts: {dirs_to_create}")

        topic_trainer_config = TopicTrainerConfig(
            root_dir=Path(config.root_dir),
            epochs=params.epochs,
            use_amp=config.use_amp,

        )

        logger.info(f"TopicTrainerConfig created: {topic_trainer_config}")
        return topic_trainer_config


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        logger.info(f"ModelEvaluationConfig loaded: {config}")
        dirs_to_create = [Path(config.report_csv).parent, Path(config.top_words_json).parent, Path(config.root_dir)]
        create_directory(dirs_to_create)
        logger.info(f"Created directories for model evaluation artifacts: {dirs_to_create}")

        model_evaluation_config = ModelEvaluationConfig(
            report_csv=Path(config.report_csv),
            top_words_json=Path(config.top_words_json),
            root_dir=Path(config.root_dir)
        )

        logger.info(f"ModelEvaluationConfig created: {model_evaluation_config}")
        return model_evaluation_config



    def get_prediction_config(self) -> PredictionConfig:
        config = self.config.inference
        logger.info(f"PredictionConfig loaded: {config}")

        prediction_config = PredictionConfig(
            model_path=Path(config.model_path),
            classic_model_path=Path(config.classic_model_path)
        )
        logger.info(f"PredictionConfig created: {prediction_config}")
        return prediction_config