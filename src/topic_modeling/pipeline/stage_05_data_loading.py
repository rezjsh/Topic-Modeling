from topic_modeling.components.data_loading import DataLoading
from topic_modeling.config.configuration import ConfigurationManager
from topic_modeling.utils.logging_setup import logger

class DataLoadingPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config

    def run_pipeline(self, dataset: dict):
        loading_config = self.config.get_data_loading_config()
        loader_comp = DataLoading(loading_config)

        logger.info("Generating split-specific DataLoaders...")

        # Constructing Loaders for each split
        train_loader = loader_comp.get_loader(
            dataset=dataset['train'],
            is_train=True
        )
        val_loader = loader_comp.get_loader(
            dataset=dataset['val'],
            is_train=False
        )
        test_loader = loader_comp.get_loader(
            dataset=dataset['test'],
            is_train=False
        )

        return train_loader, val_loader, test_loader