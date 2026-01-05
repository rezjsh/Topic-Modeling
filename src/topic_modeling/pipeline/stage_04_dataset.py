from topic_modeling.config.configuration import ConfigurationManager
from topic_modeling.components.dataset import TopicModelingDataset
from topic_modeling.utils.logging_setup import logger

class DatasetPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config

    def run_pipeline(self, transformation_output: dict, splits: dict) -> TopicModelingDataset:
        """
        Create and return a TopicModelingDataset instance.
        """

        dataset_config = self.config.get_dataset_config()
        train_ds = TopicModelingDataset(
            config=dataset_config,
            bow_matrix=transformation_output["train_bow"],
            texts=splits['train']['text'].tolist(),
            labels=splits['train']['label'].tolist()
        )
        val_ds = TopicModelingDataset(
            config=dataset_config,
            bow_matrix=transformation_output["val_bow"],
            texts=splits['val']['text'].tolist(),
            labels=splits['val']['label'].tolist()
        )
        test_ds = TopicModelingDataset(
            config=dataset_config,
            bow_matrix=transformation_output["test_bow"],
            texts=splits['test']['text'].tolist(),
            labels=splits['test']['label'].tolist()
        )
        dataset = {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds
        }
        logger.info("TopicModelingDataset instances created for train, val, and test sets.")

        return dataset