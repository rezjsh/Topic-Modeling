from torch.utils.data import DataLoader
from topic_modeling.components.dataset import TopicModelingDataset
from topic_modeling.entity.config_entity import DataLoadingConfig
from topic_modeling.utils.helpers import get_num_workers

class DataLoading:
    """
    Component to manage the creation of DataLoaders using the TopicModelingDataset.
    """
    def __init__(self, config: DataLoadingConfig):
        self.config = config
        self.num_workers = get_num_workers()

    def get_loader(self, dataset: TopicModelingDataset, is_train: bool) -> DataLoader:
        """
        Utility function to create a PyTorch DataLoader.
        """
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle if is_train else False,
            num_workers=self.num_workers if self.num_workers > 0 else self.config.num_workers, # Set to >0 for faster loading
            pin_memory=self.config.pin_memory   # Recommended for larger datasets
        )