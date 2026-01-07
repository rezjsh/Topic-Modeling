from topic_modeling.callbacks.EarlyStoppingCallback import EarlyStoppingCallback
from topic_modeling.callbacks.ModelCheckpoint import ModelCheckpoint
from topic_modeling.callbacks.ModelLogger import ModelLogger
from topic_modeling.entity.config_entity import CallbacksConfig
from topic_modeling.utils.logging_setup import logger

class CallabcksManager:
    def __init__(self, config: CallbacksConfig) -> None:
        self.config = config
        self.callbacks = []

    def build_callbacks(self) -> list:
        """Build and return a list of callback instances based on the configuration."""
        if self.config.early_stopping_callback_config is not None:
            early_stopping_callback = EarlyStoppingCallback(config=self.config.early_stopping_callback_config)
            self.callbacks.append(early_stopping_callback)

        if self.config.model_logger_callback_config is not None:
            model_logger_callback = ModelLogger(config=self.config.model_logger_callback_config)
            self.callbacks.append(model_logger_callback)

        if self.config.model_checkpoint_callback_config is not None:
            model_checkpoint_callback = ModelCheckpoint(config=self.config.model_checkpoint_callback_config)
            self.callbacks.append(model_checkpoint_callback)

        logger.info(f"Total callbacks created: {len(self.callbacks)}")

        return self.callbacks
