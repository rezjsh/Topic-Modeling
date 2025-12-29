from topic_modeling.components.callbacks import CallabcksManager
from topic_modeling.config.configuration import ConfigurationManager


class CallbacksPipeline:
    def __init__(self, config: ConfigurationManager) -> None:
        self.config = config

    def run_pipeline(self) -> list:
        """Run the callbacks pipeline to build and return the list of callbacks."""
        callbacks_config = self.config.get_callbacks_config()
        callbacks_manager = CallabcksManager(config=callbacks_config)
        return callbacks_manager