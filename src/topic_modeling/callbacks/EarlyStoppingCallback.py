from typing import Dict, Any
from topic_modeling.callbacks import BaseCallback
from topic_modeling.entity.config_entity import EarlyStoppingCallbackConfig
from topic_modeling.utils.logging_setup import logger

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, config: EarlyStoppingCallbackConfig):
        self.config = config
        self.best_score = None
        self.counter = 0
        self.stop_training = False

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        '''Check if training should be stopped early based on the monitored metric.'''
        score = logs.get(self.config.monitor)
        if score is None: return

        better = (self.best_score is None) or \
                 (score < self.best_score if self.config.mode == 'min' else score > self.best_score)

        if better:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.config.patience:
                self.stop_training = True
                logger.info(f"⏹️ Early stopping: {self.config.monitor} hasn't improved for {self.config.patience} epochs.")
