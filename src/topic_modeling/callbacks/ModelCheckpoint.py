from typing import Any, Dict
import torch
from topic_modeling.callbacks import BaseCallback
from topic_modeling.entity.config_entity import ModelCheckpointCallbackConfig
from topic_modeling.utils.logging_setup import logger

class ModelCheckpoint(BaseCallback):
    def __init__(self, config: ModelCheckpointCallbackConfig):
        self.config = config
        self.best_score = None

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        score = logs.get(self.config.monitor)
        if score is None: return

        better = (self.best_score is None) or \
                 (self.config.mode == 'min' and score < self.best_score) or \
                 (self.config.mode == 'max' and score > self.best_score)

        if better:
            self.best_score = score
            # Extract state directly from the model inside logs to avoid passing it every time
            state_dict = logs.get("model_state_dict")
            if state_dict:
                torch.save(state_dict, self.config.save_path)
                logger.info(f"💾 Checkpoint: {self.config.monitor} improved to {score:.4f}. Model saved.")