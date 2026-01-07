import json
from typing import Any, Dict
import torch
from topic_modeling.callbacks.BaseCallback import BaseCallback
from topic_modeling.entity.config_entity import ModelLoggerConfig
from topic_modeling.utils.logging_setup import logger

class ModelLogger(BaseCallback):
    def __init__(self, config: ModelLoggerConfig):
        self.config = config
        self.history = []

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        '''Logs metrics at the end of each epoch and saves them to a JSON file.'''
        # Convert any PyTorch tensors to standard Python scalars for JSON
        clean_logs = {
            k: (v.item() if isinstance(v, torch.Tensor) else v)
            for k, v in logs.items() if k != "model_state_dict"
        }
        self.history.append({"epoch": epoch, **clean_logs})

        with open(self.config.log_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Epoch {epoch} logs saved to history.json")
