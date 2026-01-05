import json
import numpy as np
import torch
import time
from typing import List, Dict, Optional, Any
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
import joblib
from topic_modeling.entity.config_entity import TopicTrainerConfig
from topic_modeling.utils.helpers import get_device
from topic_modeling.utils.logging_setup import logger

class TopicTrainer:
    """
    Unified training controller designed for speed and reliability.
    Supports Neural (PyTorch), Classic (Sklearn), and Embedding models.
    """
    def __init__(
        self,
        config: TopicTrainerConfig,
        model: Any,
        callbacks: Optional[List[Any]] = None,
    ):
        self.config = config
        self.model = model
        self.callbacks = callbacks or []
        self.device = get_device()

        # Speed Optimization: Mixed Precision (only on CUDA)
        self.use_amp = self.config.use_amp and 'cuda' in str(self.device)
        self.scaler = GradScaler(enabled=self.use_amp)

        # Auto-move neural networks to target hardware
        if hasattr(self.model, "network"):
            self.model.network.to(self.device)
            logger.info(f"🚀 Neural Model moved to {self.device} (AMP: {self.use_amp})")

    def train(self, train_data: Any, val_data: Optional[Any] = None):
        """Dispatches to the correct training strategy."""
        for cb in self.callbacks:
            if hasattr(cb, 'on_train_begin'): cb.on_train_begin()

        start_time = time.time()

        # Strategy Dispatch
        if hasattr(self.model, "network"):
            self._run_neural_loop(train_data, val_data, self.config.epochs)
        else:
            self._run_classic_fit(train_data)

        duration = time.time() - start_time
        logger.info(f"✅ Training session completed in {duration:.2f}s")

        for cb in self.callbacks:
            if hasattr(cb, 'on_train_end'): cb.on_train_end()

        return self.model

    def _run_classic_fit(self, train_data):
        """Handles Scikit-learn or BERTopic one-shot fitting."""
        model_name = getattr(self.model, 'model_type', 'Classic Model')
        logger.info(f"⚡ Fitting {model_name}...")
        self.model.fit(train_data)

    def _run_neural_loop(self, train_loader: DataLoader, val_loader: Optional[DataLoader], epochs: int):
        """High-performance PyTorch loop with AMP and Callbacks."""
        pbar = tqdm(range(1, epochs + 1), desc="Topic Modeling Training")

        for epoch in pbar:
            epoch_logs = {"epoch": epoch}

            # 1. Training Phase
            train_metrics = self._train_one_epoch(train_loader)
            epoch_logs.update(train_metrics)

            # 2. Validation Phase
            if val_loader:
                val_metrics = self._evaluate(val_loader)
                epoch_logs.update(val_metrics)

            # 3. State Management for Checkpointing Callbacks
            epoch_logs["model_state_dict"] = self.model.network.state_dict()

            # 4. Trigger Callbacks (Early Stopping, Logging, Checkpointing)
            stop_training = False
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, epoch_logs)
                if getattr(cb, "stop_training", False):
                    stop_training = True

            # 5. UI Update
            display_metrics = {k: f"{v:.4f}" for k, v in epoch_logs.items()
                              if isinstance(v, (int, float))}
            pbar.set_postfix(display_metrics)

            if stop_training:
                logger.info(f"⏹️ Early stopping triggered at epoch {epoch}.")
                break

    def _train_one_epoch(self, loader: DataLoader) -> Dict[str, float]:
        ''' Trains the model for one epoch. '''
        self.model.network.train()
        total_loss = 0
        for batch in loader:
            x = batch['bow'].to(self.device, non_blocking=True)
            self.model.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                recon, mu, logvar = self.model.network(x)
                loss = self.model.calculate_loss(x, recon, mu, logvar)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.model.optimizer)
            self.scaler.update()
            total_loss += loss.item()

        return {"train_loss": total_loss / len(loader)}

    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        ''' Evaluates the model on validation data. '''
        self.model.network.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                x = batch['bow'].to(self.device, non_blocking=True)
                with autocast(enabled=self.use_amp):
                    recon, mu, logvar = self.model.network(x)
                    loss = self.model.calculate_loss(x, recon, mu, logvar)
                total_loss += loss.item()
        return {"val_loss": total_loss / len(loader)}


    def save_all_artifacts(self, trained_model, model_name, vocab):
        """
        Saves the model weights, vocabulary, and metadata to the artifacts directory.
        """
        # Get path from config (defined in config.yaml)
        model_dir = self.config.root_dir

        # 1. Save the Model Weights/Binary
        if hasattr(trained_model, "network"):
            # Neural Path (PyTorch)
            model_path = model_dir / "model.pt"
            torch.save(trained_model.network.state_dict(), model_path)
            logger.info(f"Neural weights saved at: {model_path}")
        else:
            # Classic/Contextual Path (Sklearn/Joblib)
            model_path = model_dir / "model.joblib"
            joblib.dump(trained_model.model, model_path) # Save the *internal* sklearn model
            logger.info(f"Classic model binary saved at: {model_path}")

        # 2. Save the Vocabulary (Mandatory for Inference)
        vocab_path = model_dir / "vocab.npy"
        np.save(vocab_path, vocab)
        logger.info(f"Vocabulary saved at: {vocab_path}")

        # 3. Save Training Metadata (for reproducibility)
        meta_path = model_dir / "metadata.json"
        metadata = {
            "model_name": model_name,
            "vocab_size": len(vocab)
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)