from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from topic_modeling.entity.config_entity import NeuralModelConfig
from topic_modeling.networks.ProdLDANetwork import ProdLDANetwork
from topic_modeling.networks.NTMNetwork import NTMNetwork
from topic_modeling.utils.logging_setup import logger
from topic_modeling.utils.helpers import get_device

class NeuralModel:
    def __init__(self, config: NeuralModelConfig, model_type: str, num_topics: int, vocab: int):
        self.config = config
        self.model_type = model_type
        self.vocab_size = len(vocab)
        self.num_topics = num_topics
        self.device = get_device()
        self.vocab = vocab
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        self._init_model()
        logger.info(f"Initialized {self.model_type} Neural Model with {self.num_topics} topics on device {self.device}.")

    def _init_model(self):
        """Initializes the neural model based on the specified type."""
        if self.model_type == 'PRODLDA':
            self.network = ProdLDANetwork(config=self.config.prod_lda_network_config, vocab_size=self.vocab_size, num_topics=self.num_topics).to(self.device)
        elif self.model_type == 'NTM': # Basic NTM
            self.network = NTMNetwork(config=self.config.ntm_network_config,vocab_size=self.vocab_size, num_topics=self.num_topics).to(self.device)
        else:
            raise ValueError(f"Model type {self.model_type} not recognized for NeuralModel.")


    def fit(self, dataloader: DataLoader) -> None:
        """Trains the Neural Topic Model."""
        self.network.train()
        for epoch in range(self.config.num_epochs):
            total_loss = 0
            for batch in dataloader:
                bow = batch['bow'].to(self.device)
                self.optimizer.zero_grad()

                # Forward pass returns reconstruction and KL divergence components
                recon_bow, mu, logvar = self.network(bow)
                loss = self.calculate_loss(bow, recon_bow, mu, logvar)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

    def calculate_loss(self, bow: torch.Tensor, recon_bow: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Calculates the loss for Neural Topic Models."""
        # ELBO Loss: Reconstruction + KL Divergence
        # For ProdLDA, recon_bow is already log_softmax output (log-probabilities)
        recon_loss = -(bow * recon_bow).sum(1).mean() # Correct NLL with log-probabilities
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        logger.debug(f"Reconstruction Loss: {recon_loss.item()}, KL Divergence: {kld.item()}")
        return recon_loss + kld

    def get_topics(self) -> List[List[str]]:
        """Extracts top N words for each topic."""
        # Topics are extracted from the decoder weights (beta)
        beta = self.network.decoder.weight.data.cpu().numpy().T
        topics = []
        for i in range(self.num_topics):
            words_idx = np.argsort(beta[i])[:-(self.top_n+1):-1]
            topics.append([self.vocab[idx] for idx in words_idx])
        logger.info(f"Extracted topics: {topics}")
        return topics