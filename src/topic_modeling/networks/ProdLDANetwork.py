import torch
import torch.nn as nn
import torch.nn.functional as F

from topic_modeling.entity.config_entity import ProdLDANetworkConfig

class ProdLDANetwork(nn.Module):
    def __init__(self, config: ProdLDANetworkConfig, vocab_size: int, num_topics: int):
        super().__init__()
        self.config = config

        # Inference Network (Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, self.config.hidden),
            nn.ReLU(),
            nn.Linear(self.config.hidden, self.config.hidden),
            nn.ReLU(),
            nn.Dropout(self.config.dropout)
        )
        self.fc_mu = nn.Linear(self.config.hidden, num_topics)
        self.fc_logvar = nn.Linear(self.config.hidden, num_topics)

        # Generative Network (Decoder)
        # Note: No bias is used here to ensure weights represent pure topic-word distributions
        self.decoder = nn.Linear(num_topics, vocab_size, bias=False)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False) # Critical for stability
        self.dropout_dec = nn.Dropout(self.config.dropout)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample z while remaining differentiable."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        # 1. Encode to latent space
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # 2. Sample from Logistic Normal
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=-1) # Document-topic distribution
        theta = self.dropout_dec(theta)

        # 3. Decode (Product of Experts)
        # ProdLDA uses log_softmax over the linear weights
        recon = F.log_softmax(self.bn(self.decoder(theta)), dim=-1)

        return recon, mu, logvar