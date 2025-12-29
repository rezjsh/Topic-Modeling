import torch.nn as nn
import torch
import torch.nn.functional as F

from topic_modeling.entity.config_entity import NTMNetworkConfig

class NTMNetwork(nn.Module):
    def __init__(self, config: NTMNetworkConfig, vocab_size: int, num_topics: int):
        super().__init__()
        # Encoder is identical to ProdLDA
        self.config = config
        hidden = self.config.hidden
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus()
        )
        self.fc_mu = nn.Linear(hidden, num_topics)
        self.fc_logvar = nn.Linear(hidden, num_topics)

        # Decoder: Standard linear reconstruction
        self.decoder = nn.Linear(num_topics, vocab_size)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from a normal distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the NTM Network."""
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=-1)

        # Standard NTM uses a linear activation or sigmoid/softmax reconstruction
        recon = torch.sigmoid(self.decoder(theta))
        return recon, mu, logvar