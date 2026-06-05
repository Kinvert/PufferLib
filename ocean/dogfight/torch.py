import torch
import torch.nn as nn


class DogfightEncoder(nn.Module):
    """Df36-style Dogfight encoder for the PyTorch diagnostic backend."""

    def __init__(self, obs_size, hidden_size=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.GELU(),
        )
        nn.init.orthogonal_(self.encoder[0].weight, 1.0)
        nn.init.constant_(self.encoder[0].bias, 0.0)

    def forward(self, observations):
        return self.encoder(observations.view(observations.shape[0], -1).float())


class DogfightDecoder(nn.Module):
    """Df36-style Dogfight decoder for the PyTorch diagnostic backend."""

    def __init__(
        self,
        nvec,
        hidden_size=128,
        action_init_scale=1.0,
        value_init_scale=1.0,
    ):
        super().__init__()
        self.nvec = tuple(nvec)
        self.is_continuous = sum(self.nvec) == len(self.nvec)

        if self.is_continuous:
            num_atns = len(self.nvec)
            self.decoder_mean = nn.Linear(hidden_size, num_atns)
            nn.init.orthogonal_(self.decoder_mean.weight, action_init_scale)
            nn.init.constant_(self.decoder_mean.bias, 0.0)
            self.decoder_logstd = nn.Parameter(torch.zeros(1, num_atns))
        else:
            self.decoder = nn.Linear(hidden_size, int(sum(self.nvec)))
            nn.init.orthogonal_(self.decoder.weight, action_init_scale)
            nn.init.constant_(self.decoder.bias, 0.0)

        self.value_function = nn.Linear(hidden_size, 1)
        nn.init.orthogonal_(self.value_function.weight, value_init_scale)
        nn.init.constant_(self.value_function.bias, 0.0)

    def forward(self, hidden):
        if self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean).clamp(min=-20, max=2)
            logits = torch.distributions.Normal(mean, torch.exp(logstd))
        else:
            logits = self.decoder(hidden)
            if len(self.nvec) > 1:
                logits = logits.split(self.nvec, dim=1)

        values = self.value_function(hidden)
        return logits, values
