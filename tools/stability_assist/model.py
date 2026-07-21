from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from .spec import ACTION_SIZE, HIDDEN_SIZE, OBSERVATION_SIZE


class Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, ACTION_SIZE), nn.Tanh(),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        linear_layers = [module for module in self.layers if isinstance(module, nn.Linear)]
        for layer in linear_layers[:-1]:
            nn.init.orthogonal_(layer.weight, gain=2.0 ** 0.5)
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(linear_layers[-1].weight, gain=0.01)
        nn.init.zeros_(linear_layers[-1].bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation)


class ActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = Actor()
        self.critic = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1),
        )
        for module in self.critic:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        self.log_std = nn.Parameter(torch.full((ACTION_SIZE,), -1.2))

    def act(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor(observation)
        distribution = Normal(mean, self.log_std.exp().expand_as(mean))
        sampled_action = distribution.sample()
        log_probability = distribution.log_prob(sampled_action).sum(dim=-1)
        value = self.critic(observation).squeeze(-1)
        return sampled_action, log_probability, value

    def evaluate_actions(self, observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor(observation)
        distribution = Normal(mean, self.log_std.exp().expand_as(mean))
        log_probability = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        value = self.critic(observation).squeeze(-1)
        return log_probability, entropy, value
