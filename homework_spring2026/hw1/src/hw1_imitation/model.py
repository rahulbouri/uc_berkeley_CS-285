"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, action_dim * chunk_size))

        self.net = nn.Sequential(*layers)

        self.mse_loss = nn.MSELoss()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Args:
        state: (batch, state_dim)
        Returns:
        (batch, action_dim * chunk_size)
        """
        B = state.shape[0]
        return self.net(state).view(B, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        predicted_chunk = self.forward(state)
        return self.mse_loss(predicted_chunk, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self.forward(state)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []
        input_dim = state_dim + action_dim*chunk_size + 1

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, action_dim * chunk_size))

        self.net = nn.Sequential(*layers)

        self.mse_loss = nn.MSELoss()

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        B = state.shape[0]
        flat_dim = self.chunk_size * self.action_dim

        # x_1 = target actions (data), flattened
        x_1 = action_chunk.view(B, flat_dim)

        # x_0 = noise sample from prior
        x_0 = torch.randn_like(x_1)

        # t ~ Uniform(0, 1)
        t = torch.rand(B, 1, device=state.device)

        # Interpolate: x_t = (1 - t) * x_0 + t * x_1
        x_t = (1 - t) * x_0 + t * x_1

        # Target velocity: u = x_1 - x_0
        target_v = x_1 - x_0

        # Predict velocity
        net_input = torch.cat([state, x_t, t], dim=-1)
        predicted_v = self.net(net_input)

        return self.mse_loss(predicted_v, target_v)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        B = state.shape[0]
        flat_dim = self.chunk_size * self.action_dim
        dt = 1.0 / num_steps

        # Start from pure noise at t=0
        x_t = torch.randn(B, flat_dim, device=state.device)

        for k in range(num_steps):
            t = torch.full((B, 1), k / num_steps, device=state.device)
            net_input = torch.cat([state, x_t, t], dim=-1)
            v = self.net(net_input)
            x_t = x_t + v * dt

        return x_t.view(B, self.chunk_size, self.action_dim)

PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
