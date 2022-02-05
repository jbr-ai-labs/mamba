import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import OneHotCategorical
from networks.transformer.layers import AttentionEncoder
from networks.dreamer.utils import build_model


class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, layers, activation=nn.ReLU):
        super().__init__()

        self.feedforward_model = build_model(in_dim, out_dim, layers, hidden_size, activation)

    def forward(self, state_features):
        x = self.feedforward_model(state_features)
        action_dist = OneHotCategorical(logits=x)
        action = action_dist.sample()
        return action, x


class AttentionActor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, layers, activation=nn.ReLU):
        super().__init__()
        self.feedforward_model = build_model(hidden_size, out_dim, 1, hidden_size, activation)
        self._attention_stack = AttentionEncoder(1, hidden_size, hidden_size)
        self.embed = nn.Linear(in_dim, hidden_size)

    def forward(self, state_features):
        n_agents = state_features.shape[-2]
        batch_size = state_features.shape[:-2]
        embeds = F.relu(self.embed(state_features))
        embeds = embeds.view(-1, n_agents, embeds.shape[-1])
        attn_embeds = F.relu(self._attention_stack(embeds).view(*batch_size, n_agents, embeds.shape[-1]))
        x = self.feedforward_model(attn_embeds)
        action_dist = OneHotCategorical(logits=x)
        action = action_dist.sample()
        return action, x
