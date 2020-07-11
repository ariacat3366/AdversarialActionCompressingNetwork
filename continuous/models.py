import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        # action mean range -1 to 1
        self.actor = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(),
                                   nn.Linear(64, 32), nn.Tanh(),
                                   nn.Linear(32, action_dim), nn.Tanh())
        # critic
        self.critic = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(),
                                    nn.Linear(64, 32), nn.Tanh(),
                                    nn.Linear(32, 1))
        self.action_var = torch.full((action_dim, ),
                                     action_std * action_std).to(self.device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class AACN(nn.Module):
    def __init__(self, state_dim, e_dim, action_dim, latent_dim):
        super(AACN, self).__init__()

        self.g_layer = nn.Sequential(nn.Linear(state_dim * 2, latent_dim),
                                     nn.Tanh(),
                                     nn.Linear(latent_dim, latent_dim),
                                     nn.Tanh(), nn.Linear(latent_dim, e_dim),
                                     nn.Tanh())

        self.f_layer = nn.Sequential(nn.Linear(e_dim, latent_dim), nn.Tanh(),
                                     nn.Linear(latent_dim, latent_dim),
                                     nn.Tanh(),
                                     nn.Linear(latent_dim, action_dim),
                                     nn.Tanh())

        self.h_layer = nn.Sequential(nn.Linear(e_dim, latent_dim), nn.Tanh(),
                                     nn.Linear(latent_dim, latent_dim),
                                     nn.Tanh(), nn.Linear(latent_dim, 1),
                                     nn.Sigmoid())

    def forward(self, inputs):
        e = self.g_layer(inputs)
        f_outputs = self.f_layer(e)
        h_outputs = self.h_layer(e)
        return f_outputs, h_outputs

    def forward_g(self, inputs):
        outputs = self.f_layer(inputs)
        return outputs

    def forward_f(self, e):
        outputs = self.f_layer(e)
        return outputs

    def forward_h(self, inputs):
        outputs = self.h_layer(inputs)
        return outputs