import torch
import torch.nn as nn
from torch.distributions import Categorical


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
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var), nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var), nn.Tanh(),
            nn.Linear(n_latent_var, action_dim), nn.Softmax(dim=-1))

        # critic
        self.value_layer = nn.Sequential(nn.Linear(state_dim, n_latent_var),
                                         nn.Tanh(),
                                         nn.Linear(n_latent_var, n_latent_var),
                                         nn.Tanh(), nn.Linear(n_latent_var, 1))

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy
