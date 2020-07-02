import numpy as np
import torch
import torch.nn as nn

import gym

import discrete
import utils


def train_aacn_discrete():
    ############## Hyperparameters ##############

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random_seed = 0

    # env
    env_name = "LunarLander-v2"
    env = gym.make(env_name)

    # model
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    e_dim = 2
    latent_dim = 64  # number of variables in hidden layer
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor

    # trainint
    epoch = 100
    log_interval = 1
    batch_size = 128
    sample_size = 10000
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)

    aacn = discrete.AACN(state_dim, e_dim, action_dim, latent_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(aacn.parameters(), lr=0.001, momentum=0.9)

    list_s = np.array()
    list_a = np.array()

    # training loop
    for i_episode in range(1, epoch + 1):

        for i in range(len(list_s)):
            x_batch = list_s[batch_size * i:batch_size * (i + 1)]
            y_batch = list_a[batch_size * i:batch_size * (i + 1)]
            x_batch = torch.from_numpy(x_batch).to(device)
            y_batch = torch.from_numpy(x_batch).to(device)

            optimizer.zero_grad()
            outputs = aacn(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # logging
        if i_episode % log_interval == 0:
            pass
