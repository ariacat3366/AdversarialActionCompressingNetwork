import torch
import gym

import discrete
import utils


def train_aacn_discrete():
    ############## Hyperparameters ##############
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dim = 2
    n_latent_var = 64  # number of variables in hidden layer
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    epoch = 100
    log_interval = 1
    batch_size = 128
    random_seed = 0
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)

    # training loop
    for i_episode in range(1, epoch + 1):
        x = utils.multi_norm2d()
        for i in range(len(x) // batch_size):
            pass

        # logging
        if i_episode % log_interval == 0:
            pass


if __name__ == '__main__':
    train_aacn_discrete()