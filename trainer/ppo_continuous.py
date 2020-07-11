import torch
import torch.nn as nn
import gym
import numpy as np
import os
import matplotlib.pyplot as plt

from continuous.models import Memory
from continuous.ppo import PPO


def train_ppo_continuous(trial=0, seed=0, save_npy=False):
    ############## Hyperparameters ##############

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random_seed = seed
    data_dir = "./log/data/"
    model_dir = "./log/model/"
    model_name = "aacn-LunarLander-v2-dist4.pth"
    fig_dir = "./log/figure/ppo/"

    # creating environment
    env_name = "LunarLanderContinuous-v2"
    env = gym.make(env_name)

    # model
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # train
    solved_reward = 220  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 10000  # max training episodes
    max_timesteps = 1500  # max timesteps in one episode

    update_timestep = 4000  # update policy every n timesteps
    action_std = 0.5  # constant std for action distribution (Multivariate Normal)
    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor
    lr = 0.0003  # parameters for Adam optimizer
    betas = (0.9, 0.999)
    #############################################

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs,
              eps_clip)
    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    episode_score_list = []
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_score = 0
        for t in range(max_timesteps):
            time_step += 1

            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            episode_score += reward
            if done:
                break

        episode_score_list.append(episode_score)
        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(
                ppo.policy.to('cpu').state_dict(),
                os.path.join(model_dir,
                             'PPO_continuous_{}.pth'.format(env_name)))
            break

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(
                i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.plot(episode_score_list)
    plt.savefig(os.path.join(fig_dir, "score_aacn.png"))
    plt.close()

    if save_npy:
        npy_dir = os.path.join(data_dir, "continuous")
        if not os.path.exists(npy_dir):
            os.makedirs(npy_dir)
        np.save(os.path.join(npy_dir, f"trial_{trial}.npy"),
                np.array(episode_score_list))