import torch
import gym
import os
import numpy as np
import matplotlib.pyplot as plt

from discrete.models import Memory
from discrete.ppo import PPO
from continuous import AACN


def train_ppo_aacn(trial=0, seed=0, save_npy=False):
    ############## Hyperparameters ##############

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random_seed = seed
    data_dir = "./log/data/"
    model_dir = "./log/model/"
    model_name = "aacn-LunarLander-v2-dist4.pth"
    fig_dir = "./log/figure/ppo/"

    # env
    env_name = "LunarLanderContinuous-v2"
    env = gym.make(env_name)

    # model
    state_dim = env.observation_space.shape[0]
    action_dim = 2
    e_dim = 2
    latent_dim = 64  # number of variables in hidden layer
    dist_dim = 4
    n_latent_var = 64  # number of variables in hidden layer

    # train
    solved_reward = 250  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 2000  # max training episodes
    max_timesteps = 800  # max timesteps in one episode

    update_timestep = 2000  # update policy every n timesteps
    gamma = 0.99  # discount factor
    # K_epochs = 4  # update policy for K epochs
    K_epochs = 20  # update policy for K epochs
    eps_clip = 0.2
    lr = 0.001
    betas = (0.9, 0.999)  # clip parameter for PPO
    #############################################

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    aacn = AACN(state_dim, e_dim, action_dim, latent_dim)
    aacn.load_state_dict(
        torch.load(os.path.join(model_dir, model_name), map_location="cpu"))
    action_coordinates = np.array([[0.7, 0], [0, 0.7], [-0.7, 0], [0, -0.7]])
    action_coordinates = torch.from_numpy(action_coordinates).float()
    action_list = aacn.f_layer(action_coordinates)
    action_list = action_list.data.numpy()

    print("action_list")
    print(action_list)

    memory = Memory()
    ppo = PPO(state_dim, dist_dim, n_latent_var, lr, betas, gamma, K_epochs,
              eps_clip)
    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    episode_score_list = []
    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        episode_score = 0
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            action_prob = ppo.policy_old.act(state, memory)
            action = action_list[action_prob]
            state, reward, done, _ = env.step(action)

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

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
                os.path.join(model_dir, 'PPO_aacn_{}.pth'.format(env_name)))
            break

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(
                i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.plot(episode_score_list)
    plt.savefig(os.path.join(fig_dir, "score_aacn.png"))
    plt.close()

    if save_npy:
        npy_dir = os.path.join(data_dir, "aacn")
        if not os.path.exists(npy_dir):
            os.makedirs(npy_dir)
        np.save(os.path.join(npy_dir, f"trial_{trial}.npy"),
                np.array(episode_score_list))
