import numpy as np
import os
import torch
import torch.nn as nn
import gym
import matplotlib.pyplot as plt

import discrete
import utils


def train_aacn_discrete():
    ############## Hyperparameters ##############

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random_seed = 0
    data_dir = "./log/data/"
    model_dir = "./log/model/"
    model_name = "aacn-LunarLander-v2-dist4.pth"

    # env
    env_name = "LunarLander-v2"
    env = gym.make(env_name)

    # data
    load_pretrained_model = True
    num_dist = 4

    # model
    state_dim = env.observation_space.shape[0]
    action_dim = 4
    e_dim = 2
    latent_dim = 64  # number of variables in hidden layer
    lr = 0.001
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor

    # train
    epoch = 100
    log_interval = 10
    batch_size = 256
    sample_size = 100000

    # figure
    save_fig = True
    fig_dir = "./log/figure/"
    fig_name = "result_dist_without_d.png"
    plot_type = "dist"  # ["grid", "dist"]
    #############################################

    print(device)

    if random_seed:
        torch.manual_seed(random_seed)

    aacn = discrete.AACN(state_dim, e_dim, action_dim, latent_dim)

    if not load_pretrained_model or not os.path.exists(
            os.path.join(model_dir, model_name)):

        if os.path.exists(os.path.join(data_dir, "list_a.npy")) and \
            os.path.exists(os.path.join(data_dir,"list_s.npy")):

            list_a = np.load(os.path.join(data_dir, 'list_a.npy'))
            list_s = np.load(os.path.join(data_dir, 'list_s.npy'))
            list_dist = utils.dist.multi_norm2d(num_dist=num_dist,
                                                num_sample=sample_size)

            print("npy data loaded")

        else:
            list_a = []
            list_s = []

            obs_ = env.reset()
            done_count = 0
            for count in range(sample_size):
                action = env.action_space.sample()
                obs, r, done, _ = env.step(action)
                if done:
                    obs_ = env.reset()
                    action = env.action_space.sample()
                    obs, r, done, _ = env.step(action)
                    done_count += 1
                list_s.append(np.concatenate([obs, obs_]))
                list_a.append(action)
                obs_ = obs
                count += 1
                if count % 10000 == 0:
                    print("count: " + str(count))
                    print("done_count: " + str(done_count))
                    done_count = 0

            list_a = np.array(list_a)
            list_s = np.array(list_s)
            list_dist = utils.dist.multi_norm2d(num_dist=num_dist,
                                                num_sample=sample_size)

            np.save(os.path.join(data_dir, 'list_a.npy'), list_a)
            np.save(os.path.join(data_dir, 'list_s.npy'), list_s)

            print("complete saving npy data")

        list_a = torch.from_numpy(list_a).long().to(device)
        list_s = torch.from_numpy(list_s).float().to(device)
        list_dist = torch.from_numpy(list_dist).float().to(device)

        g_optimizer = torch.optim.Adam(aacn.g_layer.parameters(),
                                       lr=lr,
                                       betas=betas)
        f_optimizer = torch.optim.Adam(aacn.f_layer.parameters(),
                                       lr=lr,
                                       betas=betas)
        h_optimizer = torch.optim.Adam(aacn.h_layer.parameters(),
                                       lr=lr,
                                       betas=betas)

        criterion = nn.CrossEntropyLoss()

        aacn.to(device)
        aacn.train()

        # training loop
        for i_episode in range(1, epoch + 1):
            for i in range(sample_size // batch_size):
                x_batch = list_s[batch_size * i:batch_size * (i + 1)]
                y_batch = list_a[batch_size * i:batch_size * (i + 1)]

                outputs, _ = aacn(x_batch)

                g_optimizer.zero_grad()
                f_optimizer.zero_grad()
                # h_optimizer.zero_grad()

                loss = criterion(outputs, y_batch)
                loss.backward()

                g_optimizer.step()
                f_optimizer.step()
                # h_optimizer.step()

            # logging
            if i_episode % log_interval == 0:
                with torch.no_grad():
                    num = 128
                    x_batch = list_s[300:300 + num]
                    y_batch = list_a[300:300 + num]
                    result, _ = aacn(x_batch)
                    result_idx = np.argmax(result.data.cpu().numpy(), axis=1)
                    y = y_batch.data.cpu().numpy()
                    sum = 0
                    for j in range(num):
                        if result_idx[j] == y[j]:
                            sum += 1
                    print(str(sum * 100 // num) + "%")

        torch.save(
            aacn.to('cpu').state_dict(), os.path.join(model_dir, model_name))

    else:
        # aacn.load_state_dict(
        #     torch.load(os.path.join(model_dir, model_name),
        #                map_location=device))
        aacn.load_state_dict(
            torch.load(os.path.join(model_dir, model_name),
                       map_location="cpu"))
        print("model loaded")

    if save_fig:
        aacn.eval()
        num_plot_dense = 48
        if plot_type == "grid":
            grid_list = [(i, j) for j in np.linspace(-1., 1., num_plot_dense)
                         for i in np.linspace(-1., 1., num_plot_dense)]
            grid_list = np.array(grid_list)
            plot_list = torch.from_numpy(grid_list).float()
        else:
            dist_list = utils.dist.multi_norm2d(num_dist=num_dist,
                                                num_sample=num_plot_dense**2)
            plot_list = torch.from_numpy(dist_list).float()

        a_list = []
        for i in range(num_plot_dense**2 // batch_size):
            with torch.no_grad():
                x_batch = plot_list[batch_size * i:batch_size * (i + 1)]
                outputs = aacn.forward_f(x_batch)
                outputs = outputs.data.cpu().numpy()
                outputs = np.argmax(outputs, axis=1)
                a_list.append(outputs)

        a_list = np.concatenate(a_list)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        for idx in range(num_plot_dense**2):
            action = a_list[idx]
            if action == 0:
                color = "red"
            if action == 1:
                color = "yellow"
            if action == 2:
                color = "green"
            if action == 3:
                color = "blue"
            ax.scatter(plot_list[idx, 0], plot_list[idx, 1], c=color)
        plt.savefig(os.path.join(fig_dir, fig_name), format="png", dpi=100)
