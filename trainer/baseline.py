import numpy as np
import os
import torch
import torch.nn as nn
import gym
import matplotlib.pyplot as plt

import continuous
import utils


def train_baseline():
    ############## Hyperparameters ##############

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random_seed = 0
    data_dir = "./log/data/"
    model_dir = "./log/model/"
    model_name = "baseline-LunarLander-v2.pth"

    # env
    env_name = "LunarLanderContinuous-v2"
    env = gym.make(env_name)

    # data
    load_pretrained_model = False
    num_dist = 4

    # model
    state_dim = env.observation_space.shape[0]
    action_dim = 2
    e_dim = 2
    latent_dim = 64  # number of variables in hidden layer
    lr = 0.001
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor

    # train
    epoch = 50
    log_interval = 5
    batch_size = 256
    sample_size = 100000

    # figure
    fig_dir = "./log/figure/"
    #############################################

    print(device)

    if random_seed:
        torch.manual_seed(random_seed)

    aacn = continuous.AACN(state_dim, e_dim, action_dim, latent_dim)

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

        list_a = torch.from_numpy(list_a).float().to(device)
        list_s = torch.from_numpy(list_s).float().to(device)
        list_dist = torch.from_numpy(list_dist).float().to(device)

        y_real = torch.autograd.Variable(torch.ones(batch_size, 1)).to(device)
        y_fake = torch.autograd.Variable(torch.zeros(batch_size, 1)).to(device)

        print(np.shape(list_a))
        print(np.shape(list_s))
        print(np.shape(list_dist))

        g_optimizer = torch.optim.Adam(aacn.g_layer.parameters(),
                                       lr=lr,
                                       betas=betas)
        f_optimizer = torch.optim.Adam(aacn.f_layer.parameters(),
                                       lr=lr,
                                       betas=betas)
        # h_optimizer = torch.optim.Adam(aacn.h_layer.parameters(),
        #                                lr=lr,
        #                                betas=betas)

        def lr_scheduler(epoch):
            if epoch < 30:
                return 0.5
            elif epoch < 70:
                return 0.5**2
            elif epoch < 90:
                return 0.5**3
            else:
                return 0.5**4

        g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer,
                                                      step_size=30,
                                                      gamma=0.5)
        f_scheduler = torch.optim.lr_scheduler.StepLR(f_optimizer,
                                                      step_size=30,
                                                      gamma=0.5)
        # h_scheduler = torch.optim.lr_scheduler.StepLR(h_optimizer,
        #                                               step_size=30,
        #                                               gamma=0.5)

        criterion = nn.MSELoss()
        criterion_discriminator = nn.BCELoss()

        aacn.to(device)
        aacn.train()

        losses_gen = []
        # losses_dis = []

        # training loop
        for i_episode in range(1, epoch + 1):
            print(f"episode: {i_episode}")
            aacn.train()
            aacn.to(device)
            for i in range(sample_size // batch_size):

                x_batch = list_s[batch_size * i:batch_size * (i + 1)]
                y_batch = list_a[batch_size * i:batch_size * (i + 1)]
                d_batch = list_dist[batch_size * i:batch_size * (i + 1)]

                ### discriminator ###
                # g_optimizer.zero_grad()
                # f_optimizer.zero_grad()
                # h_optimizer.zero_grad()

                # d_real_outputs = aacn.h_layer(d_batch)

                # e_outputs = aacn.g_layer(x_batch)
                # d_fake_outputs = aacn.h_layer(e_outputs)

                # loss_dis_real = criterion_discriminator(d_real_outputs, y_real)
                # loss_dis_fake = criterion_discriminator(d_fake_outputs, y_fake)
                # loss_dis = loss_dis_real + loss_dis_fake

                # loss_dis.backward()

                # h_optimizer.step()

                # losses_dis.append(loss_dis.item())

                ### encoder, decoder ###
                g_optimizer.zero_grad()
                f_optimizer.zero_grad()
                # h_optimizer.zero_grad()

                e_outputs = aacn.g_layer(x_batch)
                a_outputs = aacn.f_layer(e_outputs)
                # d_fake_outputs = aacn.h_layer(e_outputs)

                loss_gen = criterion(a_outputs, y_batch)
                # loss_dis = criterion_discriminator(d_fake_outputs, y_real)
                # loss = loss_gen * 10 + loss_dis * 0.03
                loss = loss_gen

                loss.backward()

                g_optimizer.step()
                f_optimizer.step()

                losses_gen.append(loss.item())

            # logging
            if i_episode % log_interval == 0:
                aacn.eval()
                aacn.to("cpu")
                with torch.no_grad():
                    # save_fig(aacn,
                    #          "dist",
                    #          fig_dir + "process_dist/",
                    #          "dist_ep_" + str(i_episode) + ".png",
                    #          num_dist=num_dist,
                    #          batch_size=batch_size)
                    save_fig(aacn,
                             "grid",
                             fig_dir + "process_grid/",
                             "dist_ep_" + str(i_episode) + ".png",
                             num_dist=num_dist,
                             batch_size=batch_size)

                    if not os.path.exists(fig_dir + "loss/"):
                        os.makedirs(fig_dir + "loss/")

                    plt.plot(losses_gen, label="gen")
                    # plt.plot(losses_dis, label="dis")
                    plt.savefig(
                        os.path.join(fig_dir + "loss/",
                                     "losses_ep_" + str(i_episode) + ".png"))
                    plt.close()

            g_scheduler.step()
            f_scheduler.step()
            # h_scheduler.step()

        torch.save(
            aacn.to('cpu').state_dict(), os.path.join(model_dir, model_name))

        plt.plot(losses_gen, label="gen")
        # plt.plot(losses_dis, label="dis")
        plt.savefig(os.path.join(fig_dir, "losses" + ".png"))
        plt.close()

    else:
        aacn.load_state_dict(
            torch.load(os.path.join(model_dir, model_name),
                       map_location="cpu"))
        print("model loaded")

    save_fig(aacn, "grid", fig_dir, "result_grid_baseline.png", num_dist,
             batch_size)
    save_fig(aacn, "dist", fig_dir, "result_dist_baseline.png", num_dist,
             batch_size)
    test_result(aacn, env)


def save_fig(aacn,
             plot_type="grid",
             fig_dir="./log/figure/",
             fig_name="figure.png",
             num_dist=1000,
             batch_size=16):
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
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
            a_list.append(outputs)

    a_list = np.concatenate(a_list)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    colors = []
    for idx in range(num_plot_dense**2):
        action = a_list[idx]
        color = ((action[0] + 1.0) / 2.1, (action[1] + 1.0) / 2.1, 0.5)
        colors.append(color)
    ax.scatter(plot_list[:, 0], plot_list[:, 1], c=colors)
    plt.savefig(os.path.join(fig_dir, fig_name), format="png", dpi=100)
    plt.close()


def test_result(aacn, env):
    s_list = []
    obs_ = env.reset()
    batch_size = 128
    for count in range(1024):
        action = env.action_space.sample()
        obs, r, done, _ = env.step(action)
        if done:
            obs_ = env.reset()
            action = env.action_space.sample()
            obs, r, done, _ = env.step(action)
        s_list.append(np.concatenate([obs, obs_]))
        obs_ = obs
    s_list = np.array(s_list)
    s_list = torch.from_numpy(s_list).float()
    e_list = []
    for i in range(1024 // batch_size):
        x_batch = s_list[batch_size * i:batch_size * (i + 1)]
        outputs = aacn.g_layer(x_batch)
        outputs = outputs.data.cpu().numpy()
        e_list.append(outputs)
    e_list = np.concatenate(e_list)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    colors = []
    ax.scatter(e_list[:, 0], e_list[:, 1], marker=".")
    plt.savefig(os.path.join("./log/figure/", "test.png"),
                format="png",
                dpi=100)
    plt.close()
