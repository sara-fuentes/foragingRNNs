# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 22:20:23 2024

@author: saraf
"""
import torch.nn as nn
import torch
import ngym_foraging as ngym_f
from ngym_foraging.wrappers import pass_reward, pass_action
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import forage_training as ft
sys.path.append('C:/Users/saraf/anaconda3/Lib/site-packages')
sys.path.append('C:/Users/saraf')
# packages to save data
# packages to handle data
# packages to visualize data
# import gym and neurogym to create tasks
# from neurogym.utils import plotting
# import torch and neural network modules to build RNNs
# check if GPU is available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH = 'C:/Users/saraf/OneDrive/Documentos/IDIBAPS/foraging RNNs/nets/entire_net.pth'

# name of the task on the neurogym library
TASK = 'ForagingBlocks-v0'

TRAINING_KWARGS = {'dt': 100,
                   'lr': 1e-2,
                   'n_epochs': 20,
                   'batch_size': 16,
                   'seq_len': 200,
                   'TASK': TASK}

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seed=0):
        super(Net, self).__init__()

        self.hidden_size = hidden_size
        # build a recurrent neural network with a single
        # recurrent layer and rectified linear units
        # set seed for weights
        torch.manual_seed(seed)
        self.vanilla = nn.RNN(input_size, hidden_size, nonlinearity='relu')
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        # If hidden state is not provided, initialize it
        if hidden is None:
            hidden = torch.zeros(1, TRAINING_KWARGS['seq_len'],
                                 self.hidden_size)
        # get the output of the network for a given input
        out, _ = self.vanilla(x, hidden)
        x = self.linear(out)
        return x, out


# --- MAIN
if __name__ == '__main__':
    plt.close('all')
    # create folder to save data based on env seed
    # main_folder = 'C:/Users/saraf/OneDrive/Documentos/IDIBAPS/foraging RNNs/nets/'
    main_folder = '/home/molano/foragingRNNs_data/nets/'
    # Set up the task
    env_seed = 7
    num_periods = 2000
    w_factor = 0.00001
    mean_ITI = 200
    max_ITI = 400
    fix_dur = 100
    dec_dur = 100
    blk_dur = 50
    probs = np.array([0.1, 0.9])
    # create folder to save data based on parameters
    save_folder = (f"{main_folder}w{w_factor}_mITI{mean_ITI}_xITI{max_ITI}_f{fix_dur}_"
                   f"d{dec_dur}_n{np.round(num_periods/1e3, 1)}_nb{np.round(blk_dur/1e3, 1)}_"
                   f"prb{probs[0]}_seed{env_seed}")


    # get seeds from folders in save_folder
    seeds = [int(f) for f in os.listdir(save_folder) if os.path.isdir(save_folder + '/' + f)]
    # Set up the task
    env_kwargs = {'dt': TRAINING_KWARGS['dt'], 'probs': np.array([0, 1]),
                  'blk_dur': 20, 'timing':
                      {'ITI': ngym_f.random.TruncExp(mean_ITI, 100, max_ITI), # mean, min, max
                       'fixation': fix_dur, 'decision': dec_dur}}  # Decision period}

    # call function to sample
    env = gym.make(TASK, **env_kwargs)
    env = pass_reward.PassReward(env)
    env = pass_action.PassAction(env)
    # set seed
    env.seed(env_seed)
    env.reset()

    net_kwargs = {'hidden_size': 64,
                  'action_size': env.action_space.n,
                  'input_size': env.observation_space.shape[0]}
    
    TRAINING_KWARGS['env_kwargs'] = env_kwargs
    TRAINING_KWARGS['net_kwargs'] = net_kwargs
       
    num_steps_exp = 10000
    debug = False
    num_networks = len(seeds)

    # train several networks with different seeds
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    mean_perf_list = []
    for i_net in range(num_networks):
        seed = seeds[i_net]
        print('Seed: ', seed)
        # print net number and total number of nets
        print(f'Net {i_net+1}/{num_networks}')
        # load data
        save_folder_net = save_folder + '/' + str(seed)
        data_training = np.load(save_folder_net + '/data.npz', allow_pickle=True)
        # plot data
        # get mean performance from data
        mean_performance = data_training['mean_perf_list']
        # ax[0].plot(mean_performance, label='Net ' + str(net))
        # smooth mean performance
        roll = 20
        mean_performance_smooth = np.convolve(mean_performance, np.ones(roll)/roll, mode='valid')
        # check if mean performance is over a threshold at some point during training
        if np.max(mean_performance_smooth) > 0.7:
            ax[0].plot(mean_performance_smooth, label='Net ' + str(i_net) + ' smooth')
            ax[0].set_xlabel('Epochs')
            ax[0].set_ylabel('Mean performance')

        # load net
        net = Net(input_size=net_kwargs['input_size'],
                  hidden_size=net_kwargs['hidden_size'],
                  output_size=env.action_space.n)
        net = net.to(DEVICE)
        net = torch.load(save_folder_net + '/net.pth')
        # test net
        data = ft.run_agent_in_environment(num_steps_exp=num_steps_exp, env=env, net=net)
        perf = np.array(data['perf'])
        perf = perf[perf != -1]
        mean_perf = np.mean(perf)
        mean_perf_list.append(mean_perf)

        if i_net == 0:
            ft.plot_task(env_kwargs=env_kwargs, data=data, num_steps=100,
                         save_folder=save_folder_net)

    # histogram of mean performance
    ax[1].hist(mean_perf_list, bins=20)
    ax[1].set_xlabel('Mean performance')
    ax[1].set_ylabel('Frequency')
    # save figure
    f.savefig(save_folder + '/performance.png')
    plt.show()

