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
import matplotlib.patches as mpatches
import numpy as np
import os
import sys
import forage_training as ft
import statsmodels.formula.api as smf
import pandas as pd
import seaborn as sns
import glob
# check if GPU is available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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


def GLM(df):
    # Prepare df columns
    # Converting the 'outcome' column to boolean values
    select_columns = ['reward', 'actions', 'iti']
    df_glm = df.loc[:, select_columns].copy()
    # subtract 2 from actions to get 0 for left and 1 for right
    df_glm['actions'] = df_glm['actions']-2

    # calculate correct_choice regressor L+
    # define conditions
    conditions = [
        (df_glm['reward'] == 0),
        (df_glm['reward'] == 1) & (df_glm['actions'] == 1),
        (df_glm['reward'] == 1) & (df_glm['actions'] == 0),
    ]
    r_plus = [0, 1, -1]
    df_glm['r_plus'] = np.select(conditions, r_plus, default='other')
    df_glm['r_plus'] = pd.to_numeric(df_glm['r_plus'], errors='coerce')

    # same as above but for L-
    # define conditions
    conditions = [
        (df_glm['reward'] == 1),
        (df_glm['reward'] == 0) & (df_glm['actions'] == 1),
        (df_glm['reward'] == 0) & (df_glm['actions'] == 0),
    ]
    r_minus = [0, 1, -1]
    df_glm['r_minus'] = np.select(conditions, r_minus, default='other')
    df_glm['r_minus'] = pd.to_numeric(df_glm['r_minus'], errors='coerce')

    # Creating columns for previous trial results (both dfs)
    max_shift = 10
    regr_plus = ''
    regr_minus = ''
    for i in range(1, max_shift):
        df_glm[f'r_plus_{i}'] = df_glm['r_plus'].shift(i)
        df_glm[f'r_minus_{i}'] = df_glm['r_minus'].shift(i)
        regr_plus += f'r_plus_{i} + '
        regr_minus += f'r_minus_{i} + '
    regr_minus = regr_minus[:-3]
    # "variable" and "regressors" are columnames of dataframe
    # Apply glm
    mM_logit = smf.logit(formula='actions ~ ' + regr_plus + regr_minus,
                         data=df_glm).fit()

    # prints the fitted GLM parameters (coefs), p-values and some other stuff
    results = mM_logit.summary()
    print(results)
    # save param in df
    GLM_df = pd.DataFrame({
        'coefficient': mM_logit.params,
        'std_err': mM_logit.bse,
        'z_value': mM_logit.tvalues,
        'p_value': mM_logit.pvalues,
        'conf_Interval_Low': mM_logit.conf_int()[0],
        'conf_Interval_High': mM_logit.conf_int()[1]
    })

    return GLM_df


def plot_GLM(ax, GLM_df):
    orders = np.arange(len(GLM_df))

    # filter the DataFrame to separate the coefficients
    r_plus = GLM_df.loc[GLM_df.index.str.contains('r_plus'), "coefficient"]
    r_minus = GLM_df.loc[GLM_df.index.str.contains('r_minus'), "coefficient"]
    # intercept = GLM_df.loc['Intercept', "coefficient"]
    ax.plot(orders[:len(r_plus)], r_plus, marker='o', color='indianred')
    ax.plot(orders[:len(r_minus)], r_minus, marker='o', color='teal')

    # Create custom legend handles with labels and corresponding colors
    legend_handles = [
        mpatches.Patch(color='indianred', label='r+'),
        mpatches.Patch(color='teal', label='r-')
    ]

    # Add legend with custom handles
    ax.legend(handles=legend_handles)
    # ax.axhline(y=intercept, label='Intercept', color='black')
    ax.axhline(y=0, color='gray', linestyle='--')

    ax.set_ylabel('GLM weight')
    ax.set_xlabel('Previous trials')


def load_net(save_folder, performance, take_best=True):
    # check if net.pth exists in the folder (for nets that have not been saved
    # several times during training)
    net_pth_path = os.path.join(save_folder, 'net.pth')
    if os.path.exists(net_pth_path):
        # If net.pth exists, load it directly
        net = torch.load(net_pth_path)
        network_number = 0
    else:
        # If net.pth doesn't exist, find the newest net,
        # which is the file with the highest number
        net_files = [f for f in os.listdir(save_folder) if 'net' in f]
        # find the number of the newest net file, being the file names net0,
        # net1, net2, etc.
        net_files = np.array([int(f.split('net')[1].split('.pth')[0]) for f in
                              net_files])
        if take_best:
            # find the best net based on performance
            best_net = np.argmax(performance)
            # find closest network in net_files
            index = np.argmin(np.abs(net_files - best_net))
            network_number = net_files[index]
        else:
            net_files.sort()
            network_number = net_files[-1]
        net_file = 'net'+str(network_number)+'.pth'
        net_path = os.path.join(save_folder, net_file)
        net = torch.load(net_path)
    return net, network_number


def plot_mean_perf(ax, mean_performance_smooth):
    ax.plot(mean_performance_smooth)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Mean performance')


def plot_hist_mean_perf(ax, perfs):
    ax.hist(perfs, bins=20)
    ax.set_xlabel('Mean performance')
    ax.set_ylabel('Frequency')


def general_analysis(load_folder, env, take_best, num_steps_exp=50000,
                     verbose=False):

    # get seeds from folders in load_folder
    seeds = [int(f) for f in os.listdir(load_folder) if
             os.path.isdir(load_folder + '/' + f)]

    num_networks = len(seeds)

    # train several networks with different seeds
    net_nums = []
    mean_perf_list = []
    GLM_coeffs = pd.DataFrame()
    mean_perf_smooth_list = []
    mean_perf_iti = []
    for i_net in range(num_networks):
        seed = seeds[i_net]
        # print('Seed: ', seed)
        # print(f'Net {i_net+1}/{num_networks}')
        # load data
        save_folder_net = load_folder + '/' + str(seed)
        data_training = np.load(save_folder_net + '/data.npz',
                                allow_pickle=True)
        # plot data
        # get mean performance from data
        mean_performance = data_training['mean_perf_list']
        # smooth mean performance
        roll = 20
        mean_performance_smooth = np.convolve(mean_performance,
                                              np.ones(roll)/roll, mode='valid')
        mean_perf_smooth_list.append(mean_performance_smooth)

        # load net
        net = Net(input_size=NET_KWARGS['input_size'],
                  hidden_size=NET_KWARGS['hidden_size'],
                  output_size=env.action_space.n)
        net = net.to(DEVICE)
        # load network
        net, network_number = load_net(save_folder=save_folder_net,
                                       performance=mean_performance_smooth,
                                       take_best=take_best)
        net_nums.append(network_number)
        # test net
        data = ft.run_agent_in_environment(num_steps_exp=num_steps_exp,
                                           env=env, net=net)
        perf = np.array(data['perf'])
        perf = perf[perf != -1]
        mean_perf = np.mean(perf)
        mean_perf_list.append(mean_perf)

        if mean_perf > PERF_THRESHOLD:
            print(f'Net {i_net+1}/{num_networks} with seed {seed} has a'
                  'meanperformance of {mean_perf}')
            iti_mat = np.array(data['iti'])
            iti_list = np.unique(iti_mat)
            mean_performance = []
            for iti in iti_list:
                mean_performance.append(np.mean(perf[iti_mat == iti]))
            mean_perf_iti.append(mean_performance)
            df = ft.dict2df(data)
            GLM_df = GLM(df)
            # add column with network number
            GLM_df['seed'] = seed
            GLM_coeffs = pd.concat([GLM_coeffs, GLM_df], axis=0)
            if verbose:
                ft.plot_task(env_kwargs=ENV_KWARGS, data=data, num_steps=100,
                             save_folder=save_folder_net)
                f, ax_ind = plt.subplots(1, 1, figsize=(10, 6))
                plot_GLM(ax=ax_ind, GLM_df=GLM_df)
                f.savefig(save_folder_net + '/GLM_weights.png')
                plt.close(f)

    return seeds, mean_perf_list, mean_perf_smooth_list, iti_list, \
        mean_perf_iti, GLM_coeffs, net_nums


def plot_general_analysis(mean_perf_smooth_list, GLM_coeffs, mean_perf,
                          iti_list, mean_perf_iti, sv_folder,
                          take_best, seeds):
    # check if figure exists
    if not os.path.exists(sv_folder + '/training_performance.png'):
        f, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        # plot mean performance
        for perf in mean_perf_smooth_list:
            plot_mean_perf(ax=ax, mean_performance_smooth=perf)
        ax.axhline(y=PERF_THRESHOLD, color='gray', linestyle='--')
        f.savefig(sv_folder + '/training_performance.png')
    
    f, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 5))
    ax = ax.flatten()
    plot_hist_mean_perf(ax=ax[0], perfs=mean_perf)
    # plot GLM and task behavior
    # get seeds from df
    seeds = GLM_coeffs['seed'].unique()
    for s in seeds:
        # get all rows with seed s
        i = GLM_coeffs['seed'] == s
        GLM_df = GLM_coeffs.loc[i]
        plot_GLM(ax=ax[1], GLM_df=GLM_df)

    ax[1].axhline(y=0, color='gray', linestyle='--')
    ax[1].set_ylabel('GLM weight')
    ax[1].set_xlabel('Previous trials')
    ax[1].legend(loc='best')
    for mp in mean_perf_iti:
        ax[2].plot(iti_list, mp, color='lightgray')
    mp_arr = np.array(mean_perf_iti)
    ax[2].plot(iti_list, [np.mean(mp_arr[:, 0]), np.mean(mp_arr[:, 1]),
                          np.mean(mp_arr[:, 2])],
               linewidth=3.5, color='black', label='mean')
    ax[2].legend()
    ax[2].set_xlabel('ITI')
    ax[2].set_ylabel('Performance')
    f.savefig(sv_folder + '/performance_bests'+str(take_best)+'.png')
    plt.show()


def test_networks(folder, env, take_best, sv_folder, verbose=False,
                  num_steps_tests=50000):

    mean_perf_all = []
    nets_seeds_all = []
    net_nums_all = []
    mean_perf_smooth_all = []
    iti_list_all = np.array([])
    mean_perf_iti_all = []
    GLM_coeffs_all = pd.DataFrame()
    files = glob.glob(folder+'/*')
    # get only files with n_pers in name
    files = [f for f in files if 'n_pers' in f]
    for f in files:
        # get num periods from folder name save_folder + 'n_pers_'+np.round
        # (num_periods/1e3, 1)+'k'
        num_periods = int(float(f.split('n_pers_')[1].split('k')[0])*1e3)
        # get seed
        seed = int(f.split('_s')[1][0])
        # find folder with start equal to save_folder
        seeds, mean_perf_list, mean_perf_smooth_list, iti_list, \
            mean_perf_iti, GLM_coeffs, net_nums = \
            general_analysis(load_folder=f, env=env, take_best=take_best,
                             num_steps_exp=num_steps_tests, verbose=verbose)
        mean_perf_all += mean_perf_list
        nets_seeds_all += seeds
        net_nums_all += net_nums
        mean_perf_smooth_all += mean_perf_smooth_list
        iti_list_all = np.concatenate((iti_list_all, iti_list))
        mean_perf_iti_all += mean_perf_iti
        GLM_coeffs_all = pd.concat([GLM_coeffs_all, GLM_coeffs], axis=0)
    iti_list_all = np.unique(iti_list_all)
    if verbose:
        plot_general_analysis(mean_perf_smooth_list=mean_perf_smooth_all,
                              GLM_coeffs=GLM_coeffs_all,
                              mean_perf=mean_perf_all,
                              iti_list=iti_list_all,
                              mean_perf_iti=mean_perf_iti_all,
                              seeds=nets_seeds_all,
                              sv_folder=sv_folder, take_best=take_best)
    # save data
    GLM_coeffs_all.to_csv(folder + '/GLM_coeffs.csv')
    data = {'mean_perf_all': mean_perf_all, 'nets_seeds_all': nets_seeds_all,
            'net_nums_all': net_nums_all, 'iti': iti_list_all,
            'mean_perf_iti_all': mean_perf_iti_all, 'folders': files}
    np.savez(sv_folder + '/data_analysis.npz', **data)
    return data
# TODO: create a function that tests the network in different environments

def plot_mean_perf_by_seq_len(mperf_lists, seq_len_list):
    # boxplot of mean performance by sequence length
    f, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.boxplot(data=mperf_lists, ax=ax)
    
    # Add dots for individual data points
    sns.stripplot(data=mperf_lists, color='k', size=5, ax=ax)
    
    ax.set_xticklabels(seq_len_list)
    ax.set_xlabel('Sequence length')
    ax.set_ylabel('Mean performance')
    ax.set_title('Boxplot of Mean Performance by Sequence Length')
    
    plt.show()


# --- MAIN
if __name__ == '__main__':
    plt.close('all')
    take_best = True
    num_steps_tests = 500
    verbose = True
    PERF_THRESHOLD = 0.7
    # create folder to save data based on env seed
    # main_folder = 'C:/Users/saraf/OneDrive/Documentos/IDIBAPS/foraging RNNs/nets/'
    main_folder = '/home/molano/foragingRNNs_data/nets/'

    # put the parameters in a dictionary
    task_params = {'env_seed': 8, 'w_factor': 0.00001,
                   'mean_ITI': 200, 'max_ITI': 400,
                   'fix_dur': 100, 'dec_dur': 100, 'blk_dur': 50,
                   'probs': np.array([0.1, 0.9])}
    folder = (f"{main_folder}w{task_params['w_factor']}_mITI{task_params['mean_ITI']}"
              f"_xITI{task_params['max_ITI']}_f{task_params['fix_dur']}_"
              f"d{task_params['dec_dur']}_nb{np.round(task_params['blk_dur']/1e3, 1)}_"
              f"prb{task_params['probs'][0]}")

    # create folder to save data based on parameters
    ENV_KWARGS = {'dt': TRAINING_KWARGS['dt'], 'probs': task_params['probs'],
                  'blk_dur': task_params['blk_dur'], 'timing':
                    {'ITI': ngym_f.random.TruncExp(task_params['mean_ITI'], 100, task_params['max_ITI']), # mean, min, max
                     'fixation': task_params['fix_dur'], 'decision': task_params['dec_dur']}} # Decision period
    # change specific parameters to test the network in different environments
    mean_ITI_test = [200, 400, 600, 800]
    max_ITI_test = [400, 600, 800, 1000]
    for mITI, mxITI in zip(mean_ITI_test, max_ITI_test):
        ENV_KWARGS['timing']['ITI'] = ngym_f.random.TruncExp(mITI, 100, mxITI)
        # create folder name to save test data
        save_folder = (f"{folder}/mITI{mITI}"
                f"_xITI{mxITI}_f{ENV_KWARGS['timing']['fixation']}_"
                f"d{ENV_KWARGS['timing']['decision']}_nb{np.round(ENV_KWARGS['blk_dur']/1e3, 1)}_"
                f"prb{ENV_KWARGS['probs'][0]}")
        # create save folder
        os.makedirs(save_folder, exist_ok=True)
        # Set up the task
        # call function to sample
        env = gym.make(TASK, **ENV_KWARGS)
        env = pass_reward.PassReward(env)
        env = pass_action.PassAction(env)
        # set seed
        env.seed(123)
        env.reset()

        NET_KWARGS = {'hidden_size': 64,
                    'action_size': env.action_space.n,
                    'input_size': env.observation_space.shape[0]}

        test_networks(folder=folder, env=env, take_best=take_best,verbose=verbose,
                      num_steps_tests=num_steps_tests, sv_folder=save_folder)
