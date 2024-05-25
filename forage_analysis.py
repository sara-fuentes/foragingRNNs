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
import itertools
import pickle

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


def GLM_regressors(df):
    # Prepare df columns
    # Converting the 'outcome' column to boolean values
    select_columns = ['reward', 'actions', 'iti']
    df_glm = df.loc[:, select_columns].copy()
    # subtract 2 from actions to get 0 for left and 1 for right
    df_glm['actions'] = df_glm['actions']-2
    df_glm['actions'][df_glm['actions']<0] = np.nan
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
    regressors = regr_plus + regr_minus[:-3]
    return df_glm, regressors


def plot_GLM(ax, GLM_df, alpha=1):
    orders = np.arange(len(GLM_df))

    # filter the DataFrame to separate the coefficients
    r_plus = GLM_df.loc[GLM_df.index.str.contains('r_plus'), "coefficient"]
    r_minus = GLM_df.loc[GLM_df.index.str.contains('r_minus'), "coefficient"]
    # intercept = GLM_df.loc['Intercept', "coefficient"]
    ax.plot(orders[:len(r_plus)], r_plus, marker='.', color='indianred', alpha=alpha)
    ax.plot(orders[:len(r_minus)], r_minus, marker='.', color='teal', alpha=alpha)

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


def general_analysis(load_folder, file, env, take_best, num_steps_exp=200000,
                     verbose=False, env_seed='123', plot_raster=False):
    # train several networks with different seeds
    net_nums = []
    mean_perf_list = []
    GLM_coeffs = pd.DataFrame()
    mean_perf_smooth_list = []
    mean_perf_iti = []
    df = pd.read_csv(file)
    # get list of net seeds from df
    net_seeds = df['net_seed'].unique()
    # net_seeds = net_seeds[4:]
    num_networks = len(net_seeds)
    for i_net, ns in enumerate(net_seeds):
        # if ns != 31076:
        #     continue
        print(f'Analyzing net {i_net+1}/{num_networks} with seed {ns}')
        # build file name envS_XX_netS_XX
        save_folder_net = load_folder+'/envS_'+env_seed+'_netS_'+str(ns)
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
        # TODO: check if load_net works without net as a parameter
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
        print(f'Net {i_net+1}/{num_networks} with seed {ns} has a'
              'meanperformance of {mean_perf}')

        if mean_perf > PERF_THRESHOLD:
            df = ft.dict2df(data)
            df_glm, regressors = GLM_regressors(df)
            try:
                # "variable" and "regressors" are columnames of dataframe
                # Apply glm
                mM_logit = smf.logit(formula='actions ~ ' + regressors,
                                    data=df_glm).fit()
                # save param in df
                GLM_df = pd.DataFrame({
                    'coefficient': mM_logit.params,
                    'std_err': mM_logit.bse,
                    'z_value': mM_logit.tvalues,
                    'p_value': mM_logit.pvalues,
                    'conf_Interval_Low': mM_logit.conf_int()[0],
                    'conf_Interval_High': mM_logit.conf_int()[1]
                })

                GLM_df['iti'] = -1
                # add column with network number
                GLM_df['seed'] = ns
                GLM_coeffs = pd.concat([GLM_coeffs, GLM_df], axis=0)
                if verbose:
                    f, ax_ind = plt.subplots(1, 1, figsize=(10, 6))
                    plot_GLM(ax=ax_ind, GLM_df=GLM_df)
                    f.savefig(save_folder_net + '/GLM_weights.png')
                    plt.close(f)
            except Exception as e:
                print(e)

            iti_mat = np.array(data['iti'])
            iti_list = np.unique(iti_mat)
            # create bins for iti: 1-2, 3-4, 5-7
            iti_bins = np.array([1, 3, 5, 7])
            # check that no iti is smaller than 1 and larger than 7
            assert np.all(iti_list >= 1) and np.all(iti_list <= 7)
            
            mean_performance = []
            if verbose:
                f, ax_ind = plt.subplots(1, 1, figsize=(10, 6))

            for iti in range(len(iti_bins)-1):
                # get itis in corresponding bin
                i_iti = np.logical_and(iti_mat >= iti_bins[iti],
                                       iti_mat < iti_bins[iti+1])
                mean_performance.append(np.mean(perf[i_iti]))
                # get df for iti
                df_glm_iti = df_glm.loc[i_iti]
                try:
                    # "variable" and "regressors" are columnames of dataframe
                    # Apply glm
                    mM_logit = smf.logit(formula='actions ~ ' + regressors,
                                        data=df_glm_iti).fit()
                    # save param in df
                    GLM_df = pd.DataFrame({
                        'coefficient': mM_logit.params,
                        'std_err': mM_logit.bse,
                        'z_value': mM_logit.tvalues,
                        'p_value': mM_logit.pvalues,
                        'conf_Interval_Low': mM_logit.conf_int()[0],
                        'conf_Interval_High': mM_logit.conf_int()[1]
                    })
                    GLM_df['iti'] = (iti_bins[iti]+iti_bins[iti+1])/2
                    # add column with network number
                    GLM_df['seed'] = ns
                    GLM_coeffs = pd.concat([GLM_coeffs, GLM_df], axis=0)
                    if verbose:
                        plot_GLM(ax=ax_ind, GLM_df=GLM_df, alpha=1/iti_bins[iti])
                except Exception as e:
                    print(e)
            if verbose:
                f.savefig(save_folder_net + '/GLM_weights_itis.png')
                plt.close(f)
            # TODO: check plot for this figure
            mean_perf_iti.append(mean_performance)
                      
            if plot_raster:
                raster_plot(df)
                ft.plot_task(env_kwargs=ENV_KWARGS, data=data, num_steps=300,
                             save_folder=save_folder_net)

    return net_seeds, mean_perf_list, mean_perf_smooth_list, \
        iti_bins, mean_perf_iti, GLM_coeffs, net_nums


def plot_general_analysis(mean_perf_smooth_list, GLM_coeffs, mean_perf,
                          iti_bins, mean_perf_iti, sv_folder,
                          take_best, seeds, dpi=300):
    # Constants
    PERF_THRESHOLD = 0.5  # Assuming a threshold value

    # Ensure the save folder exists
    if not os.path.exists(sv_folder):
        os.makedirs(sv_folder)

    # Check if training performance figure exists
    if not os.path.exists(sv_folder + '/training_performance.png'):
        f, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        # Plot mean performance
        for perf in mean_perf_smooth_list:
            plot_mean_perf(ax=ax, mean_performance_smooth=perf)
        ax.axhline(y=PERF_THRESHOLD, color='gray', linestyle='--')
        f.savefig(sv_folder + '/training_performance.png', dpi=dpi)
        f.savefig(sv_folder + '/training_performance.svg', dpi=dpi)
        plt.close(f)
    
    # Create the main figure with subplots
    f, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    ax = ax.flatten()
    
    # Plot histogram of mean performance
    plot_hist_mean_perf(ax=ax[0], perfs=mean_perf)
    # Save individual subplot
    fig0 = plt.figure(figsize=(4, 3))
    ax0 = fig0.add_subplot(111)
    plot_hist_mean_perf(ax=ax0, perfs=mean_perf)
    fig0.savefig(sv_folder + '/hist_mean_perf.png', dpi=dpi)
    fig0.savefig(sv_folder + '/hist_mean_perf.svg', dpi=dpi)
    plt.close(fig0)
    
    # Plot GLM and task behavior
    GLM_coeffs_all_itis = GLM_coeffs[GLM_coeffs['iti'] == -1]
    seeds = GLM_coeffs_all_itis['seed'].unique()
    for s in seeds:
        i = GLM_coeffs_all_itis['seed'] == s
        GLM_df = GLM_coeffs_all_itis.loc[i]
        if any(GLM_df['coefficient'] > 50):
            continue
        plot_GLM(ax=ax[1], GLM_df=GLM_df, alpha=0.5)
    ax[1].axhline(y=0, color='gray', linestyle='--')
    ax[1].set_ylabel('GLM weight')
    ax[1].set_xlabel('Previous trials')
    ax[1].legend(loc='best')
    # Save individual subplot
    fig1 = plt.figure(figsize=(4, 3))
    ax1 = fig1.add_subplot(111)
    for s in seeds:
        i = GLM_coeffs_all_itis['seed'] == s
        GLM_df = GLM_coeffs_all_itis.loc[i]
        if any(GLM_df['coefficient'] > 50):
            continue
        plot_GLM(ax=ax1, GLM_df=GLM_df, alpha=0.1)
    mean_df = GLM_coeffs_all_itis.groupby(GLM_coeffs_all_itis.index).mean()
    sem_df = GLM_coeffs_all_itis.groupby(GLM_coeffs_all_itis.index).sem()
    plot_GLM_means(ax=ax1, GLM_df=mean_df, sem_df=sem_df)

    ax1.axhline(y=0, color='gray', linestyle='--')
    ax1.set_ylabel('GLM weight')
    ax1.set_xlabel('Previous trials')
    ax1.legend(loc='best')
    fig1.savefig(sv_folder + '/GLM_task_behavior.png', dpi=dpi)
    fig1.savefig(sv_folder + '/GLM_task_behavior.svg', dpi=dpi)
    plt.close(fig1)

    # Plot GLM means for different ITIs
    
    filtered_GLM_coeffs = GLM_coeffs[GLM_coeffs['iti'] != -1]
    grouped_iti = filtered_GLM_coeffs.groupby('iti')
    counter = 0
    for iti, group_df in grouped_iti:
        mean_df = group_df.groupby(group_df.index).mean()
        sem_df = group_df.groupby(group_df.index).sem()
        plot_GLM_means(ax=ax[3], GLM_df=mean_df, sem_df=sem_df, alpha=1 /iti)
        counter += 1
    ax[3].legend(loc='best')
    # Save individual subplot
    fig, ax_iti = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharey=True)
    iti_counter = 0
    for iti, group_df in grouped_iti:
        for s in seeds:
            i = group_df['seed'] == s
            df_seed = group_df.loc[i]
            if any(df_seed['coefficient'] > 50):
                continue
            plot_GLM(ax=ax_iti[iti_counter], GLM_df=df_seed, alpha=0.2)
        mean_df = group_df.groupby(group_df.index).mean()
        sem_df = group_df.groupby(group_df.index).sem()
        plot_GLM_means(ax=ax_iti[iti_counter], GLM_df=mean_df, sem_df=sem_df)
        iti_counter += 1
    fig.savefig(sv_folder + '/GLM_means_ITI.png', dpi=dpi)
    fig.savefig(sv_folder + '/GLM_means_ITI.svg', dpi=dpi)
    plt.close(fig)

    # Plot normalized performance by ITI
    normalized_mean_perf_iti = [mp / mp[0] for mp in mean_perf_iti]
    for mp in normalized_mean_perf_iti:
        ax[2].plot(iti_bins, mp, color='lightgray')
    mp_arr = np.array(normalized_mean_perf_iti)
    ax[2].plot(iti_bins, np.mean(mp_arr, axis=0), linewidth=2, color='black', label='mean')
    ax[2].legend()
    ax[2].set_xlabel('ITI')
    ax[2].set_ylabel('Performance')
    # Save individual subplot
    fig2 = plt.figure(figsize=(4, 3))
    ax2 = fig2.add_subplot(111)
    for mp in normalized_mean_perf_iti:
        ax2.plot(iti_bins, mp, color='lightgray')
    mp_arr = np.array(normalized_mean_perf_iti)
    ax2.plot(iti_bins, np.mean(mp_arr, axis=0), linewidth=2, color='black', label='mean')
    ax2.legend()
    ax2.set_xlabel('ITI')
    ax2.set_ylabel('Performance')
    fig2.savefig(sv_folder + '/performance_by_ITI.png', dpi=dpi)
    fig2.savefig(sv_folder + '/performance_by_ITI.svg', dpi=dpi)
    plt.close(fig2)
    
    # Save the entire figure with all subplots
    f.savefig(sv_folder + '/performance_bests'+str(take_best)+'.png', dpi=dpi)
    f.savefig(sv_folder + '/performance_bests'+str(take_best)+'.svg', dpi=dpi)

    plt.show()

def test_networks(folder, file, env, take_best, verbose=False,
                  num_steps_tests=50000, env_seed='123'):

    mean_perf_all = []
    nets_seeds_all = []
    net_nums_all = []
    mean_perf_smooth_all = []
    iti_list_all = np.array([])
    mean_perf_iti_all = []
    GLM_coeffs_all = pd.DataFrame()
    df = pd.read_csv(folder+'/'+file)
    # get list of net seeds from df
    net_seeds = df['net_seed'].unique()
    for ns in net_seeds:
        # build file name envS_XX_netS_XX
        f = folder+'/envS_'+env_seed+'_netS_'+str(ns)
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
                              sv_folder=folder, take_best=take_best)
    # save data
    GLM_coeffs_all.to_csv(folder + '/GLM_coeffs.csv')
    data = {'mean_perf_all': mean_perf_all, 'nets_seeds_all': nets_seeds_all,
            'net_nums_all': net_nums_all, 'iti': iti_list_all,
            'mean_perf_iti_all': mean_perf_iti_all}
    np.savez(folder + '/data_analysis.npz', **data)
    return data
# TODO: create a function that tests the network in different environments


def plot_mean_perf_by_param(mperfs, param):
    # boxplot of mean performance by sequence length
    sns.set(style="whitegrid")
    # Create a line plot
    plt.figure(figsize=(10, 6))  # You can adjust the size of the figure
    sns.violinplot(data=mperfs, x=param, y='performance', cut=0)
    # Add a swarm plot
    sns.swarmplot(data=mperfs, x=param, y='performance', color='k', size=8, alpha=0.3)

    plt.title('Performance as Function of '+param)
    plt.xlabel(param)
    plt.ylabel('Average Performance')
    plt.legend(title='Net Seed', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.figure(figsize=(12, 6))  # You can adjust the size of the figure
    # Create histograms for each sequence length
    g = sns.FacetGrid(mperfs, col=param, col_wrap=4, height=3)
    g.map(sns.histplot, "performance")

    plt.subplots_adjust(top=0.9)
    # Show the plot
    plt.show()
    # Calculate basic statistics
    stats = mperfs.groupby(param)['performance'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    print(stats)


def get_mean_perf_by_param(param, main_folder, filename, param_mat, min_nsteps=300000, plot=True):
    
    df_path = os.path.join(main_folder, filename)
    df = pd.read_csv(df_path)
    
    # remove rows with sequence length not in param_mat
    filtered_df = df[df[param].isin(param_mat)]
    
    # remove rows with num_periods smaller than 5000
    filtered_df = filtered_df[filtered_df['num_periods']*filtered_df['seq_len'] >= min_nsteps]
    
    # add column called performance showing wether action was equal to gt
    filtered_df['performance'] = (filtered_df['actions'] == filtered_df['gt']).astype(int)

    # compute average grouping by param and net_seed using groupby
    grouped_df = filtered_df.groupby([param, 'net_seed'])

    mperfs = grouped_df['performance'].mean().reset_index()
    if plot:
        plot_mean_perf_by_param(mperfs=mperfs, param=param)


def plot_boxplots_by_param_comb(df, plot_column='performance',
                                plot_title=None):
    f, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='param_combination', y=plot_column, ax=ax)
    sns.stripplot(data=df, x='param_combination', y=plot_column, color='black', size=4, ax=ax)

    if plot_title:
        ax.set_title(plot_title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def get_mean_perf_by_param_comb(lr_mat, blk_dur_mat, seq_len_mat, main_folder, filename,
                           min_nsteps=300000, plot=True):
    df_path = os.path.join(main_folder, filename)
    df = pd.read_csv(df_path)

    # remove rows with lr not in lr_mat
    filtered_df = df[df['lr'].isin(lr_mat)]
    # remove rows with blk_dur not in blk_dur_mat
    filtered_df = filtered_df[filtered_df['blk_dur'].isin(blk_dur_mat)]
    # remove rows with seq_len not in seq_len_mat
    filtered_df = filtered_df[filtered_df['seq_len'].isin(seq_len_mat)]

    # remove rows with num_periods * seq_len smaller than min_nsteps
    filtered_df = filtered_df[filtered_df['num_periods']*filtered_df['seq_len'] >= min_nsteps]

    # add column called performance showing wether action was equal to gt
    filtered_df['performance'] = (filtered_df['actions'] == filtered_df['gt']).astype(int)

    # compute average grouping by lr, blk_dur and seq_len using groupby
    grouped_df = filtered_df.groupby(['lr', 'blk_dur', 'seq_len', 'net_seed'])

    mperfs = grouped_df['performance'].mean().reset_index()

    mperfs['param_combination'] = mperfs['lr'].astype(str) + '_' + mperfs['blk_dur'].astype(str) + '_' + mperfs['seq_len'].astype(str)


    if plot:
        plot_boxplots_by_param_comb(df=mperfs, plot_title='Performance by Parameter Combination')


def get_perf_by_param_comb_all_nets(lr_mat, blk_dur_mat, seq_len_mat, main_folder, filename,
                           min_nsteps=300000, plot=True):
    
    df_path = os.path.join(main_folder, filename)
    df = pd.read_csv(df_path)

    # remove rows with lr not in lr_mat
    filtered_df = df[df['lr'].isin(lr_mat)]
    # remove rows with blk_dur not in blk_dur_mat
    filtered_df = filtered_df[filtered_df['blk_dur'].isin(blk_dur_mat)]
    # remove rows with seq_len not in seq_len_mat
    filtered_df = filtered_df[filtered_df['seq_len'].isin(seq_len_mat)]

    # remove rows with num_periods * seq_len smaller than min_nsteps
    filtered_df = filtered_df[filtered_df['num_periods']*filtered_df['seq_len'] >= min_nsteps]

    # add column called performance showing wether action was equal to gt
    filtered_df['performance'] = (filtered_df['actions'] == filtered_df['gt']).astype(int)

    # compute average grouping by lr, blk_dur and seq_len using groupby
    grouped_df = filtered_df.groupby(['lr', 'blk_dur', 'seq_len'])

    mperfs = grouped_df['performance'].mean().reset_index()


    if plot:
        plot_perf_heatmaps(df=mperfs, blk_dur_mat=blk_dur_mat)
    

def plot_perf_heatmaps(df, blk_dur_mat):
    # Plot heatmap for each blk_dur
    f, ax = plt.subplots(1, len(blk_dur_mat), figsize=(15, 5))

    for i, blk_dur in enumerate(blk_dur_mat):
        df_blk_dur = df[df['blk_dur'] == blk_dur]
        pivot_df_blk_dur = df_blk_dur.pivot(index='lr', columns='seq_len', values='performance')
        sns.heatmap(pivot_df_blk_dur, annot=True, fmt=".2f", cmap='viridis', ax=ax[i])
        ax[i].set_title(f'Performance for blk_dur = {blk_dur}')
        ax[i].set_xlabel('Sequence Length')
        ax[i].set_ylabel('Learning Rate')

          # Annotate each cell with the performance value
        for j, lr in enumerate(pivot_df_blk_dur.index):
            for k, seq_len in enumerate(pivot_df_blk_dur.columns):
                perf = pivot_df_blk_dur.loc[lr, seq_len]
                ax[i].text(k + 0.5, j + 0.5, f'{perf:.2f}', ha='center', va='center', color='gray')
    plt.tight_layout()
    plt.show()

def bin_mean(iti_bins):
    # Reshape the array to form pairs of consecutive elements
    bin_pairs = np.stack((iti_bins[:-1], iti_bins[1:]), axis=1)
    # Calculate the mean along the second axis
    return np.mean(bin_pairs, axis=1)

# Define the function to plot GLM coefficients for every ITI
def plot_GLM_means(ax, GLM_df, sem_df, alpha=1):
    # Filter the DataFrame to separate the coefficients
    r_plus = GLM_df.loc[GLM_df.index.str.contains('r_plus'), "coefficient"]
    r_minus = GLM_df.loc[GLM_df.index.str.contains('r_minus'), "coefficient"]

     # Get the standard errors
    r_plus_err = sem_df.loc[sem_df.index.str.contains('r_plus'), "coefficient"]
    r_minus_err = sem_df.loc[sem_df.index.str.contains('r_minus'), "coefficient"]

    # Get the orders for plotting
    orders = np.arange(len(r_plus))

    # Plot the mean coefficients with error bars
    ax.errorbar(orders, r_plus, yerr= r_plus_err, marker='.', color='indianred', alpha=alpha, label='r+')
    ax.errorbar(orders, r_minus, yerr= r_minus_err, marker='.', color='teal', alpha=alpha, label='r-')

    # Create custom legend handles with labels and corresponding colors
    legend_handles = [
        mpatches.Patch(color='indianred', label='r+'),
        mpatches.Patch(color='teal', label='r-')
    ]

    # Add legend with custom handles
    ax.legend(handles=legend_handles)
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_ylabel('GLM weight')
    ax.set_xlabel('Previous trials')

def save_general_analysis_results(sv_folder, seeds, mean_perf_list, mean_perf_smooth_list, iti_bins, mean_perf_iti, GLM_coeffs, net_nums):
    with open(f'{sv_folder}/analysis_results.pkl', 'wb') as f:
        pickle.dump({
            'seeds': seeds,
            'mean_perf_list': mean_perf_list,
            'mean_perf_smooth_list': mean_perf_smooth_list,
            'iti_bins': iti_bins,
            'mean_perf_iti': mean_perf_iti,
            'GLM_coeffs': GLM_coeffs,
            'net_nums': net_nums
        }, f)

def load_analysis_results(folder):
    with open(f'{folder}/analysis_results.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['seeds'], data['mean_perf_list'], data['mean_perf_smooth_list'], data['iti_bins'], data['mean_perf_iti'], data['GLM_coeffs'], data['net_nums']

def raster_plot(df):

    # Compute probabilities
    df['time'] = range(len(df))

    # Probability of action = 3
    prob_action_3 = df['actions'].rolling(window=5).apply(lambda x: np.mean(x == 3), raw=True)

    # Probability of prob_r
    prob_reward_3 = df['prob_r']

    # Plotting
    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.plot(df['time'], prob_action_3, 'g-', label='Probability of Right')
    ax1.plot(df['time'], prob_reward_3, 'k-', label='Blocks')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Probability of Right')  # , color='g')

    ax1.legend(loc='upper left')

    # Plotting choices (actions = 2, 3) as ticks
    for i, row in df.iterrows():
        if row['actions'] in [2, 3]:
            if row['actions'] == row['gt']:
                ax1.plot(row['time'], row['actions']-2, 'kx', markersize=10)
            else:
                ax1.plot(row['time'], row['actions']-2+(-1)**(row['actions']==2)*0.2, 'kx', markersize=5)

    plt.title('Probability of Action 3 and Reward Probability over Time')
    plt.show()

# --- MAIN
if __name__ == '__main__':
# define parameters configuration
    PERF_THRESHOLD = 0.7
    env_seed = 123
    total_num_timesteps = 6000
    num_periods = 2000
    env_seed = 123
    num_periods = 40
    TRAINING_KWARGS['num_periods'] = num_periods
    # create folder to save data based on env seed
    main_folder = 'C:/Users/saraf/OneDrive/Documentos/IDIBAPS/foraging RNNs/nets/'
    # main_folder = '/home/molano/Dropbox/Molabo/foragingRNNs/' # '/home/molano/foragingRNNs_data/nets/'
   # main_folder = '/home/manuel/foragingRNNs/files/'
    # Set up the task
    w_factor = 0.01
    mean_ITI = 400
    max_ITI = 800
    fix_dur = 100
    dec_dur = 100
    blk_dur = 25
    probs = np.array([0.2, 0.8])
    ENV_KWARGS = {'dt': TRAINING_KWARGS['dt'], 'probs': probs,
                'blk_dur': blk_dur, 'timing':
                    {'ITI': ngym_f.random.TruncExp(mean_ITI, 100, max_ITI),
                        # mean, min, max
                        'fixation': fix_dur, 'decision': dec_dur},
                    # Decision period}
                    'rewards': {'abort': 0., 'fixation': 0., 'correct': 1.}}
    TRAINING_KWARGS['classes_weights'] =\
        torch.tensor([w_factor*TRAINING_KWARGS['dt']/(mean_ITI),
                    w_factor*TRAINING_KWARGS['dt']/fix_dur, 2, 2])
    # call function to sample
    env = gym.make(TASK, **ENV_KWARGS)
    env = pass_reward.PassReward(env)
    env = pass_action.PassAction(env)
    # set seed
    env.seed(env_seed)
    env.reset()
    NET_KWARGS = {'hidden_size': 128,
                'action_size': env.action_space.n,
                'input_size': env.observation_space.shape[0]}
    TRAINING_KWARGS['env_kwargs'] = ENV_KWARGS
    TRAINING_KWARGS['net_kwargs'] = NET_KWARGS
    # create folder to save data based on parameters
    folder = (f"{main_folder}w{w_factor}_mITI{mean_ITI}_xITI{max_ITI}_f{fix_dur}_"
                    f"d{dec_dur}_prb{probs[0]}")
    filename = main_folder+'/training_data_w1e-02.csv'
    redo = False
    # Check if analysis_results.pkl exists in the main folder
    if not os.path.exists(f'{folder}/analysis_results.pkl') or redo:
        seeds, mean_perf_list, mean_perf_smooth_list, \
                iti_bins, mean_perf_iti, GLM_coeffs, net_nums = \
        general_analysis(load_folder=folder, file=filename, env=env, take_best=True, num_steps_exp=100000,
                        verbose=True)
        # TODO: move inside general_analysis
        save_general_analysis_results(sv_folder=folder, seeds=seeds, mean_perf_list=mean_perf_list,
                                    mean_perf_smooth_list=mean_perf_smooth_list, iti_bins=iti_bins, 
                                    mean_perf_iti=mean_perf_iti, GLM_coeffs=GLM_coeffs, net_nums=net_nums)
    
    # Load the results
    seeds, mean_perf_list, mean_perf_smooth_list, iti_bins, mean_perf_iti, GLM_coeffs, net_nums = load_analysis_results(folder=folder)

    bin_mean_list = bin_mean(iti_bins)
    plot_general_analysis(mean_perf_smooth_list=mean_perf_smooth_list, GLM_coeffs=GLM_coeffs,
                          mean_perf=mean_perf_list, iti_bins=bin_mean_list, mean_perf_iti=mean_perf_iti,
                          sv_folder=folder, take_best=True, seeds=seeds)
    
    # # Save config as npz
    # np.savez(save_folder+'/config.npz', **TRAINING_KWARGS)

    # num_epochs = TRAINING_KWARGS['n_epochs']
    # num_steps_plot = 100
    # num_steps_test = 1000
    # debug = False
    # num_networks = 4
    # criterion = nn.CrossEntropyLoss(weight=TRAINING_KWARGS['classes_weights'])
    # train = False
    # # define parameter to explore
    # seq_len_mat = np.array([50, 300, 1000])
    # mperf_lists = get_mean_perf_by_param(main_folder, filename, seq_len_mat, w_factor, mean_ITI, max_ITI, fix_dur, dec_dur, blk_dur, probs)

    # plt.close('all')
    # take_best = True
    # num_steps_tests = 500
    # verbose = True
    # PERF_THRESHOLD = 0.7
    # # create folder to save data based on env seed
    # # main_folder = 'C:/Users/saraf/OneDrive/Documentos/IDIBAPS/foraging RNNs/nets/'
    # main_folder = '/home/molano/foragingRNNs_data/nets/'

    # # put the parameters in a dictionary
    # task_params = {'env_seed': 8, 'w_factor': 0.00001,
    #                'mean_ITI': 200, 'max_ITI': 400,
    #                'fix_dur': 100, 'dec_dur': 100, 'blk_dur': 50,
    #                'probs': np.array([0.1, 0.9])}
    # folder = (f"{main_folder}w{task_params['w_factor']}_mITI{task_params['mean_ITI']}"
    #           f"_xITI{task_params['max_ITI']}_f{task_params['fix_dur']}_"
    #           f"d{task_params['dec_dur']}_nb{np.round(task_params['blk_dur']/1e3, 1)}_"
    #           f"prb{task_params['probs'][0]}")

    # # create folder to save data based on parameters
    # ENV_KWARGS = {'dt': TRAINING_KWARGS['dt'], 'probs': task_params['probs'],
    #               'blk_dur': task_params['blk_dur'], 'timing':
    #                 {'ITI': ngym_f.random.TruncExp(task_params['mean_ITI'], 100, task_params['max_ITI']), # mean, min, max
    #                  'fixation': task_params['fix_dur'], 'decision': task_params['dec_dur']}} # Decision period
    # # change specific parameters to test the network in different environments
    # mean_ITI_test = [200, 400, 600, 800]
    # max_ITI_test = [400, 600, 800, 1000]
    # for mITI, mxITI in zip(mean_ITI_test, max_ITI_test):
    #     ENV_KWARGS['timing']['ITI'] = ngym_f.random.TruncExp(mITI, 100, mxITI)
    #     # create folder name to save test data
    #     save_folder = (f"{folder}/mITI{mITI}"
    #             f"_xITI{mxITI}_f{ENV_KWARGS['timing']['fixation']}_"
    #             f"d{ENV_KWARGS['timing']['decision']}_nb{np.round(ENV_KWARGS['blk_dur']/1e3, 1)}_"
    #             f"prb{ENV_KWARGS['probs'][0]}")
    #     # create save folder
    #     os.makedirs(save_folder, exist_ok=True)
    #     # Set up the task
    #     # call function to sample
    #     env = gym.make(TASK, **ENV_KWARGS)
    #     env = pass_reward.PassReward(env)
    #     env = pass_action.PassAction(env)
    #     # set seed
    #     env.seed(123)
    #     env.reset()

    #     NET_KWARGS = {'hidden_size': 64,
    #                 'action_size': env.action_space.n,
    #                 'input_size': env.observation_space.shape[0]}

    #     test_networks(folder=folder, env=env, take_best=take_best,verbose=verbose,
    #                   num_steps_tests=num_steps_tests, sv_folder=save_folder)
