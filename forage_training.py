"""
Created on Thu Feb  8 22:20:23 2024

@author: saraf
"""

import torch.nn as nn
import torch
import gymnasium as gym
import neurogym as ngym
from neurogym.wrappers import pass_reward, pass_action, side_bias
import forage_analysis as fa

import matplotlib.pyplot as plt
from scipy.special import erf
import pandas as pd
import numpy as np
import os
import glob
import tkinter as tk
# from tkinter import simpledialog


# check if GPU is available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# name of the task on the neurogym library
TASK = 'Foraging-v0'
# TASK = 'ForagingBlocks-v0' 
# TASK = 'PerceptualDecisionMaking-v0'
TRAINING_KWARGS = {'dt': 100,
                   'lr': 1e-2,
                   'seq_len': 300,
                   'TASK': TASK}

def create_env(env_seed, mean_ITI, max_ITI, fix_dur, dec_dur,
               blk_dur, probs):
    """
    Create an environment with the specified parameters.
    """
    if TASK == 'Foraging-v0':
        env_kwargs = {'dt': TRAINING_KWARGS['dt'], 'timing':
                        {'ITI': ngym.ngym_random.TruncExp(mean_ITI, 100, max_ITI),
                            # mean, min, max
                            'fixation': fix_dur, 'decision': dec_dur},
                        # Decision period}
                        'rewards': {'abort': 0., 'fixation': 0., 'correct': 1.}}
        # call function to sample
        env = gym.make(TASK, **env_kwargs)
        env = pass_reward.PassReward(env)
        env = pass_action.PassAction(env)
        env = side_bias.SideBias(env, probs=probs, block_dur=blk_dur)
    elif TASK == 'ForagingBlocks-v0':
        env_kwargs = {'dt': TRAINING_KWARGS['dt'], 'probs': probs[0],
                        'blk_dur': blk_dur, 'timing':
                            {'ITI': ngym.ngym_random.TruncExp(mean_ITI, 100, max_ITI),        
                             # mean, min, max
                            'fixation': fix_dur, 'decision': dec_dur},
                        # Decision period}
                        'rewards': {'abort': 0., 'fixation': 0., 'correct': 1.}}
        # call function to sample
        env = gym.make(TASK, **env_kwargs)
        env = pass_reward.PassReward(env)
        env = pass_action.PassAction(env)

    # set seed
    env.seed(env_seed)
    env.reset()
    # store mean ITI in env_kwargs
    env_kwargs['mean_ITI'] = mean_ITI
    return env_kwargs, env

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


def analysis_activity_by_condition(activity, info, config,
                                   conditions=['choice']):
    """
    Plot single neuron activity by condition.
    """
    for condition in conditions:
        values = pd.unique(info[condition])
        f, ax = plt.subplots(figsize=(10, 3), ncols=len(
            values), sharex=True, dpi=150)
        t_plot = np.arange(activity.shape[1]) * config['dt']
        for i_v, value in enumerate(values):
            # plot the average activity across neurons and
            # trials for each condition
            a = activity[info[condition] == value]
            ax[i_v].imshow(a.mean(axis=0).T, aspect='auto', cmap='viridis')
            ax[i_v].set_xlabel('Time (ms)')
            ax[i_v].set_ylabel('Mean activity for ' +
                               condition + ' = ' + str(value))
            # change the xticks to show time in ms
            ax[1].set_xticks(np.arange(0, activity.shape[1], 10))
            ax[1].set_xticklabels(t_plot[::10])
        plt.tight_layout()


def probit(x, beta, alpha):
    """
    Return probit function with parameters alpha and beta.

    Parameters
    ----------
    x : float
        independent variable.
    beta : float
        sensitiviy.
    alpha : TYPE
        bias term.

    Returns
    -------
    probit : float
        probit value for the given x, beta and alpha.

    """
    probit = 1/2*(1+erf((beta*x+alpha)/np.sqrt(2)))
    return probit


def equalize_arrays(array_list):
    """

    Parameters
    ----------
    array_list : TYPE
        DESCRIPTION.

    Returns
    -------
    padded_arrays : TYPE
        DESCRIPTION.

    """
    # Find the maximum shape among the arrays
    max_shape = max(arr.shape[0] for arr in array_list)

    # Pad each array to match the maximum shape
    padded_arrays = []
    for arr in array_list:
        if len(arr.shape) == 1:  # If the array is one-dimensional
            pad_width = ((max_shape - arr.shape[0], 0))  # Pad at the beginning
        elif len(arr.shape) == 2:  # If the array is two-dimensional
            pad_width = ((max_shape - arr.shape[0], 0), (0, 0))
        elif len(arr.shape) == 3:
            pad_width = ((max_shape - arr.shape[0], 0), (0, 0), (0, 0))
        padded_array = np.pad(arr, pad_width, mode='constant',
                              constant_values=0)
        padded_arrays.append(padded_array)

    return padded_arrays


def run_agent_in_environment(num_steps_exp, env, net=None):
    """
    Run the agent in the environment for a specified number of steps.

    Parameters
    ----------
    env :
        The environment in which the agent interacts.
    num_steps_exp : int
        The number of steps to run the agent in the environment.

    Returns
    -------
    data : dict
        A dictionary containing recorded data:
            - 'ob': Observations received from the environment.
            - 'actions': Actions taken by the agent.
            - 'gt': Ground truth information.
            - 'perf': Information on performance.
            - 'rew_mat': Reward matrix.
    """
    actions = []
    act_pr_mat = []
    gt = []
    perf = []
    rew_mat = []
    iti = []
    prob_r = []
    prob_l = []
    rew = 0
    action = 0
    ob, _, _, _, _ = env.step(action)
    inputs = [ob]
    if net is not None:
        hidden = torch.zeros(1, 1, net.hidden_size)
    for stp in range(int(num_steps_exp)):
        if net is None:
            action = env.action_space.sample()
        else:
            ob_tensor = torch.tensor([ob], dtype=torch.float32)
            ob_tensor = ob_tensor.unsqueeze(0)
            action_probs, hidden = net(x=ob_tensor, hidden=hidden)
            act_pr_mat.append(action_probs)
            # Assuming `net` returns action probabilities
            action_probs = torch.nn.functional.softmax(action_probs, dim=2)
            action = torch.argmax(action_probs[0, 0]).item()

        ob, rew, _, _, info = env.step(action)

        inputs.append(ob)
        actions.append(action)
        gt.append(info.get('gt', None))
        if isinstance(rew, np.ndarray):
            rew_mat.append(rew[0])
        else:
            rew_mat.append(rew)
        if info.get('new_trial', False):
            perf.append(info.get('performance', None))
            iti.append(np.sum(env.gt == 0))
            prob_r.append(env.trial['probs'][0])
            prob_l.append(env.trial['probs'][1])
        else:
            perf.append(-1)

    perf = np.array(perf)
    mean_perf = np.mean(perf[perf != -1])
    mean_rew = np.mean(rew_mat)

    data = {'ob': np.array(inputs[:-1]).astype(float),
            'actions': actions, 'gt': gt, 'perf': perf,
            'rew_mat': rew_mat, 'mean_perf': mean_perf,
            'mean_rew': mean_rew, 'iti': iti, 'prob_r': prob_r,
            'prob_l': prob_l, 'act_pr_mat': act_pr_mat}
    return data


def plot_dataset(dataset):
    f, ax = plt.subplots(nrows=4, sharex=True)
    for i in range(2):
        inputs = dataset['inputs'][i, :, :]
        labels = dataset['labels'][i, :]
        labels_b = labels[:, np.newaxis]
        ax[2*i].imshow(inputs.T, aspect='auto')
        ax[2*i+1].imshow(labels_b.T, aspect='auto')


def plot_task(env_kwargs, data, num_steps, save_folder=None):
    """
    Parameters
    ----------
    env_kwargs : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    num_steps : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    f, ax = plt.subplots(ncols=1, nrows=4, figsize=(6, 3),
                         dpi=150, sharex=True)

    ax[0].plot(np.arange(1, num_steps+1)*env_kwargs['dt'],
               data['ob'][:num_steps], '-+')
    ax[0].set_ylabel('Inputs')
    ax[0].legend(['Fixation', 'Reward', 'Choice'])
    ax[1].plot(np.arange(1, num_steps+1)*env_kwargs['dt'],
               data['gt'][:num_steps], label='Targets', color='k')
    ax[1].plot(np.arange(1, num_steps+1)*env_kwargs['dt'],
               data['actions'][:num_steps], label='Choice',
               linestyle='--', marker='+')
    ax[1].set_ylabel('Actions / Targets')
    ax[1].legend()
    ax[2].plot(np.arange(1, num_steps+1)*env_kwargs['dt'],
               data['perf'][:num_steps],
               label='perf')
    # set title with average performance
    perf = np.array(data['perf'])
    perf = perf[perf != -1]
    mean_perf = np.mean(perf)
    ax[2].set_title('Mean performance: ' + str(np.round(mean_perf, 2)))
    ax[2].set_ylabel('Performance')
    ax[3].plot(np.arange(1, num_steps+1)*env_kwargs['dt'],
               data['rew_mat'][:num_steps], label='perf')
    ax[3].set_ylabel('Reward')
    ax[3].set_xlabel('Time (ms)')
    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(save_folder + '/task.png')
        plt.savefig(save_folder + '/task.svg')
        plt.close(f)


def dict2df(data):
    """
    Transform data dictionary to pandas dataframe.

    Parameters
    ----------
    data : dict
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    # transform data to a pandas dataframe.
    actions = np.array(data['actions'])
    reward = np.array(data['rew_mat'])
    gt = np.array(data['gt'])
    # keep only gt corresponding to choice
    indx = (gt != 0) & (gt != 1)
    actions = actions[indx]
    reward = reward[indx]
    gt = gt[indx]
    df = pd.DataFrame({'actions': actions, 'gt': gt, 'iti': data['iti'],
                       'prob_r': data['prob_r'], 'reward': reward})
    return df


def train_network(num_periods, criterion, env, net_kwargs, env_kwargs,
                  seq_len, debug=False, seed=0, save_folder=None, num_trials_perf=100):
    """
    Train a recurrent neural network (RNN) in a specified environment.

    Parameters
    ----------
    num_periods : int
        Number of training periods.
    criterion : torch.nn.Module
        Loss function to optimize.
    env : gym.Env
        Environment in which the agent will be trained.
    net_kwargs : dict
        Dictionary containing network parameters including input_size, hidden_size, and output_size.
    env_kwargs : dict
        Dictionary containing environment parameters.
    seq_len : int
        Sequence length for training.
    debug : bool, optional
        If True, enables debug mode with additional plots (default is False).
    seed : int, optional
        Random seed for reproducibility (default is 0).
    save_folder : str, optional
        Folder to save the trained network and training data (default is None).
    num_trials_perf : int, optional
        Number of trials to compute performance (default is 100).

    Returns
    -------
    dict
        A dictionary containing the training performance metrics including mean performance, mean reward, loss, and error metrics.
    """
    # create network    
    net = Net(input_size=net_kwargs['input_size'],
              hidden_size=net_kwargs['hidden_size'],
              output_size=env.action_space.n, seed=seed)

    # Move network to the device (CPU or GPU)
    net = net.to(DEVICE)
    torch.save(net, save_folder + '/net.pth')
    optimizer = torch.optim.Adam(net.parameters(), lr=TRAINING_KWARGS['lr'])

    mean_perf_list = []
    mean_rew_list = []
    loss_1st_ep_list = []
    error_no_action_list = []
    error_fixation_list = []
    error_2_list = []
    error_3_list = []
    log_per = 20
    # compute mean trial duration
    mean_trial_dur =\
          (env_kwargs['mean_ITI']+env_kwargs['timing']['fixation']+env_kwargs['timing']['decision'])/env_kwargs['dt']
    # mean number of trials per episode
    num_trials_per_ep = seq_len//mean_trial_dur
    # number of episodes to compute performance
    num_eps_perf = num_trials_perf//num_trials_per_ep
    # mean performance across episodes
    temp_mean_perf = []
    # maximum performance to save the network
    maximum_perf = 0
    # open txt file to save data
    for i_per in range(num_periods):
        if i_per % 100 == 0:
            print('Period:', i_per, 'of', num_periods)
        data = run_agent_in_environment(env=env, net=net,
                                        num_steps_exp=seq_len)
        # transform list of torch to torch tensor
        outputs = torch.stack(data['act_pr_mat'], dim=1)
        # squeeze the outputs tensor to remove the dimension of size 1
        outputs = outputs.squeeze()
        labels = np.array(data['gt'])
        outputs = outputs.type(torch.float).to(DEVICE)
        labels = torch.from_numpy(labels).type(torch.long).to(DEVICE)

        if debug:
            plot_task(env_kwargs=env_kwargs, data=data,
                      num_steps=seq_len, save_folder=save_folder)
        # transform data to a pandas dataframe.
        df = dict2df(data)
        mean_perf_list.append(data['mean_perf'])
        mean_rew_list.append(data['mean_rew'])
        # we need to zero the parameter gradients to re-initialize and
        # avoid they accumulate across epochs
        optimizer.zero_grad()
        # compute loss with respect to the labels
        loss = criterion(outputs, labels)
        # compute gradients
        loss.backward()
        # update weights
        optimizer.step()
        loss_step = loss.item()
        loss_1st_ep_list.append(loss_step)
        # compute performance for the last num_eps_perf episodes
        temp_mean_perf.append(data['mean_perf'])
        if len(temp_mean_perf) > num_eps_perf:
            temp_mean_perf.pop(0)
        if i_per % log_per == log_per-1 and np.mean(temp_mean_perf) >= maximum_perf:
            # open and write to file
            # print temp_mean_perf rounding to 2 decimals and transforming to normal numbers (not )
            with open(save_folder + '/training_data.txt', 'a') as f:
                f.write('------------\n')
                f.write('Period: ' + str(i_per) + ' of ' + str(num_periods) + '\n')
                f.write('Mean performance period: ' + str(data['mean_perf']) + '\n')
                f.write('Mean performance: ' + str(round(np.mean(temp_mean_perf), 2)) + '\n')
                f.write('Mean reward: ' + str(data['mean_rew']) + '\n')
                f.write('Loss: ' + str(loss_step) + '\n')
            # save net
            torch.save(net, save_folder + '/net.pth')
            maximum_perf = np.mean(temp_mean_perf)

        error_dict = compute_error(data)
        error_no_action_list.append(error_dict['error_no_action'])
        error_fixation_list.append(error_dict['error_fixation'])
        error_2_list.append(error_dict['error_2'])
        error_3_list.append(error_dict['error_3'])
    dict = {'mean_perf_list': mean_perf_list, 'mean_rew_list': mean_rew_list,
            'loss_1st_ep_list': loss_1st_ep_list,
            'error_no_action_list': error_no_action_list,
            'error_fixation_list': error_fixation_list,
            'error_2_list': error_2_list,
            'error_3_list': error_3_list}
    return dict


def preprocess_activity(activity):
    silent_idx = np.where(activity.sum(axis=(0, 1)) == 0)[0]
    print('fraction of silent neurons:', len(silent_idx)/activity.shape[-1])

    # activity for one trial, but now excluding the
    # silent neurons
    clean_activity = activity[:, :, np.delete(
        np.arange(activity.shape[-1]), silent_idx)]

    # min_max scaling
    minmax_activity = np.array(
        [neuron-neuron.min() for neuron in
         clean_activity.transpose(2, 0, 1)]).transpose(1, 2, 0)
    minmax_activity = np.array(
        [neuron/neuron.max() for neuron in
         minmax_activity.transpose(2, 0, 1)]).transpose(1, 2, 0)
    return minmax_activity


def plot_activity(activity, obs, actions, gt, config, trial):

    # Load and preprocess results
    f, ax = plt.subplots(figsize=(5, 4), nrows=3, dpi=150)

    # time in ms
    t_plot = np.arange(activity.shape[1]) * config['dt']

    # plot the observations for one trial. Note that we will visualize the
    # inputs as a matrix instead of traces, as we have done before.
    ax[0].plot(obs[trial])
    ax[0].set_title('Observations')
    ax[0].set_ylabel('Stimuli')
    # change the xticks to show time in ms
    # plot the activity for one trial
    ax[1].imshow(activity[trial].T, aspect='auto', cmap='viridis')
    ax[1].set_title('Activity')
    ax[1].set_ylabel('Neurons')
    # plt.colorbar(im, ax=ax[1])
    # change the xticks to show time in ms
    ax[1].set_xticks(np.arange(0, activity.shape[1], 10))
    ax[1].set_xticklabels(t_plot[::10])

    ax[2].plot(actions[trial], label='actions')
    ax[2].plot(gt[trial], '--', label='gt')
    ax[2].legend()
    ax[2].set_title('Actions')
    ax[2].set_xlabel('Time (ms)')
    ax[2].set_ylabel('Action')
    # change the xticks to show time in ms

    plt.tight_layout()


def plot_perf_rew_loss(num_periods, mean_perf, mean_rew, loss_1st_ep,
                       save_folder_net):
    """
    Plots mean performance and mean reward as a function of period

    Parameters
    ----------
    num_periods : int
        number of periods
    mean_perf : list
    mean_rew : list

    """
    period = range(num_periods)
    f, ax = plt.subplots(nrows=3)
    ax[0].plot(period, mean_perf, marker='.', linestyle='-')
    ax[0].set_ylabel('Mean performance', fontsize=14)
    ax[0].tick_params(axis='both', labelsize=12)
    ax[1].plot(period, mean_rew, marker='.', linestyle='-')
    ax[1].set_ylabel('Mean reward', fontsize=14)
    ax[1].tick_params(axis='both', labelsize=12)
    ax[2].plot(period, loss_1st_ep, marker='.', linestyle='-')
    ax[2].set_ylabel('Loss 1st epoch', fontsize=14)
    ax[2].set_xlabel('Period', fontsize=14)
    ax[2].tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(save_folder_net + '/perf_rew_loss.png')


def compute_error(data):

    gt = np.array(data['gt'])
    act = np.array(data['actions'])

    indices_0 = np.where(gt == 0)[0]
    prediction_no_action = act[indices_0]
    error_no_action = np.sum(prediction_no_action != 0)

    indices_1 = np.where(gt == 1)[0]
    prediction_fixation = act[indices_1]
    error_fixation = np.sum(prediction_fixation != 1)

    indices_2 = np.where(gt == 2)[0]
    prediction_2 = act[indices_2]
    error_2 = np.sum(prediction_2 != 2)

    indices_3 = np.where(gt == 3)[0]
    prediction_3 = act[indices_3]
    error_3 = np.sum(prediction_3 != 3)

    error_dict = {'error_no_action': error_no_action,
                  'error_fixation': error_fixation,
                  'error_2': error_2,
                  'error_3': error_3}

    return error_dict


def plot_error(num_periods, error_no_action_list, error_fixation_list,
               error_2_list, error_3_list, save_folder_net):
    period = range(num_periods)
    f, ax = plt.subplots(nrows=4, sharex=True)
    plt.suptitle('Error', fontsize=16)
    ax[0].plot(period, error_no_action_list, marker='.', linestyle='-')
    ax[0].set_ylabel('No action', fontsize=14)
    ax[0].tick_params(axis='both', labelsize=12)
    ax[1].plot(period, error_fixation_list, marker='.', linestyle='-')
    ax[1].set_ylabel('Fixation', fontsize=14)
    ax[1].tick_params(axis='both', labelsize=12)
    ax[2].plot(period, error_2_list, marker='.', linestyle='-')
    ax[2].set_ylabel('Decision 2', fontsize=14)
    ax[2].tick_params(axis='both', labelsize=12)
    ax[3].plot(period, error_3_list, marker='.', linestyle='-')
    ax[3].set_ylabel('Decision 3', fontsize=14)
    ax[3].tick_params(axis='both', labelsize=12)
    ax[3].set_xlabel('Period', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_folder_net + '/error.png')


def plot_performace_by_iti(data, save_folder):
    # boxplot of performance by ITI
    # list of unique ITIs
    iti_mat = np.array(data['iti'])
    iti_list = np.unique(iti_mat)
    perf_mat = np.array(data['perf'])
    perf_mat = perf_mat[perf_mat != -1]

    # list of performances for each unique ITI
    mean_perf = []
    ste_perf = []
    for iti in iti_list:
        mean_perf.append(np.mean(perf_mat[iti_mat == iti]))
        ste_perf.append(np.std(perf_mat[iti_mat == iti]) /
                        np.sqrt(np.sum(iti_mat == iti)))
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), dpi=150)
    ax[0].errorbar(iti_list, mean_perf, yerr=ste_perf, fmt='o')
    ax[0].set_xlabel('ITI')
    ax[0].set_ylabel('Performance')
    # plot bar plot with number of trials for each ITI
    ax[1].bar(iti_list, [np.sum(iti_mat == iti) for iti in iti_list])
    ax[1].set_xlabel('ITI')
    ax[1].set_ylabel('Number of trials')
    plt.tight_layout()
    plt.savefig(save_folder + '/perf_iti.png')


def process_dataframe(main_folder, filename, df, save_folder, env_seed, seed,
                      mean_ITI, fix_dur, blk_dur, seq_len, num_periods, lr):
    """
    Process a dataframe located in the specified folder.
    If the dataframe exists, modify it. Otherwise, create it with desired
    columns.

    Parameters:
        folder_path (str): Path to the folder containing the dataframe.
        filename (str): Name of the dataframe file.
        columns (list): List of column names for the dataframe.
        df (pandas DataFrame): DataFrame resulting from dict2df function.
            Contains actions, gt, iti, prob_r and reward

    Returns:
        DataFrame: The processed or newly created dataframe. Contains the
        columns:'params', 'env_seed', 'net_seed', 'actions', 'gt', 'iti',
        'prob_r', 'reward', 'mean_ITI', 'fix_dur', 'blk_dur',
        'seq_len'
    """
    columns = ['params', 'env_seed', 'net_seed', 'actions', 'gt', 'iti',
               'prob_r', 'reward', 'mean_ITI', 'fix_dur',
               'blk_dur', 'seq_len', 'num_periods', 'lr']
    # Check if the folder exists, if not, create it
    # if not os.path.exists(main_folder):
    #     os.makedirs(main_folder)

    file_path = os.path.join(main_folder, filename)

    # Check if the file exists
    if os.path.exists(file_path):
        # File exists, load the dataframe
        training_df = pd.read_csv(file_path)
        # Perform modifications here if needed
        # For example, add new columns, modify existing ones, etc.
    else:
        # File doesn't exist, create a new dataframe with desired columns
        training_df = pd.DataFrame(columns=columns)

    # remove main_folder path from save_folder to obtain just the params:
    params = os.path.relpath(save_folder, main_folder)

    # You can perform further operations here if needed
    values_to_add = pd.DataFrame({'params': [params]*len(df),
                                  'env_seed': [env_seed]*len(df),
                                  'net_seed': [seed]*len(df),
                                  'mean_ITI': [mean_ITI]*len(df),
                                  'fix_dur': [fix_dur]*len(df),
                                  'blk_dur': [blk_dur]*len(df),
                                  'seq_len': [seq_len]*len(df),
                                  'num_periods': [num_periods]*len(df),
                                  'lr': [lr]*len(df)})
    result_df = pd.concat([df, values_to_add], axis=1)
    # reset index after concatenation
    result_df.reset_index(drop=True, inplace=True)

    training_df = pd.concat([training_df, result_df], axis=0)
    # reset index after concatenation
    training_df.reset_index(drop=True, inplace=True)

    # Save the dataframe to file
    training_df.to_csv(file_path, index=False)

    return training_df


def train_multiple_networks(mean_ITI, max_ITI, fix_dur, blk_dur, w_factor,
                            num_networks, env, env_seed, main_folder,
                            save_folder, filename, env_kwargs, net_kwargs,
                            num_periods, seq_len, debug=False, num_steps_test=1000,
                            num_steps_plot=100,lr=None):
    if lr is not None:
        TRAINING_KWARGS['lr'] = lr
    # set weights for the loss function
    class_weights =\
    torch.tensor([w_factor*TRAINING_KWARGS['dt']/(mean_ITI),
                    w_factor*TRAINING_KWARGS['dt']/fix_dur, 2, 2])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    mperf_list = []
    for _ in range(num_networks):
        seed = np.random.randint(0, 1000000)
        # create folder to save data based on net seed and env seed
        save_folder_net = save_folder + '/envS_' + str(env_seed) + '_netS_' + str(seed)
        # create folder to save data based on net seed
        os.makedirs(save_folder_net, exist_ok=True)
        # save all params in a txt file
        with open(save_folder_net + '/params.txt', 'w') as f:
            f.write('env_seed: ' + str(env_seed) + '\n')
            f.write('net_seed: ' + str(seed) + '\n')
            f.write('mean_ITI: ' + str(mean_ITI) + '\n')
            f.write('max_ITI: ' + str(max_ITI) + '\n')
            f.write('fix_dur: ' + str(fix_dur) + '\n')
            f.write('blk_dur: ' + str(blk_dur) + '\n')
            f.write('seq_len: ' + str(seq_len) + '\n')
            f.write('num_periods: ' + str(num_periods) + '\n')
            f.write('lr: ' + str(lr) + '\n')
            f.write('w_factor: ' + str(w_factor) + '\n')
        data_behav = train_network(num_periods=num_periods,
                                   criterion=criterion,
                                   env=env, net_kwargs=net_kwargs,
                                   env_kwargs=env_kwargs,
                                   seq_len=seq_len,
                                   debug=debug, seed=seed,
                                   save_folder=save_folder_net)
        # save data as npz
        np.savez(save_folder_net + '/training_data.npz', **data_behav)
        if debug:
            # get data from data_behav
            mean_perf_list = data_behav['mean_perf_list']
            mean_rew_list = data_behav['mean_rew_list']
            loss_1st_ep_list = data_behav['loss_1st_ep_list']
            error_no_action_list = data_behav['error_no_action_list']
            error_fixation_list = data_behav['error_fixation_list']
            error_2_list = data_behav['error_2_list']
            error_3_list = data_behav['error_3_list']
            plot_perf_rew_loss(num_periods, mean_perf_list,
                            mean_rew_list,
                            loss_1st_ep_list, save_folder_net)

            plot_error(num_periods, error_no_action_list,
                    error_fixation_list,  error_2_list, error_3_list,
                    save_folder_net)
        # load network
        net = Net(input_size=net_kwargs['input_size'],
                hidden_size=net_kwargs['hidden_size'],
                output_size=env.action_space.n, seed=seed)

        # Move network to the device (CPU or GPU)
        net = torch.load(save_folder_net+'/net.pth', map_location=DEVICE, weights_only=False)
        net = net.to(DEVICE)  # Move it explicitly
        with torch.no_grad():
            data = run_agent_in_environment(num_steps_exp=num_steps_test, env=env,
                                            net=net)
            np.savez(save_folder_net + '/test_data.npz', **data)
            mperf_list.append(data['mean_perf'])
            plot_task(env_kwargs=env_kwargs, data=data, num_steps=num_steps_plot,
                    save_folder=save_folder_net)
            plot_performace_by_iti(data, save_folder=save_folder_net)
            plt.close('all')
            # df = dict2df(data)

            # # save the data fom the net in the dataframe
            # training_df = process_dataframe(main_folder=main_folder, num_periods=num_periods,
            #                                 filename=filename, df=df,
            #                                 save_folder=save_folder, lr=lr,
            #                                 env_seed=env_seed, seed=seed,
            #                                 mean_ITI=mean_ITI, fix_dur=fix_dur,
            #                                 blk_dur=blk_dur, seq_len=seq_len)
    return mperf_list, None


if __name__ == '__main__':
    # define parameters configuration
    env_seed = 123
    num_steps_plot = 200
    num_steps_test = 10000
    num_networks = 10
    # create folder to save data based on env seed
    # main_folder = 'C:/Users/saraf/OneDrive/Documentos/IDIBAPS/foraging RNNs/nets/'
    main_folder = '/home/manuel.molano/foragingRNNs/files/' # '/home/molano/foragingRNNs_data/nets/'

    # Create the main Tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    # Prompt the user for input using a pop-up dialog
    # experiment_type = simpledialog.askstring("Experiment Type", "Are you running a normal experiment (press Enter) or a test ('test')?")
    # Determine the variable based on user input
    test_flag = '' # experiment_type
    filename = 'training_data'+test_flag+'.csv'
    # Set up the task
    # THIS WORKS:
    # env_seed: 123
    # net_seed: 14710
    # mean_ITI: 400
    # fix_dur: 100
    # blk_dur: 25
    # seq_len: 100
    # num_periods: 12000
    # lr: 0.001
    # w_factor: 0.01

    w_factor = 0.01
    mean_ITI = 400
    max_ITI = 800
    fix_dur = 100
    dec_dur = 100
    prob = 0.8
    probs = np.array([[1-prob, prob], [prob, 1-prob]])
    # create folder to save data based on parameters
    save_folder = (f"{main_folder}{TASK}_w{w_factor}_mITI{mean_ITI}_xITI{max_ITI}_f{fix_dur}_"
                    f"d{dec_dur}_"f"prb{probs[0]}")   
    save_folder = save_folder.replace('[', '').replace(']', '')
    save_folder = save_folder.replace(' ', '')
    save_folder = save_folder.replace('-v0', '')
    # define parameter to explore
    lr_mat = np.array([1e-3]) # np.array([1e-3, 1e-2, 3e-2]) Learning Rate
    blk_dur_mat = np.array([25]) # np.array([25, 50, 100]) Block duration
    seq_len_mat = np.array([100]) # np.array([50, 300, 500]) Sequence length ()
    total_num_timesteps = 1200000 # 
    train = True
    debug = False
    if train:
        for bd in blk_dur_mat:
            # create the environment with the parameters
            env_kwargs, env = create_env(env_seed=env_seed, mean_ITI=mean_ITI, max_ITI=max_ITI,
                                        fix_dur=fix_dur, dec_dur=dec_dur,
                                        blk_dur=bd, probs=probs)
            if debug:
                data = run_agent_in_environment(num_steps_exp=10000, env=env)
                gt = np.array(data['gt'])
                perf = np.array(data['perf'])
                gt = gt[perf!=-1]
                prob = np.array(data['prob_l'])
                plt.figure()
                plt.plot(gt, color='k')
                plt.plot(prob+2, color='r')
                # check mean gt when prob is 0.8
                choice_l = gt == 3
                mean_gt = np.mean(choice_l[prob==0.8])
                print(mean_gt)
                plt.show()
            net_kwargs = {'hidden_size': 128,
                        'action_size': env.action_space.n,
                        'input_size': env.observation_space.n}

            for seq_len in seq_len_mat:
                num_periods = total_num_timesteps // seq_len
                for lr in lr_mat:
                    _, _ = train_multiple_networks(mean_ITI=mean_ITI, max_ITI=max_ITI, fix_dur=fix_dur, blk_dur=bd,
                                                num_networks=num_networks, env=env, w_factor=w_factor,
                                                env_seed=env_seed, main_folder=main_folder, save_folder=save_folder,
                                                filename=filename, env_kwargs=env_kwargs, net_kwargs=net_kwargs,
                                                num_periods=num_periods, seq_len=seq_len, lr=lr, debug=debug,
                                                num_steps_test=num_steps_test, num_steps_plot=num_steps_plot)

    # find all experiments in the folder
    exp_folders = glob.glob(save_folder + '/env*')
    # create list to store training data
    training_data = []
    test_data = [] 
    # for each exp folder find data.npz
    for exp_folder in exp_folders:
        try:
            data = np.load(exp_folder + '/training_data.npz')
        except FileNotFoundError:
            data = np.load(exp_folder + '/data.npz')
        # get mean performance
        mean_perf = data['mean_perf_list']
        # smooth mean performance
        mean_perf = np.convolve(mean_perf, np.ones(100)/100, mode='valid')
        training_data.append(mean_perf)
        try:
            # get testing performance
            data = np.load(exp_folder + '/test_data.npz')
            # get mean performance
            test_data.append(data['mean_perf'])
        except FileNotFoundError:
            continue
     # create figure to plot performance across training
    f, ax = plt.subplots(figsize=(7, 3), nrows=1, ncols=2, dpi=150)
    ax[0].plot(np.array(training_data).T, color='gray', alpha=0.5)
    ax[0].plot(np.mean(training_data, axis=0), color='k', linewidth=2)
    ax[0].set_xlabel('Period')
    ax[0].set_ylabel('Mean performance')
    ax[1].hist(test_data)
    ax[1].set_xlabel('Mean performance')
    ax[1].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    plt.savefig(save_folder + '/mean_perf.png')
    # filename ='training_data_bias_corrected_th04.csv'

    # # boxplots for each parameter configuration
    # lr_mat =  np.array([1e-3]) # np.array([1e-3, 1e-2, 3e-2])
    # seq_len_mat =  np.array([50]) # np.array([50, 300, 500])
    # blk_dur_mat = np.array([25])
    
    # fa.get_mean_perf_by_param_comb(lr_mat=lr_mat, blk_dur_mat=blk_dur_mat, seq_len_mat=seq_len_mat, main_folder=main_folder,
    #                                filename=filename)

    # fa.get_perf_by_param_comb_all_nets(lr_mat=lr_mat, blk_dur_mat=blk_dur_mat, seq_len_mat=seq_len_mat, main_folder=main_folder,
    #                                    filename=filename)


