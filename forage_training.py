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
import sklearn.discriminant_analysis as sklda
import sklearn.model_selection as sklms
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
import sys
import time
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

# name of the task on the neurogym library
TASK = 'ForagingBlocks-v0'

TRAINING_KWARGS = {'dt': 100,
                   'lr': 1e-2,
                   'n_epochs': 100,
                   'batch_size': 16,
                   'seq_len': 200,
                   'TASK': TASK}


def get_dataset(TASK, env_kwargs):
    """
    Create neurogym dataset and environment.

    args:
        TASK (str): name of the task on the neurogym library
        env_kwargs (dict): task parameters

    returns:
        dataset (neurogym.Dataset): dataset object from which we can sample
        trials and labels
        env (gym.Env): task environment
    """

    # Make supervised dataset using neurogym's Dataset class
    dataset = ngym_f.Dataset(TASK,
                             env_kwargs=env_kwargs,
                             batch_size=TRAINING_KWARGS['batch_size'],
                             seq_len=TRAINING_KWARGS['seq_len'])
    env = dataset.env

    return dataset, env


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seed=0):
        super(Net, self).__init__()

        self.hidden_size = hidden_size
        # INSTRUCTION 1: build a recurrent neural network with a single
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
        # INSTRUCTION 2: get the output of the network for a given input
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
            # INSTRUCTION 13: plot the average activity across neurons and
            # trials for each condition
            a = activity[info[condition] == value]
            ax[i_v].imshow(a.mean(axis=0).T, aspect='auto', cmap='viridis')
            ax[i_v].set_xlabel('Time (ms)')
            ax[i_v].set_ylabel('Mean activity for ' +
                               condition + ' = ' + str(value))
            # change the xticks to show time in ms
            ax[1].set_xticks(np.arange(0, activity.shape[1], 10))
            ax[1].set_xticklabels(t_plot[::10])

        # plt.legend(title=condition, loc='center left',
        # bbox_to_anchor=(1.0, 0.5))


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
    gt = []
    perf = []
    rew_mat = []
    iti = []
    prob_r = []
    prob_l = []
    rew = 0
    action = 0
    ob, _, _, _ = env.step(action)
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
            # Assuming `net` returns action probabilities
            action_probs = torch.nn.functional.softmax(action_probs, dim=2)
            action = torch.argmax(action_probs[0, 0]).item()

        ob, rew, done, info = env.step(action)

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
            
    print('------------') 
    perf = np.array(perf)           
    mean_perf = np.mean(perf[perf != -1])
    print('mean performance: ', mean_perf)
    mean_rew = np.mean(rew_mat)
    print('mean reward: ', mean_rew)
    print('------------')
    data = {'ob': np.array(inputs[:-1]).astype(float),
            'actions': actions, 'gt': gt, 'perf': perf,
            'rew_mat': rew_mat, 'mean_perf': mean_perf,
            'mean_rew': mean_rew, 'iti': iti, 'prob_r': prob_r,
            'prob_l': prob_l}
    return data


def build_dataset(data):

    # OBSERVATION
    ob_array = np.array(data['ob'])
    # reshape
    inputs = ob_array.reshape(TRAINING_KWARGS['batch_size'],
                              TRAINING_KWARGS['seq_len'], 3)
        
    # labels
    labels = np.array(data['gt'])
    # reshape
    labels = labels.reshape(TRAINING_KWARGS['batch_size'],
                            TRAINING_KWARGS['seq_len'])

    dataset = {'inputs': inputs, 'labels': labels}
    return dataset


def plot_dataset(dataset, batch=0):
    f, ax = plt.subplots(nrows=4, sharex=True)
    for i in range(2):
        inputs = dataset['inputs'][i,:,:]
        labels = dataset['labels'][i,:]
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
    f, ax = plt.subplots(ncols=1, nrows=4, figsize=(8, 4),
                         dpi=150, sharex=True)

    ax[0].plot(np.arange(1, num_steps+1)*env_kwargs['dt'],
               data['ob'][:num_steps], '-+')
    ax[0].set_ylabel('Inputs')
    ax[0].legend(['Fixation', 'Reward', 'Choice'])
    ax[1].plot(np.arange(1, num_steps+1)*env_kwargs['dt'],
               data['gt'][:num_steps], label='Targets', color='k')
    ax[1].plot(np.arange(1, num_steps+1)*env_kwargs['dt'],
               data['actions'][:num_steps], label='Choice', linestyle='--', marker='+')
    ax[1].set_ylabel('Actions / Targets')
    ax[1].legend()
    ax[2].plot(np.arange(1, num_steps+1)*env_kwargs['dt'], data['perf'][:num_steps],
               label='perf')
    # set title with average performance
    perf = np.array(data['perf'])
    perf = perf[perf != -1]
    mean_perf = np.mean(perf)
    ax[2].set_title('Mean performance: ' + str(np.round(mean_perf, 2)))
    ax[2].set_ylabel('Performance')
    ax[3].plot(np.arange(1, num_steps+1)*env_kwargs['dt'], data['rew_mat'][:num_steps],
               label='perf')
    ax[3].set_ylabel('Reward')
    ax[3].set_xlabel('Time (ms)')
    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(save_folder + '/task.png')


def train_network(num_epochs, num_periods, num_steps_exp,
                   criterion, env, net_kwargs, env_kwargs, debug=False, seed=0):
    """
    """
    net = Net(input_size=net_kwargs['input_size'],
              hidden_size=net_kwargs['hidden_size'],
              output_size=env.action_space.n, seed=seed)

    # Move network to the device (CPU or GPU)
    net = net.to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=TRAINING_KWARGS['lr'])

    mean_perf_list = []
    mean_rew_list = []
    loss_1st_ep_list = []
    error_no_action_list = []
    error_fixation_list = []
    error_2_list = []
    error_3_list = []
    
    for i_per in range(num_periods):
        # dataset = {'inputs':seq_len x batch_size x num_inputs,
        #            'labels': seq_len x batch_size}
        print('Period: ', i_per, 'of', num_periods)
        with torch.no_grad():
            data = run_agent_in_environment(env=env, net=net,
                                            num_steps_exp=num_steps_exp)
        if debug:
            plot_task(env_kwargs=env_kwargs, data=data,
                      num_steps=num_steps_exp)
        # transform data to a pandas dataframe. 
        # First, transform variables already existing in data
        # Transform means: change the name, shape, values and type of the variable
        # df = dict2df(data)
        # print frequency of each action when data['perf'] != -1 using numpy
        # print('Frequency of each action when perf != -1:')
        # print(np.unique(np.array(data['actions'])[data['perf'] != -1], return_counts=True))
        # print('Frequency of each target when perf != -1:')
        # print(np.unique(np.array(data['gt'])[data['perf'] != -1], return_counts=True))
        mean_perf_list.append(data['mean_perf'])
        mean_rew_list.append(data['mean_rew'])
        # end function

        dataset = build_dataset(data)
        if debug:
            plot_dataset(dataset)
        # Train model with RL data
        loss_1st_ep = train(num_epochs=num_epochs, dataset=dataset,
                            net=net, optimizer=optimizer,
                            criterion=criterion, env=env)
        loss_1st_ep_list.append(loss_1st_ep)
        
        error_dict = compute_error(data)
        error_no_action_list.append(error_dict['error_no_action'])
        error_fixation_list.append(error_dict['error_fixation'])
        error_2_list.append(error_dict['error_2'])        
        error_3_list.append(error_dict['error_3'])
    dict = {'mean_perf_list': mean_perf_list, 'mean_rew_list': mean_rew_list,
            'loss_1st_ep_list': loss_1st_ep_list, 'error_no_action_list': error_no_action_list,
            'error_fixation_list': error_fixation_list, 'error_2_list': error_2_list,
            'error_3_list': error_3_list}
    return dict, net


def train(num_epochs, net, optimizer, criterion, env, dataset):
    """
    Train the neural network.

    Parameters
    ----------
    num_epochs : int
        The number of epochs for training.
    net:
        The neural network model to be trained.
    optimizer:
        The optimizer used for updating the model paramenters.
    criterion:
        The loss function used for computing the loss.
    env:


    Returns
    -------
    None.

    """
    # print('Training task ', TASK)
    running_loss = 0.0
    for ep in range(num_epochs):
        # get inputs and labels and pass them to the GPU
        # TODO: HERE
        inputs = dataset['inputs']
        labels = dataset['labels']
        # inputs = np.expand_dims(inputs, axis=2)
        inputs = torch.from_numpy(inputs).type(torch.float).to(DEVICE)
        labels = torch.from_numpy(labels.flatten()).type(torch.long).to(DEVICE)
        # print shapes of inputs and labels
        if ep == -1:
            print('inputs shape: ', inputs.shape)
            print('labels shape: ', labels.shape)
            print('Max labels: ', labels.max())
        # we need zero the parameter gradients to re-initialize and avoid they
        # accumulate across epochs
        optimizer.zero_grad()

        # INSTRUCTION 3: FORWARD PASS: get the output of the network for a
        # given input
        outputs, _ = net(inputs)

        # reshape outputs so they have the same shape as labels
        outputs = outputs.view(-1, env.action_space.n)

        #  INSTRUCTION 4: compute loss with respect to the labels
        loss = criterion(outputs, labels)

        # INSTRUCTION 5: compute gradients
        loss.backward()

        # INSTRUCTION 6: update weights
        optimizer.step()

        # print average loss over last 200 training iterations and save the
        # current network
        running_loss += loss.item()
        if ep == 0:
            loss_1st_ep =  running_loss / 200
        # if ep % 2 == 0:
        #     print('{:d} loss: {:0.5f}'.format(ep + 1, running_loss / 200))
        #     running_loss = 0.0

            # save current state of network's parameters
            # torch.save(net.state_dict(), get_modelpath(TASK) / 'net.pth')
            
    return loss_1st_ep

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
    # INSTRUCTION 11: plot the activity for one trial
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

def plot_perf_rew_loss(num_periods, mean_perf, mean_rew, loss_1st_ep, save_folder_net):
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
        ste_perf.append(np.std(perf_mat[iti_mat == iti])/np.sqrt(np.sum(iti_mat == iti)))
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



# --- MAIN
if __name__ == '__main__':
    plt.close('all')
    env_seed = 6
    num_periods = 1000
    TRAINING_KWARGS['num_periods'] = num_periods
    # create folder to save data based on env seed
    # main_folder = 'C:/Users/saraf/OneDrive/Documentos/IDIBAPS/foraging RNNs/nets/'
    main_folder = '/home/molano/foragingRNNs_data/nets/'
    # Set up the task
    w_factor = 0.1
    mean_ITI = 150
    max_ITI = 200
    fix_dur = 100
    dec_dur = 100
    blk_dur = 50
    env_kwargs = {'dt': TRAINING_KWARGS['dt'], 'probs': np.array([0, 1]),
                  'blk_dur': blk_dur, 'timing':
                      {'ITI': ngym_f.random.TruncExp(mean_ITI, 100, max_ITI), # mean, min, max
                       'fixation': fix_dur, 'decision': dec_dur},  # Decision period}
                       'rewards':{'abort': 0., 'fixation': 0., 'correct': 1.}}  
    TRAINING_KWARGS['classes_weights'] =\
         torch.tensor([w_factor*TRAINING_KWARGS['dt']/(mean_ITI), w_factor*TRAINING_KWARGS['dt']/fix_dur, 2, 2])
    # torch.tensor([1., 1., 1., 1.]) 
    # call function to sample
    env = gym.make(TASK, **env_kwargs)
    env = pass_reward.PassReward(env)
    env = pass_action.PassAction(env)
    # set seed
    env.seed(env_seed)
    env.reset()
    net_kwargs = {'hidden_size': 128,
                  'action_size': env.action_space.n,
                  'input_size': env.observation_space.shape[0]}
    TRAINING_KWARGS['env_kwargs'] = env_kwargs
    TRAINING_KWARGS['net_kwargs'] = net_kwargs
    # plot task
    data = run_agent_in_environment(num_steps_exp=100, env=env)
    plot_task(env_kwargs=env_kwargs, data=data, num_steps=100)
    
    # create folder to save data based on parameters
    save_folder =\
          f"{main_folder}w{w_factor}_mITI{mean_ITI}_xITI{max_ITI}_f{fix_dur}_d{dec_dur}_n{num_periods}_nb{np.round(blk_dur/1e3, 1)}K_{env_seed}"


    # create folder to save data based on env seed
    os.makedirs(save_folder, exist_ok=True)


    # Save config as npz
    np.savez(save_folder+'/config.npz', **TRAINING_KWARGS)
    # Save config
    # with open(save_folder+'/config.json', 'w') as f:
    #     json.dump(TRAINING_KWARGS, f)
    # asdasdasd
    num_epochs = TRAINING_KWARGS['n_epochs']
    num_steps_exp =\
        TRAINING_KWARGS['seq_len']*TRAINING_KWARGS['batch_size']
    num_steps_plot = 100
    debug = False
    num_networks = 10
    criterion = nn.CrossEntropyLoss(weight=TRAINING_KWARGS['classes_weights'])
    # train several networks with different seeds
    for i_net in range(num_networks):
        seed = np.random.randint(0, 10000)
        # create folder to save data based on net seed
        save_folder_net = save_folder + '/' + str(seed)
        # create folder to save data based on net seed
        os.makedirs(save_folder_net, exist_ok=True)
        
        data_behav, net = train_network(num_epochs=num_epochs, num_periods=TRAINING_KWARGS['num_periods'],
                                  num_steps_exp=num_steps_exp, criterion=criterion,
                                  env=env, net_kwargs=net_kwargs, env_kwargs=env_kwargs,
                                  debug=debug, seed=seed)
        # save data as npz
        # HINT: use npy?
        np.savez(save_folder_net + '/data.npz', **data_behav)

        # save net
        torch.save(net, save_folder_net + '/net.pth')
            
        # get data from data_behav
        mean_perf_list = data_behav['mean_perf_list']
        mean_rew_list = data_behav['mean_rew_list']
        loss_1st_ep_list = data_behav['loss_1st_ep_list']
        error_no_action_list = data_behav['error_no_action_list']
        error_fixation_list = data_behav['error_fixation_list']
        error_2_list = data_behav['error_2_list']
        error_3_list = data_behav['error_3_list']
        plot_perf_rew_loss(TRAINING_KWARGS['num_periods'], mean_perf_list, mean_rew_list,
                        loss_1st_ep_list, save_folder_net)
        
        plot_error(TRAINING_KWARGS['num_periods'], error_no_action_list, error_fixation_list, 
                error_2_list, error_3_list, save_folder_net)
        data = run_agent_in_environment(num_steps_exp=num_steps_exp, env=env, net=net)
        plot_task(env_kwargs=env_kwargs, data=data, num_steps=num_steps_plot,
                   save_folder=save_folder_net)
        plot_performace_by_iti(data, save_folder=save_folder_net)
       # plt.show()
        plt.close('all')

    
    # load configuration file - we might have run the training on the cloud
    # and might now open the results locally
    # with open(get_modelpath(TASK) / 'config.json') as f:
    #     config = json.load(f)
