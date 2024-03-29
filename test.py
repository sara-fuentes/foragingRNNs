# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 22:20:23 2024

@author: saraf
"""

import torch.nn as nn
import torch
import ngym_foraging as ngym_f
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


def get_modelpath(TASK):
    # Make a local file directories
    path = Path('.') / 'files'
    os.makedirs(path, exist_ok=True)
    path = path / TASK
    os.makedirs(path, exist_ok=True)
    return path


def get_dataset(TASK, env_kwargs, training_kwargs):
    """
    Create neurogym dataset and environment.

    args:
        TASK (str): name of the task on the neurogym library
        env_kwargs (dict): task parameters
        training_kwargs (dict): training parameters

    returns:
        dataset (neurogym.Dataset): dataset object from which we can sample
        trials and labels
        env (gym.Env): task environment
    """

    # Make supervised dataset using neurogym's Dataset class
    dataset = ngym_f.Dataset(TASK,
                             env_kwargs=env_kwargs,
                             batch_size=training_kwargs['batch_size'],
                             seq_len=training_kwargs['seq_len'])
    env = dataset.env

    return dataset, env


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()

        # INSTRUCTION 1: build a recurrent neural network with a single
        # recurrent layer and rectified linear units

        self.vanilla = nn.RNN(input_size, hidden_size, nonlinearity='relu')
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # INSTRUCTION 2: get the output of the network for a given input
        out, _ = self.vanilla(x)
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


def run_agent_in_environment(num_steps, env):
    """
    Run the agent in the environment for a specified number of steps.

    Parameters
    ----------
    env :
        The environment in which the agent interacts.
    num_steps : int
        The number of steps to run the agent in the environment

    Returns
    -------
    data : dict
        A dictionary containing recorded data:
            -'ob': Observations received from the environment.
            -'actions': Actions taken by the agent.
            -'gt': Ground truth information
    perf : list
        A list containing information on performance
    rew_mat: ----------
    """
    inputs = []
    actions = []
    gt = []
    perf = []
    rew_mat = []
    trial_count = 0 
    for stp in range(int(num_steps)):
        action = env.action_space.sample()
        ob, rew, done, info = env.step(action)
        inputs.append(ob)
        actions.append(action)
        gt.append(info['gt'])
        if isinstance(rew, np.ndarray):
            rew_mat.append(rew[0])
        else:
            rew_mat.append(rew)
        if info['new_trial']:
            perf.append(info['performance'])
        else:
            perf.append(0)

    data = {'ob': np.array(inputs).astype(float),
            'actions': actions, 'gt': gt}
    return perf, rew_mat, data


def show_task(env_kwargs, data, num_steps):
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
               data['ob'], '-+', label='Fixation')
    ax[0].set_ylabel('Inputs')
    ax[0].legend()
    ax[1].plot(np.arange(1, num_steps+1)*env_kwargs['dt'],
               data['gt'], label='Targets', color='k')
    ax[1].plot(np.arange(1, num_steps+1)*env_kwargs['dt'],
               data['actions'], label='Choice', linestyle='--')
    ax[1].set_ylabel('Actions / Targets')
    ax[1].legend()
    ax[2].plot(np.arange(1, num_steps+1)*env_kwargs['dt'], perf, label='perf')
    ax[2].set_ylabel('Performance')
    ax[3].plot(np.arange(1, num_steps+1)*env_kwargs['dt'], rew_mat,
               label='perf')
    ax[3].set_ylabel('Reward')
    ax[3].set_xlabel('Time (ms)')


def train_network(num_epochs, net, optimizer, criterion, env, DEVICE, TASK):
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
    print('Training task ', TASK)
    running_loss = 0.0

    for i in range(num_epochs):
        # get inputs and labels and pass them to the GPU
        inputs, labels = dataset()
        inputs = np.expand_dims(inputs, axis=2)
        inputs = torch.from_numpy(inputs).type(torch.float).to(DEVICE)
        labels = torch.from_numpy(labels.flatten()).type(torch.long).to(DEVICE)
        # print shapes of inputs and labels
        if i == 0:
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
        if i % 200 == 199:
            print('{:d} loss: {:0.5f}'.format(i + 1, running_loss / 200))
            running_loss = 0.0

            # save current state of network's parameters
            torch.save(net.state_dict(), get_modelpath(TASK) / 'net.pth')
    print('Finished training')


def evaluate_network(net, env, num_trials, DEVICE):
    """
    Evaluate the neural network on the specified environment.

    Parameters
    ----------
    net :
        The neural network model to be evaluated.
    env :
        The environment in which the network will be evaluated.
    num_trials : int
        The number of trials for evaluation.
    DEVICE :
        The device to be used for computation.

    Returns
    -------
    None.

    """
    # Since we will not train the network anymore, we can turn off the gradient
    # computation. The most common way to do this is to use the context manager
    # torch.no_grad() as follows:
    with torch.no_grad():
        net = Net(input_size=training_kwargs['net_kwargs']['input_size'],
                  hidden_size=training_kwargs['net_kwargs']['hidden_size'],
                  output_size=training_kwargs['net_kwargs']['action_size'])

        net = net.to(DEVICE)  # pass to GPU for running forwards steps

        # load the trained network's weights from the saved file
        net.load_state_dict(torch.load(get_modelpath(TASK) / 'net.pth'))

        # empty lists / dataframe to store activity, choices, and trial
        # inforation
        activity = list()
        obs = list()
        actions = list()
        gt = list()
        info = pd.DataFrame()
        for i in range(num_trials):
            # create new trial
            env.new_trial()
            # read out the inputs in that trial
            inputs = env.ob[:, np.newaxis, np.newaxis]
            inputs = torch.from_numpy(inputs).type(torch.float)
            # as before you can print the shapes of the variables to understand
            # what they are and how to use them
            # do this for the rest of the variables as you build the code
            if i == 0:
                print('Shape of inputs: ' + str(inputs.shape))
            # INSTRUCTION 7: get the network's prediction for the current input
            action_pred, hidden = net(inputs)
            action_pred = torch.nn.functional.softmax(action_pred, dim=2)
            action_pred = action_pred.detach().numpy()

            # INSTRUCTION 8: get the network's choice.
            # Take into account the shape of action_pred. Remember that the
            # network makes a prediction for each time step in the trial.
            # Which is the prediction we really care about when evaluating the
            # network's performance?
            actions_trial = np.argmax(action_pred[:, 0], axis=1)

            # INSTRUCTION 9: check if the choice is correct
            # Again, which is the target we want when evaluating the network's
            # performance?
            choice = actions_trial[-1]
            correct = choice == env.gt[-1]

            # Log trial info
            trial_info = env.trial
            trial_info['probs'] = [trial_info['probs']]
            # write choices and outcome
            trial_info.update({'correct': correct, 'choice': choice})
            trial_info = pd.DataFrame(trial_info, index=[0])
            info = pd.concat([info, trial_info], ignore_index=True)

            # Log activity
            activity.append(np.array(hidden)[:, 0])
            # log actions
            actions.append(actions_trial)
            gt.append(env.gt)

            # Log the inputs (or observations) received by the network
            obs.append(env.ob)
        obs = np.array(equalize_arrays(obs))
        activity = np.array(equalize_arrays(activity))
        actions = np.array(equalize_arrays(actions))
        gt = np.array(equalize_arrays(gt))

        print('Average performance', np.mean(info['correct']))
        # print stats of the activity: max, min, mean, std
        print('Activity stats:')
        print('Max: ' + str(np.max(activity)) +
              ', Min: ' + str(np.min(activity)) +
              ', Mean: ' + str(np.mean(activity)) +
              ', Std: ' + str(np.std(activity)) +
              ', Shape: ' + str(activity.shape))

        # print the variables in the info dataframe
        print('Info dataframe:')
        print(info.head())
        return activity, obs, actions, gt, info


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


# --- MAIN
if __name__ == '__main__':
    plt.close('all')
    # Set up config:
    training_kwargs = {'dt': 100,
                       'lr': 1e-2,
                       'n_epochs': 2000,
                       'batch_size': 16,
                       'seq_len': 100,
                       'TASK': TASK}

    env_kwargs = {'dt': training_kwargs['dt'], 'probs': np.array([0.2, 0.8]),
                  'blk_dur': 50}

    # call function to sample
    dataset, env = get_dataset(
        TASK=TASK, env_kwargs=env_kwargs, training_kwargs=training_kwargs)

    inputs, labels = dataset()
    print('inputs shape:', inputs.shape)
    print('labels shape:', labels.shape)
    print('Example inputs:')
    print('Fixation     Stimulus Left Stimulus Right')
    print(inputs[:20, 0])
    print('Example labels:')
    print(labels[:20, 0])

    num_steps = 400

    perf, rew_mat, data = run_agent_in_environment(num_steps=num_steps,
                                                   env=env)

    show_task(env_kwargs=env_kwargs, data=data, num_steps=num_steps)

    num_neurons = 64

    net_kwargs = {'hidden_size': num_neurons,
                  'action_size': env.action_space.n,
                  'input_size': env.observation_space.n}

    net = Net(input_size=env.observation_space.n,
              hidden_size=net_kwargs['hidden_size'],
              output_size=env.action_space.n)

    # Move network to the device (CPU or GPU)
    net = net.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=training_kwargs['lr'])

    training_kwargs['env_kwargs'] = env_kwargs
    training_kwargs['net_kwargs'] = net_kwargs

    # Save config
    # with open(get_modelpath(TASK) / 'config.json', 'w') as f:
    #     json.dump(training_kwargs, f)

    num_epochs = training_kwargs['n_epochs']

    train_network(num_epochs=num_epochs, net=net, optimizer=optimizer,
                  criterion=criterion, env=env, DEVICE=DEVICE,
                  TASK=TASK)

    # load configuration file - we might have run the training on the cloud
    # and might now open the results locally
    # with open(get_modelpath(TASK) / 'config.json') as f:
    #     config = json.load(f)

    # Environment
    env = gym.make(TASK, **training_kwargs['env_kwargs'])
    env.reset(no_step=True)  # this is to initialize the environment

    num_trials = 1000
    activity, obs, actions, gt, info = evaluate_network(net=net, env=env,
                                                        num_trials=num_trials,
                                                        DEVICE=DEVICE)

    clean_minmax_activity = preprocess_activity(activity)
    plot_activity(activity=clean_minmax_activity, obs=obs, actions=actions,
                  gt=gt, config=training_kwargs, trial=0)
