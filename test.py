import sys
sys.path.append('C:/Users/saraf/anaconda3/Lib/site-packages')
sys.path.append('C:/Users/saraf')
# packages to save data
import os
from pathlib import Path
import json

# packages to handle data
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf


# packages to visualize data
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.model_selection as sklms
import sklearn.discriminant_analysis as sklda

# import gym and neurogym to create tasks
import gym
import neurogym as ngym
# from neurogym.utils import plotting

# import torch and neural network modules to build RNNs
import torch
import torch.nn as nn

# check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# name of the task on the neurogym library
envid = 'PerceptualDecisionMaking-v0'


def get_modelpath(envid):
    # Make a local file directories
    path = Path('.') / 'files'
    os.makedirs(path, exist_ok=True)
    path = path / envid
    os.makedirs(path, exist_ok=True)
    return path


def get_dataset(envid, env_kwargs, training_kwargs):
    """
    Create neurogym dataset and environment.

    args:
        envid (str): name of the task on the neurogym library
        env_kwargs (dict): task parameters
        training_kwargs (dict): training parameters
    
    returns:
        dataset (neurogym.Dataset): dataset object from which we can sample trials and labels
        env (gym.Env): task environment
    """

    # Make supervised dataset using neurogym's Dataset class
    dataset = ngym.Dataset(envid, 
                           env_kwargs=env_kwargs, 
                           batch_size=training_kwargs['batch_size'],
                           seq_len=training_kwargs['seq_len'])
    env = dataset.env
    
    return dataset, env



class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        
        # INSTRUCTION 1: build a recurrent neural network with a single recurrent layer and rectified linear units
        self.vanilla = nn.RNN(input_size, hidden_size, nonlinearity='relu')
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # INSTRUCTION 2: get the output of the network for a given input
        out, _ = self.vanilla(x)
        x = self.linear(out)
        return x, out


def plot_activity(activity, obs, config, trial):

    # Load and preprocess results
    f, ax = plt.subplots(figsize=(5, 4), nrows=2, dpi=150)

    # time in ms
    t_plot = np.arange(activity.shape[1]) * config['dt']

    # plot the observations for one trial. Note that we will visualize the inputs as a matrix instead of traces, as we have done before.
    im = ax[0].imshow(obs[trial].T, aspect='auto', vmin=0, vmax=1)
    ax[0].set_title('Observations')
    ax[0].set_ylabel('Stimuli')

    # change the xticks to show time in ms
    ax[0].set_xticks(np.arange(0, activity.shape[1], 10))
    ax[0].set_xticklabels(t_plot[::10])
    plt.colorbar(im, ax=ax[0])
    # INSTRUCTION 11: plot the activity for one trial
    im = ax[1].imshow(activity[trial].T, aspect='auto', cmap='viridis')
    ax[1].set_title('Activity')
    ax[1].set_xlabel('Time (ms)')
    ax[1].set_ylabel('Neurons')
    plt.colorbar(im, ax=ax[1])
    # change the xticks to show time in ms
    ax[1].set_xticks(np.arange(0, activity.shape[1], 10))
    ax[1].set_xticklabels(t_plot[::10])

    plt.tight_layout()


def analysis_activity_by_condition(activity, info, config, conditions=['choice']):
    """
    Plot single neuron activity by condition.
    """   
    for condition in conditions:
        values = pd.unique(info[condition])
        f, ax = plt.subplots(figsize=(10, 3), ncols=len(values), sharex=True, dpi=150)
        t_plot = np.arange(activity.shape[1]) * config['dt']
        for i_v, value in enumerate(values):
            # INSTRUCTION 13: plot the average activity across neurons and trials for each condition
            a = activity[info[condition] == value]
            ax[i_v].imshow(a.mean(axis=0).T, aspect='auto', cmap='viridis')
            ax[i_v].set_xlabel('Time (ms)')
            ax[i_v].set_ylabel('Mean activity for ' + condition + ' = ' + str(value))
            # change the xticks to show time in ms
            ax[1].set_xticks(np.arange(0, activity.shape[1], 10))
            ax[1].set_xticklabels(t_plot[::10])

        # plt.legend(title=condition, loc='center left', bbox_to_anchor=(1.0, 0.5))


# --- MAIN
if __name__ == '__main__':
    # Set up config:
    training_kwargs = {'dt': 100,
                    'lr': 1e-2,
                    'n_epochs': 2000,
                    'batch_size': 16,
                    'seq_len': 100,
                    'envid': envid}


    # Set up task parameters
    if envid == 'PerceptualDecisionMaking-v0':
        env_kwargs = {'dt': training_kwargs['dt'],
                    'timing': {'fixation': 300, 
                                'stimulus': 1000, 
                                'delay': 0, 
                                'decision': 300}, 
                    'sigma': 2,                
                    'dim_ring': 2}               
    else:
        env_kwargs = {'dt': training_kwargs['dt']}

    # call function to sample
    dataset, env = get_dataset(envid=envid, env_kwargs=env_kwargs, training_kwargs=training_kwargs)

    inputs, labels = dataset()
    print('inputs shape:', inputs.shape)
    print('labels shape:', labels.shape)
    print('Example inputs:')
    print('Fixation     Stimulus Left Stimulus Right')
    print(inputs[:20, 0, :])
    print('Example labels:')
    print(labels[:20, 0])

    mpl.rcParams['font.family'] = ['DejaVu Serif']
    num_steps = 40
    inputs = []
    actions = []
    gt = []
    perf = []
    trial_count = 0
    for stp in range(int(num_steps)):
        action = env.action_space.sample()
        # Yoy can also try to set the action to one constant value, e.g. action = 1 
        ob, rew, done, info = env.step(action)
        inputs.append(ob)
        actions.append(action)
        gt.append(info['gt'])
        if info['new_trial']:
            perf.append(info['performance'])
        else:
            perf.append(0)


    data = {'ob': np.array(inputs).astype(float),
            'actions': actions, 'gt': gt}
    # Plot
    f, ax = plt.subplots(ncols=1, nrows=3, figsize=(8, 4), dpi=150, sharex=True)

    ax[0].plot(np.arange(1, num_steps+1)*env_kwargs['dt'], data['ob'][:, 0], label='Fixation')
    ax[0].plot(np.arange(1, num_steps+1)*env_kwargs['dt'], data['ob'][:, 1], label='Stim. L.')
    ax[0].plot(np.arange(1, num_steps+1)*env_kwargs['dt'], data['ob'][:, 2], label='Stim. R.')
    ax[0].set_ylabel('Inputs')
    ax[0].legend()
    ax[1].plot(np.arange(1, num_steps+1)*env_kwargs['dt'], data['gt'], label='Targets', color='k')
    ax[1].plot(np.arange(1, num_steps+1)*env_kwargs['dt'], data['actions'], label='Choice', linestyle='--')
    ax[1].set_ylabel('Actions / Targets')
    ax[1].legend()
    ax[2].plot(np.arange(1, num_steps+1)*env_kwargs['dt'], perf, label='perf')
    ax[2].set_ylabel('Performance')
    ax[2].set_xlabel('Time (ms)')

    num_neurons = 64

    net_kwargs = {'hidden_size': num_neurons,
                'action_size': env.action_space.n,
                'input_size': env.observation_space.shape[0]} # size of the input to the network

    net = Net(input_size=env.observation_space.shape[0],
          hidden_size=net_kwargs['hidden_size'],
          output_size=env.action_space.n)

    # Move network to the device (CPU or GPU)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=training_kwargs['lr'])


    training_kwargs['env_kwargs'] = env_kwargs
    training_kwargs['net_kwargs'] = net_kwargs

    # Save config
    with open(get_modelpath(envid) / 'config.json', 'w') as f:
        json.dump(training_kwargs, f)


    print('Training task ', envid)

    num_epochs = training_kwargs['n_epochs']


    running_loss = 0.0

    for i in range(num_epochs):
    
        # get inputs and labels and pass them to the GPU
        inputs, labels = dataset()
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)
        # print shapes of inputs and labels
        if i == 0:
            print('inputs shape: ', inputs.shape)
            print('labels shape: ', labels.shape)
            print('Max labels: ', labels.max())
        # we need zero the parameter gradients to re-initialize and avoid they accumulate across epochs
        optimizer.zero_grad()

        # INSTRUCTION 3: FORWARD PASS: get the output of the network for a given input
        outputs, _ = net(inputs)

        #reshape outputs so they have the same shape as labels
        outputs = outputs.view(-1, env.action_space.n)

        #  INSTRUCTION 4: compute loss with respect to the labels
        loss = criterion(outputs, labels)
        
        # INSTRUCTION 5: compute gradients
        loss.backward()
        
        # INSTRUCTION 6: update weights
        optimizer.step()

        # print average loss over last 200 training iterations and save the current network
        running_loss += loss.item()
        if i % 200 == 199:
            print('{:d} loss: {:0.5f}'.format(i + 1, running_loss / 200))
            running_loss = 0.0
            
            # save current state of network's parameters
            torch.save(net.state_dict(), get_modelpath(envid) / 'net.pth')

    print('Finished Training')

    # load configuration file - we might have run the training on the cloud and might now open the results locally
    with open(get_modelpath(envid) / 'config.json') as f:
        config = json.load(f)


    
    # Environment
    env = gym.make(envid, **config['env_kwargs'])
    try:
        env.timing = config['env_kwargs']['timing']
    except KeyError:
        timing = {}
        for period in env.timing.keys():
            period_times = [env.sample_time(period) for _ in range(100)]
            timing[period] = np.median(period_times)
        env.timing = timing
    env.reset(no_step=True) # this is to initialize the environment 

    # Since we will not train the network anymore, we can turn off the gradient computation. The most commun way to do this is to use the context manager torch.no_grad() as follows:
    with torch.no_grad():
        net = Net(input_size=config['net_kwargs']['input_size'],
                hidden_size=config['net_kwargs']['hidden_size'],
                output_size=config['net_kwargs']['action_size'])
        
        net = net.to(device) # pass to GPU for running forwards steps
        
        # load the trained network's weights from the saved file
        net.load_state_dict(torch.load(get_modelpath(envid) / 'net.pth'))

        # how many trials to run
        num_trial = 1000
        
        # empty lists / dataframe to store activity, choices, and trial information
        activity = list()
        obs = list()
        info = pd.DataFrame()
        
    for i in range(num_trial):

        # create new trial
        env.new_trial()
        
        # read out the inputs in that trial
        inputs = torch.from_numpy(env.ob[:, np.newaxis, :]).type(torch.float)
        # as before you can print the shapes of the variables to understand what they are and how to use them
        # do this for the rest of the variables as you build the code
        if i == 0:
            print('Shape of inputs: ' + str(inputs.shape))
        # INSTRUCTION 7: get the network's prediction for the current input
        action_pred, hidden = net(inputs)
        action_pred = action_pred.detach().numpy()
        
        # INSTRUCTION 8: get the network's choice. 
        # Take into account the shape of action_pred. Remember that the network makes a prediction for each time step in the trial.
        # Which is the prediction we really care about when evaluating the network's performance?
        choice = np.argmax(action_pred[-1, 0, :])
        
        # INSTRUCTION 9: check if the choice is correct
        # Again, which is the target we want when evaluating the network's performance?
        correct = choice == env.gt[-1]

        # Log trial info
        trial_info = env.trial
        trial_info.update({'correct': correct, 'choice': choice}) # write choices and outcome
        info = info._append(trial_info, ignore_index=True)
        
        # Log activity
        activity.append(np.array(hidden)[:, 0, :])
        
        # Log the inputs (or observations) received by the network
        obs.append(env.ob)

    print('Average performance', np.mean(info['correct']))

    activity = np.array(activity)
    obs = np.array(obs)

    # print stats of the activity: max, min, mean, std
    print('Activity stats:')
    print('Max: ' + str(np.max(activity)) + \
        ', Min: ' + str(np.min(activity)) + \
        ', Mean: ' + str(np.mean(activity)) + \
        ', Std: ' + str(np.std(activity)) + \
            ', Shape: ' + str(activity.shape))

    
    # print the variables in the info dataframe
    print('Info dataframe:')
    print(info.head())

    # Plot the psychometric curve. You can use the function you wrote in the first tutorial.

    
    # plot the probability of choosing right as a function of the signed coherence and then fit a psychometric curve to the data.
    mpl.rcParams['font.family'] = ['DejaVu Serif']
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

    f, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=150)
    choice = info['choice'].values
    # translate choice to 0 and 1
    choice_01 = np.copy(choice)
    choice_01 -= 1
    gt = info['ground_truth'].values
    coherence = info['coh'].values
    # get signed coherence
    signed_coherence = np.copy(coherence)
    signed_coherence[gt == 0] = -signed_coherence[gt == 0]
    # INSTRUCTION 10: plot the probability of choosing right as a function of the signed coherence
    for sc in signed_coherence:
        prob_right = np.mean(choice_01[signed_coherence == sc])
        std_right = np.std(choice_01[signed_coherence == sc])/np.sqrt(np.sum(signed_coherence == sc))
        ax.errorbar(sc, prob_right, yerr=std_right, color='k')
        ax.plot(sc, prob_right, 'o', color='k')
        ax.set_xlabel('Signed coherence')
        ax.set_ylabel('P(right)')
    # fit psychometric curve
    pars, _ = curve_fit(probit, signed_coherence, choice_01, p0=[0, 1])
    x = np.linspace(-50, 50, 100)
    ax.plot(x, probit(x, *pars), color='k')
    plt.show()

    plot_activity(activity=activity, obs=obs, config=config, trial=0)

    
    silent_idx = np.where(activity.sum(axis=(0, 1))==0)[0]

    print('fraction of silent neurons:', len(silent_idx)/activity.shape[-1])
    # INSRTUCTION 12: plot the activity for one trial, but now excluding the silent neurons
    clean_activity = activity[:,:,np.delete(np.arange(activity.shape[-1]), silent_idx)]
    plot_activity(activity=clean_activity, obs=obs, config=config, trial=0)

    
    # min_max scaling
    minmax_activity = np.array([neuron-neuron.min() for neuron in clean_activity.transpose(2,0,1)]).transpose(1,2,0)
    minmax_activity = np.array([neuron/neuron.max() for neuron in minmax_activity.transpose(2,0,1)]).transpose(1,2,0)

    plot_activity(activity=minmax_activity, obs=obs, config=config, trial=0)

    analysis_activity_by_condition(minmax_activity, info, config, conditions=['choice']) # other conditions: correct, ground_truth

    
   
    # number of CV splits
    n_splits = 100

    # set
    mean_acc = np.zeros([n_splits, minmax_activity.shape[1]]) * np.nan

    for i in range(n_splits):    
        
    # transpose tensor to be shape [trials, time, neurons]
        for xi,x in enumerate(minmax_activity.transpose(1,0,2)):

            # INSTRUCTION 14: split data into train and test sets using sklms.
            x_train, x_test, y_train, y_test = sklms.train_test_split(x, info.ground_truth.values, random_state=i)

            # INSTRUCTION 15: fit a linear discriminant analysis model to the training data using sklda
            lda_fitted = sklda.LinearDiscriminantAnalysis(solver='lsqr').fit(X=x_train, y=y_train)

            # INSTRUCTION 16: predict the labels for the test data
            y_pred = lda_fitted.predict(x_test)

            # INSTRUCTION 17: compute the accuracy of the model
            correct = 1 - np.abs(y_pred - y_test)

            mean_acc[i,xi] = correct.mean()

    
    # calculate 95% CI
    ci_acc = np.percentile(mean_acc, [5,95], axis=0)


    
    # for plotting: time axis, stim and resp times
    t_plot = np.arange(activity.shape[1]) * config['dt']
    stim_onset = t_plot[np.where(obs[0,:,1]!=0)[0][0]]
    resp_onset= t_plot[np.where(obs[0,:,0]!=1)[0][0]]

    # plot linear classification accuracy
    plt.figure(figsize=(4,3), dpi=150)
    plt.plot(t_plot, np.zeros(mean_acc.shape[1])+.5, 'k--', alpha=.2)
    plt.plot(stim_onset, .48, '^', color = 'r', ms=10)
    plt.plot(resp_onset, .48, '^', color='b', ms=10)
    plt.plot(t_plot, np.mean(mean_acc, axis=0), 'k')
    plt.fill_between(t_plot, ci_acc[0], ci_acc[1], color='k', alpha=.2)
    plt.ylabel('classification accuracy')
    plt.xlabel('time')
    plt.xlim(t_plot[0],t_plot[-1])


    
    mean_acc = np.zeros([len(np.unique(info.coh.values)), n_splits, minmax_activity.shape[1]])

    for ci,c in enumerate(np.unique(info.coh.values)):
        
        # INSTRUCTION 18: get the indices of the trials with the current coherence
        cidx = np.where(info.coh.values==c)
        
        for i in range(n_splits):    

            # transpose tensor to be shape [trials, time, neurons]
            for xi,x in enumerate(minmax_activity[cidx].transpose(1,0,2)):

                # train-test-split
                x_train, x_test, y_train, y_test = sklms.train_test_split(x, info.ground_truth.values[cidx], random_state=i)

                # fit to train data
                lda_fitted = sklda.LinearDiscriminantAnalysis(solver='lsqr').fit(X=x_train, y=y_train)

                # predict test set labels
                y_pred = lda_fitted.predict(x_test)

                # is the response correct for each trial?
                correct = 1 - np.abs(y_pred - y_test)

                mean_acc[ci,i,xi] = correct.mean()

    
    # colors corresponding to different values of color gradient
    colors = plt.get_cmap('magma')(np.linspace(0.1,.9, len(np.unique(info.coh.values))))

    # plot linear classification accuracy
    plt.figure(figsize=(4,3), dpi=120)

    # plot mean acc for each coherence level
    for ci in range(len(np.unique(info.coh.values))):
        plt.plot(t_plot, np.mean(mean_acc[ci], axis=0), color=colors[ci], label=np.unique(info.coh.values)[ci])

    plt.plot(t_plot, np.zeros(mean_acc[ci].shape[1])+.5, 'k--', alpha=.2)    
    plt.plot(stim_onset, .48, '^', color = 'k', ms=10)
    plt.plot(resp_onset, .48, '^', color='k', ms=10)
    plt.ylabel('classification accuracy')
    plt.xlabel('time')
    plt.xlim(t_plot[0], t_plot[-1])
    plt.legend(frameon=False)
