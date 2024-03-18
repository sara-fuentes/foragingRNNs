import torch
import torch.nn as nn
import ngym_foraging as ngym_f
import gym
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
sys.path.append('C:/Users/saraf/OneDrive/Documentos/GitHub/foragingRNNs')
import forage_training as ft

TASK = 'ForagingBlocks-v0'
TRAINING_KWARGS = {'dt': 100,
                   'lr': 1e-2,
                   'n_epochs': 20,
                   'batch_size': 16,
                   'seq_len': 200,
                   'TASK': TASK}

num_steps_exp = TRAINING_KWARGS['seq_len']*TRAINING_KWARGS['batch_size']


# --- MAIN
if __name__ == '__main__':
    plt.close('all')

    num_neurons = 64

    env = gym.make(TASK, **TRAINING_KWARGS['env_kwargs'])

    net_kwargs = {'hidden_size': num_neurons,
                  'action_size': env.action_space.n,
                  'input_size': env.observation_space.n+1+1}
    
    env_kwargs = {'dt': TRAINING_KWARGS['dt'], 'probs': np.array([0, 1]),
                  'blk_dur': 20, 'timing':
                      {'ITI': ngym_f.random.TruncExp(200, 100, 300),
                       'fixation': 200, 'decision': 200}}  # Decision period}

    # call function to sample
    dataset, env = ft.get_dataset(TASK=TASK, env_kwargs=env_kwargs)

    net = ft.Net(input_size=net_kwargs['input_size'],
              hidden_size=net_kwargs['hidden_size'],
              output_size=env.action_space.n)

    net.load_state_dict(torch.load('C:/Users/saraf/OneDrive/Documentos/IDIBAPS/foraging RNNs/nets/net.pth'))

    data = ft.run_agent_in_environment(env=env, net=net, num_steps_exp=num_steps_exp)
