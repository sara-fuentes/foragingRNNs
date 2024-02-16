#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:46:44 2020

@author: molano
"""

import numpy as np
# each Node has 4 GPUs, each with 40 cpus
# here we do: 8 seed x 2 n_ch x 10 num_cpus
# num_thrds is the number of agents to run

# this simulations are an update on the ones in bernstein_diff_blocks_durs:

# 1. They fix the aberage block duration to 80 trials (ctx_ch_prob = 0.0125)
# 2. Stimulus can last 100 or 200 ms (1 or 2 timesteps) instead of being fixed to 100 ms
# 3. run_time is increased from 40 to 48 so the test_2AFC_all tests can be run. 
# 4. The bsc_run.py has been modified so the models are saved more often.
# 5. The simulations correspond to the new NeuroGym version in which the observation 
#    is provided in advance so the agent can respond to it.
explore = {'seed': np.arange(0, 8),
           'n_ch': [2],
           'alpha_net': [0.5],
           'rollout': [30],
           'rank':[1,2,3]}

# other
experiment     = 'vanilla_RNNs'
general_params = {'seed': None, 'alg': None, 'train_mode': 'SL', # seed None alg None
                  'task': 'Foraging_Blocks-v0', 'n_lstm': 512,#64,#512,#64,
                  'num_trials': 1000000, 'num_cpu': 20, 'rollout': None,## rollout: None
                  'run_time': 48, 'num_thrds': 20, 'vanilla': False,
                  'alpha_net': None, 'sigma_net': 0.005, 'rank':None} # alpha_net: None
#
algs = {'A2C': {}, 'ACER': {}, 'ACKTR': {}, 'SL': {}, 'PPO2': {'nminibatches': 4}}




# task
task_kwargs = {'Foraging_Blocks-v0': {'n_ch': None, 'ob_nch': False, 'stim_scale': 0.5,
                                      'zero_irrelevant_stim': True, 'rewards': {'abort': 0.0},#{'abort': -0.1},
                                                   'timing': { 'fixation': ('truncated_exponential', [1600, 1450, 1600]),#('constant', 500),#
                                                              'stimulus': ('constant',300),
                                                              'decision': ('constant',100)}}}
                                                              # n_ch: None



# wrappers
wrapps = {'TrialHistoryEv-v0': {'probs': 0.8, 'predef_tr_mats': True,
                                'ctx_ch_prob': 0.0125, 'death_prob': 0.00000001},
          'Variable_nch-v0': {'block_nch': 5000, 'prob_12': 0.01,
                              'sorted_ch': True},
          # 'PassAction-v0': {},
          'PassReward-v0': {},
          'BiasCorrection-v0': {'choice_w': 100},  # used for N=2, N=4, seed=1
          'MonitorExtended-v0': {'folder': '', 'sv_fig': True, 'sv_per': 100000,
                                 'fig_type': 'svg'}}  # XXX: monitor always last

test_kwargs = {'/test_2AFC/': {'test_retrain': 'test', 'sv_per': 100000,
                               'num_steps': 1000000,
                               'sv_values': True, 'rerun': False,
                               'wrappers': {'Variable_nch-v0': {'block_nch': 10**9,
                                                                'prob_12': 1}}}}

# '/retrain/': {'test_retrain': 'retrain', 'sv_per': 100000,
#               'num_steps': 20000000, 'sv_values': False,
#               'rerun': False}

sl_kwargs = {
    'steps_per_epoch': 100,
    'lr': 0.0005,#0.01,#lrank0.0001,#later#0.0001,#faster#
    'btch_s': 8,
    'loss': 'categorical_crossentropy',
    'stateful': True,
}


