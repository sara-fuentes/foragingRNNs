#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 07:26:31 2020

@author: manuel

"""
import os
import sys
import numpy as np
import glob
import importlib
import argparse
from copy import deepcopy as deepc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# from torch.nn import init
# from torch.nn import functional as F
# import math
# import torch.optim as optim
import time
import random
# import datetime

# import get_activity_lr_bf30Apr as ga
import gym
import ngym_foraging as ngym_f
# import ngym_priors as ngym_p  # need to import it so ngym envs are registered
# from neurogym.utils import plotting
from neurogym.wrappers import ALL_WRAPPERS
# from ngym_priors.wrappers import ALL_WRAPPERS as all_wrpps_p
# from stable_baselines.common.policies import LstmPolicy
# from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
# from stable_baselines.common.vec_env import SubprocVecEnv
# from stable_baselines.common import set_global_seeds
# from stable_baselines.common.callbacks import CheckpointCallback

from math import sqrt
# import time
from utils import rest_arg_parser
from utils import get_name_and_command_from_dict as gncfd


# ALL_WRAPPERS.update(all_wrpps_p)

def set_global_seeds(seed):
    """
    set the seed for python random, tensorflow, numpy and gym spaces

    :param seed: (int) the seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # prng was removed in latest gym version
    if hasattr(gym.spaces, 'prng'):
        gym.spaces.prng.seed(seed)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def order_by_sufix(file_list):
    file_list = [os.path.basename(x) for x in file_list]
    flag = 'model.zip' in file_list
    file_list = [x for x in file_list if x != 'model.zip']
    sfx = [int(x[x.find('_')+1:x.rfind('_')]) for x in file_list]
    sorted_list = [x for _, x in sorted(zip(sfx, file_list))]
    if flag:
        sorted_list.append('model.zip')
    return sorted_list, np.max(sfx)


def loss_mse(output, target, mask):
    """
    Mean squared error loss
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: idem
    :param mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    # Compute loss for each (trial, timestep) (average accross output dimensions)
    loss_tensor = (mask * (target - output)).pow(2).mean(dim=-1)
    # Account for different number of masked values per trial
    loss_by_trial = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1)
    return loss_by_trial.mean()


def scratch_model(xstim, noise_std, r, h, wi, wrec, b, wo, alpha, hs):
    """
    RNN from scratch
    :param : numpy structure
    :return: firing rate; hidden state; output
    """
    if r is None:
        r = np.tanh(h.copy())
    ns = np.random.randn(hs)*noise_std
    h = h+ns+alpha*(-h+(r@wrec.T)+xstim@wi)
    r = np.tanh(h+b)
    out = np.tanh(h)@wo
    return r, h, out


def scratch_model_lr(xstim, noise_std, r, h, wi, m, n, b, wo, alpha, hs):
    if r is None:
        r = np.tanh(h.copy())
    ns = np.random.randn(hs)*noise_std
    h = h+ns+alpha*(-h+(r@n)@m.T/hs+xstim@wi)
    r = np.tanh(h+b)
    out = np.tanh(h)@wo/hs
    return r, h, out


# TODO rename biases train_biases, add to cloning function (important !!!!)
class FullRankRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha=0.2,
                 rho=1, train_wi=False, train_wo=False, train_wrec=True,
                 train_h0=False, train_si=True, train_so=True, wi_init=None,
                 wo_init=None, wrec_init=None, si_init=None, so_init=None,
                 b_init=None, add_biases=False, non_linearity=torch.tanh,
                 output_non_linearity=torch.tanh):
        """
        :param input_size: int
        :param hidden_size: int
        :param output_size: int
        :param noise_std: float
        :param alpha: float, value of dt/tau
        :param rho: float, std of gaussian distribution for initialization
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_h0: bool
        :param wi_init: torch tensor of shape (input_dim, hidden_size)
        :param wo_init: torch tensor of shape (hidden_size, output_dim)
        :param wrec_init: torch tensor of shape (hidden_size, hidden_size)
        :param si_init: input scaling, torch tensor of shape (input_dim)
        :param so_init: output scaling, torch tensor of shape (output_dim)
        """
        super(FullRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.rho = rho
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_h0 = train_h0
        self.train_si = train_si
        self.train_so = train_so
        self.non_linearity = non_linearity
        self.output_non_linearity = output_non_linearity

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.si = nn.Parameter(torch.Tensor(input_size))
        if train_wi:
            self.si.requires_grad = False
        else:
            self.wi.requires_grad = False
        if not train_si:
            self.si.requires_grad = False
        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_wrec:
            self.wrec.requires_grad = False
        self.b = nn.Parameter(torch.Tensor(hidden_size))
        if not add_biases:
            self.b.requires_grad = False
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.so = nn.Parameter(torch.Tensor(output_size))
        if train_wo:
            self.so.requires_grad = False
        if not train_wo:
            self.wo.requires_grad = False
        if not train_so:
            self.so.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_()
            else:
                self.wi.copy_(wi_init)
            if si_init is None:
                self.si.set_(torch.ones_like(self.si))
            else:
                self.si.copy_(si_init)
            if wrec_init is None:
                self.wrec.normal_(std=rho / sqrt(hidden_size))
            else:
                self.wrec.copy_(wrec_init)
            if b_init is None:
                self.b.zero_()
            else:
                self.b.copy_(b_init)
            if wo_init is None:
                self.wo.normal_(std=1 / hidden_size)
            else:
                self.wo.copy_(wo_init)
            if so_init is None:
                self.so.set_(torch.ones_like(self.so))
            else:
                self.so.copy_(so_init)
            self.h0.zero_()
        self.wi_full, self.wo_full = [None] * 2
        self._define_proxy_parameters()

    def _define_proxy_parameters(self):
        self.wi_full = (self.wi.t() * self.si).t()
        self.wo_full = self.wo * self.so

    def forward(self, input, return_dynamics=False, initial_states=None):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: bool
        :param initial_states: None or torch tensor of shape
        (batch_size, hidden_size) of initial state vectors for each trial if
        desired
        :return: if return_dynamics=False, output tensor of shape
                (batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, output tensor, trajectories tensor
                 of shape (batch_size, #timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        if initial_states is None:
            initial_states = self.h0
        h = initial_states.clone()
        r = self.non_linearity(initial_states)
        self._define_proxy_parameters()
        noise = torch.randn(batch_size, seq_len,
                            self.hidden_size, device=self.wrec.device)
        output = torch.zeros(batch_size, seq_len,
                             self.output_size, device=self.wrec.device)
        if return_dynamics:
            trajectories = torch.zeros(
                batch_size, seq_len + 1, self.hidden_size, device=self.wrec.device)
            trajectories[:, 0, :] = h

        # simulation loop
        for i in range(seq_len):
            h = h + self.noise_std * noise[:, i, :] + self.alpha * \
                (-h + r.matmul(self.wrec.t()) +
                 input[:, i, :].matmul(self.wi_full))
            r = self.non_linearity(h + self.b)
            output[:, i, :] = self.output_non_linearity(h) @ self.wo_full

            if return_dynamics:
                trajectories[:, i + 1, :] = h

        if not return_dynamics:
            return output, h
        else:
            return output, h, trajectories

    def clone(self):
        new_net = FullRankRNN(self.input_size, self.hidden_size, self.output_size,
                              self.noise_std, self.alpha, self.rho, self.train_wi,
                              self.train_wo, self.train_wrec, self.train_h0,
                              self.train_si, self.train_so, self.wi, self.wo,
                              self.wrec, self.si, self.so, self.b, False,
                              self.non_linearity, self.output_non_linearity)
        return new_net


class LowRankRNN(nn.Module):
    """
    This class implements the low-rank RNN. Instead of being parametrized by
    an NxN connectivity matrix, it is parametrized by two Nxr matrices m and n
    such that the connectivity is m * n^T
    """

    def __init__(self, input_size, hidden_size, output_size, noise_std, alpha,
                 rank=1, train_wi=False, train_wo=False, train_wrec=True,
                 train_h0=False, train_si=False, train_so=False, wi_init=None,
                 wo_init=None, m_init=None, n_init=None, si_init=None,
                 so_init=None, h0_init=None, add_biases=False,
                 non_linearity=torch.tanh, output_non_linearity=torch.tanh):
        """
        :param input_size: int
        :param hidden_size: int
        :param output_size: int
        :param noise_std: float
        :param alpha: float, value of dt/tau
        :param rank: int
        :param train_wi: bool
        :param train_wo: bool
        :param train_wrec: bool
        :param train_h0: bool
        :param train_si: bool
        :param train_so: bool
        :param wi_init: torch tensor of shape (input_dim, hidden_size)
        :param wo_init: torch tensor of shape (hidden_size, output_dim)
        :param m_init: torch tensor of shape (hidden_size, rank)
        :param n_init: torch tensor of shape (hidden_size, rank)
        :param si_init: torch tensor of shape (input_size)
        :param so_init: torch tensor of shape (output_size)
        :param h0_init: torch tensor of shape (hidden_size)
        """
        super(LowRankRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std = noise_std
        self.alpha = alpha
        self.rank = rank
        self.train_wi = train_wi
        self.train_wo = train_wo
        self.train_wrec = train_wrec
        self.train_h0 = train_h0
        self.train_si = train_si
        self.train_so = train_so
        self.non_linearity = non_linearity
        self.output_non_linearity = output_non_linearity

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.si = nn.Parameter(torch.Tensor(input_size))
        if train_wi:
            self.si.requires_grad = False
        else:
            self.wi.requires_grad = False
        if not train_si:
            self.si.requires_grad = False
        self.m = nn.Parameter(torch.Tensor(hidden_size, rank))
        self.n = nn.Parameter(torch.Tensor(hidden_size, rank))
        if not train_wrec:
            self.m.requires_grad = False
            self.n.requires_grad = False
        self.b = nn.Parameter(torch.Tensor(hidden_size))
        if not add_biases:
            self.b.requires_grad = False
        self.wo = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.so = nn.Parameter(torch.Tensor(output_size))
        if train_wo:
            self.so.requires_grad = False
        if not train_wo:
            self.wo.requires_grad = False
        if not train_so:
            self.so.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False

        # Initialize parameters
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_()
            else:
                self.wi.copy_(wi_init)
            if si_init is None:
                self.si.set_(torch.ones_like(self.si))
            else:
                self.si.copy_(si_init)

            if m_init is None:
                self.m.normal_()
            else:
                self.m.copy_(m_init)
            if n_init is None:
                self.n.normal_()
            else:
                self.n.copy_(n_init)
            self.b.zero_()     # TODO add biases initializer

            if wo_init is None:
                self.wo.normal_(std=4.)
            else:
                self.wo.copy_(wo_init)
            if so_init is None:
                self.so.set_(torch.ones_like(self.so))
            else:
                self.so.copy_(so_init)

            if h0_init is None:
                self.h0.zero_()
            else:
                self.h0.copy_(h0_init)
        self.wrec, self.wi_full, self.wo_full = [None] * 3
        self._define_proxy_parameters()

    def _define_proxy_parameters(self):
        # For optimization purposes the full connectivity matrix is
        # never computed explicitly
        self.wrec = None
        self.wi_full = (self.wi.t() * self.si).t()
        self.wo_full = self.wo * self.so

    def forward(self, input, return_dynamics=False, initial_states=None):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_dynamics: boolean
        :return: if return_dynamics=False, output tensor of shape
                (batch_size, #timesteps, output_dimension)
                 if return_dynamics=True, output tensor, trajectories tensor
                 of shape (batch_size, #timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        if initial_states is None:
            initial_states = self.h0
        h = initial_states.clone()
        r = self.non_linearity(h)
        self._define_proxy_parameters()
        noise = torch.randn(batch_size, seq_len,
                            self.hidden_size, device=self.m.device)
        output = torch.zeros(batch_size, seq_len,
                             self.output_size, device=self.m.device)
        if return_dynamics:
            trajectories = torch.zeros(
                batch_size, seq_len + 1, self.hidden_size, device=self.m.device)
            trajectories[:, 0, :] = h
        # simulation loop
        for i in range(seq_len):
            h = h + self.noise_std * noise[:, i, :] + self.alpha * \
                (-h + r.matmul(self.n).matmul(self.m.t()) / self.hidden_size +
                    input[:, i, :].matmul(self.wi_full))
            r = self.non_linearity(h + self.b)
            output[:, i, :] = self.output_non_linearity(
                h) @ self.wo_full / self.hidden_size
            if return_dynamics:
                trajectories[:, i + 1, :] = h

        if not return_dynamics:
            return output, h  # @logYX extracting the hidden_state
        else:
            return output, h, trajectories

    def clone(self):
        new_net = LowRankRNN(self.input_size, self.hidden_size, self.output_size,
                             self.noise_std, self.alpha, self.rank, self.train_wi,
                             self.train_wo, self.train_wrec, self.train_h0,
                             self.train_si, self.train_so, self.wi, self.wo,
                             self.m, self.n, self.si, self.so, self.h0, False,
                             self.non_linearity, self.output_non_linearity)
        new_net._define_proxy_parameters()
        return new_net

    def load_state_dict(self, state_dict, strict=True):
        """
        override
        """
        if 'rec_noise' in state_dict:
            del state_dict['rec_noise']
        super().load_state_dict(state_dict, strict)
        self._define_proxy_parameters()

    def svd_reparametrization(self):
        """
        Orthogonalize m and n via SVD
        """
        with torch.no_grad():
            structure = (self.m @ self.n.t()).numpy()
            m, s, n = np.linalg.svd(structure, full_matrices=False)
            m, s, n = m[:, :self.rank], s[:self.rank], n[:self.rank, :]
            self.m.set_(torch.from_numpy(m * np.sqrt(s)))
            self.n.set_(torch.from_numpy(n.transpose() * np.sqrt(s)))
            self._define_proxy_parameters()


def train(net, dataset, _input, _target, _mask, stps_ep, n_epochs=10, lr=1e-2,
          n_ch=2, batch_size=1, plot_learning_curve=False, plot_gradient=False,
          mask_gradients=False, clip_gradient=None, early_stop=None,
          keep_best=False, cuda=False, resample=False,
          initial_states=None):
    """
    Train a network
    :param net: nn.Module
    :param _input: torch tensor of shape (num_trials, num_timesteps, input_dim)
    :param _target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param _mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :param n_epochs: int
    :param lr: float, learning rate
    :param batch_size: int
    :param plot_learning_curve: bool
    :param plot_gradient: bool
    :param mask_gradients: bool, set to True if training the
        SupportLowRankRNN_withMask for reduced models
    :param clip_gradient: None or float, if not None the value at which gradient
        norm is clipped
    :param early_stop: None or float, set to target loss value after which
        to immediately stop if attained
    :param keep_best: bool, if True, model with lower loss from training
        process will be kept (for this option, the
        network has to implement a method clone())
    :param resample: for SupportLowRankRNNs, set True
    :param initial_states: None or torch tensor of shape (batch_size, hidden_size)
    of initial state vectors if desired
    :return: nothing
    """
    print("Training...")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # criterion    = nn.CrossEntropyLoss(ignore_index=0)
    all_losses = []
    # output_size = 1

    if plot_gradient:
        gradient_norms = []

    # CUDA management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device('cpu')
        else:
            if cuda:
                device = torch.device('cuda')
                print('>>>>>>>>>>>train cuda......')
            else:
                device = torch.device(f'cuda:{cuda}')
                print('>>>>>>>>>>>>>>train cuda........')
    else:
        device = torch.device('cpu')
        print("Warning: CUDA not available on this machine, switching to CPU")
    net.to(device=device)

    dataset._inputs = torch.from_numpy(dataset._inputs).type(torch.float)
    dataset._target = torch.from_numpy(dataset._target).type(torch.float)
    dataset._mask = torch.from_numpy(dataset._mask).type(torch.float)

    dataset._inputs = dataset._inputs.to(device=device, dtype=torch.float32)
    dataset._target = dataset._target.to(device=device, dtype=torch.float32)
    dataset._mask = dataset._mask.to(device=device, dtype=torch.float32)

    inputs, labels, mask = dataset()  # get inputs and targets
    # print('inpyt.....',inputs[2,30:40,:])
    # transpose
    if initial_states is not None:
        initial_states = initial_states.to(device=device, dtype=torch.float32)

    # Initialize setup to keep best network
    with torch.no_grad():
        output, hidden = net(inputs, initial_states=initial_states)
        # print('input......',inputs)
        initial_loss = loss_mse(output, labels, mask)
        print("initial loss: %.3f" % (initial_loss.item()))
        if keep_best:
            best = net.clone()
            best_loss = initial_loss.item()
        hidden = hidden.detach()
        initial_states = hidden

    # Training loop
    n_epochs, mini_seq = n_epochs, 25  # stps_ep//2
    for epoch in range(n_epochs):  # go through the entire dataset
        begin = time.time()
        losses = []  # losses over the whole epoch
        # each epoch
        for i in range(stps_ep//mini_seq):
            batch_random_idx = np.random.choice(
                batch_size, batch_size, replace=True)
            seq_random_idx = np.random.choice(
                stps_ep-mini_seq, batch_size, replace=True)
            seq_random_idx = seq_random_idx*rollout
            initial_states = None  # init state is None for individual mini_batch
            for ii in range(mini_seq):
                inputs, labels, mask = dataset.data_augment(
                    batch_random_idx, seq_random_idx)  # get inputs and targets
                optimizer.zero_grad()
                if initial_states is not None:
                    output, hidden = net(inputs, initial_states=initial_states)
                else:
                    output, hidden = net(inputs)
                loss = loss_mse(output, labels, mask)
                all_losses.append(loss.item())
                losses.append(loss.item())
                loss.backward()
                if mask_gradients:
                    net.m.grad = net.m.grad * net.m_mask
                    net.n.grad = net.n.grad * net.n_mask
                    net.wi.grad = net.wi.grad * net.wi_mask
                    net.wo.grad = net.wo.grad * net.wo_mask
                    net.unitn.grad = net.unitn.grad * net.unitn_mask
                    net.unitm.grad = net.unitm.grad * net.unitm_mask
                    net.unitwi.grad = net.unitwi.grad * net.unitwi_mask
                if clip_gradient is not None:
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), clip_gradient)
                if plot_gradient:
                    tot = 0
                    for param in [p for p in net.parameters() if p.requires_grad]:
                        tot += (param.grad ** 2).sum()
                    gradient_norms.append(sqrt(tot))
                optimizer.step()
                # These 2 lines important to prevent memory leaks # YX
                loss.detach()
                output.detach()
                # hidden states are reused
                hidden = hidden.detach()
                initial_states = hidden

                if resample:
                    net.resample_basis()

                seq_random_idx = seq_random_idx+rollout

        # print("epoch %d:  loss=%.3f  *" % (epoch, np.mean(losses)))
        if keep_best and np.mean(losses) < best_loss:
            best = net.clone()
            best_loss = np.mean(losses)
            print("epoch %d:  loss=%.3f  (took %.2f s) *" %
                  (epoch, np.mean(losses), time.time() - begin))
        else:
            print("epoch %d:  loss=%.3f  (took %.2f s)" %
                  (epoch, np.mean(losses), time.time() - begin))

    if plot_learning_curve:
        plt.plot(all_losses)
        plt.title("Learning curve")
        plt.show()

    if plot_gradient:
        plt.plot(gradient_norms)
        plt.title("Gradient norm")
        plt.show()

    if keep_best:
        net.load_state_dict(best.state_dict())


def test_env(env, kwargs, num_steps=100):
    """Test if all one environment can at least be run."""
    env = gym.make(env, **kwargs)
    env.reset()
    for stp in range(num_steps):
        action = 0
        state, rew, done, info = env.step(action)
        if done:
            env.reset()
    return env


def apply_wrapper(env, wrap_string, params):
    wrap_str = ALL_WRAPPERS[wrap_string]
    wrap_module = importlib.import_module(wrap_str.split(":")[0])
    wrap_method = getattr(wrap_module, wrap_str.split(":")[1])
    return wrap_method(env, **params)


def arg_parser():
    """
    Create an argparse.ArgumentParser for neuro environments
    """
    aadhf = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=aadhf)
    parser.add_argument('--dt', help='time step for environment', type=int,
                        default=None)
    parser.add_argument('--folder', help='where to save the data and model',
                        type=str, default=None)
    parser.add_argument('--train_mode', help='RL or SL',
                        type=str, default=None)
    parser.add_argument('--alg', help='RL algorithm to use',
                        type=str, default=None)
    parser.add_argument('--task', help='task', type=str, default=None)
    parser.add_argument('--num_trials',
                        help='number of trials to train', type=int,
                        default=None)
    parser.add_argument('--n_lstm',
                        help='number of units in the network',
                        type=int, default=None)
    parser.add_argument('--rollout',
                        help='rollout used to train the network',
                        type=int, default=None)
    parser.add_argument('--seed', help='seed for task',
                        type=int, default=None)
    parser.add_argument('--n_ch', help='number of choices',
                        type=int, default=None)
    parser.add_argument('--vanilla',
                        help='whether to use vanilla-RNNs (default: LSTM)',
                        type=bool, default=None)
    parser.add_argument('--rank',
                        help='rank truncation of the rnn connectivity',
                        type=int, default=None)  # @@YX
    parser.add_argument('--sigma_net', help='noise added to the stimulus',
                        type=float, default=None)
    parser.add_argument('--alpha_net', help='alpha for vanilla RNN time-constant',
                        type=float, default=None)

    # n-alternative task params
    parser.add_argument('--stim_scale', help='stimulus evidence',
                        type=float, default=None)
    parser.add_argument('--sigma', help='noise added to the stimulus',
                        type=float, default=None)
    parser.add_argument('--ob_nch', help='Whether to provide num. channels',
                        type=bool, default=None)
    parser.add_argument('--ob_histblock', help='Whether to provide hist block cue',
                        type=bool, default=None)
    parser.add_argument('--zero_irrelevant_stim', help='Whether to zero' +
                        ' irrelevant stimuli', type=bool, default=None)
    parser.add_argument('--cohs',
                        help='coherences to use for stimuli',
                        type=float, nargs='+', default=None)

    # NAltConditionalVisuomotor task parameters
    parser.add_argument('--n_stims', help='number of stimuli', type=int,
                        default=None)

    # CV-learning task
    parser.add_argument('--th_stage',
                        help='threshold to change the stage',
                        type=float, default=None)
    parser.add_argument('--keep_days',
                        help='minimum number of sessions to spend on a stage',
                        type=int, default=None)
    parser.add_argument('--stages',
                        help='stages used for training',
                        type=int, nargs='+', default=None)

    # trial-hist/side-bias/persistence wrappers
    parser.add_argument('--probs', help='prob of main transition in the ' +
                        'n-alt task with trial hist.', type=float,
                        default=None)

    # trial_hist wrapper parameters
    parser.add_argument('--block_dur',
                        help='dur. of block in the trial-hist wrappr (trials)',
                        type=int, default=None)
    parser.add_argument('--num_blocks', help='number of blocks', type=int,
                        default=None)
    parser.add_argument('--rand_blcks', help='whether transition matrix is' +
                        ' built randomly', type=bool, default=None)
    parser.add_argument('--blk_ch_prob', help='prob of trans. mat. change',
                        type=float, default=None)
    parser.add_argument('--balanced_probs', help='whether transition matrix is' +
                        ' side-balanced', type=bool, default=None)

    # trial_hist evolution wrapper parameters
    parser.add_argument('--ctx_dur',
                        help='dur. of context in the trial-hist wrappr (trials)',
                        type=int, default=None)
    parser.add_argument('--num_contexts', help='number of contexts', type=int,
                        default=None)
    parser.add_argument('--ctx_ch_prob', help='prob of trans. mat. change',
                        type=float, default=None)
    parser.add_argument('--death_prob', help='prob. of starting next generation',
                        type=float, default=None)
    parser.add_argument('--fix_2AFC', help='whether 2AFC is included in tr. mats',
                        type=bool, default=None)
    parser.add_argument('--rand_pretrain', help='pretrain with random transitions',
                        type=bool, default=None)

    # performance wrapper
    parser.add_argument('--perf_th', help='perf. threshold to change block',
                        type=float, default=None)
    parser.add_argument('--perf_w', help='window to compute performance',
                        type=int, default=None)

    # time-out wrapper parameters
    parser.add_argument('--time_out', help='time-out after error', type=int,
                        default=None)
    parser.add_argument('--stim_dur_limit', help='upper limit for stimulus' +
                        ' duration (should not be larger than stimulus period' +
                        'actual limit will be randomly choose for each trial)',
                        type=int, default=None)

    # perf-phases wrapper
    parser.add_argument('--start_ph', help='where to start the phase counter',
                        type=int, default=None)
    parser.add_argument('--end_ph', help='where to end the phase counter',
                        type=int, default=None)
    parser.add_argument('--step_ph', help='steps for phase counter',
                        type=int, default=None)
    parser.add_argument('--wait',
                        help='number of trials to wait before changing the phase',
                        type=int, default=None)

    # variable-nch wrapper parameters
    parser.add_argument('--block_nch',
                        help='dur. of blck in the variable-nch wrapper (trials)',
                        type=int, default=None)
    parser.add_argument('--blocks_probs', help='probability of each block',
                        type=float, nargs='+', default=None)
    parser.add_argument('--prob_12', help='percentage of 2AFC trials',
                        type=float, default=None)

    # reaction-time wrapper params
    parser.add_argument('--urgency', help='float value that will be added to the' +
                        ' reward to push the agent to respond quiclky (expected' +
                        ' to be negative)', type=float, default=None)

    # noise wrapper params
    parser.add_argument('--ev_incr', help='float value that allows gradually' +
                        ' increasing the evidence: (obs = obs*dt*ev_incr+noise)',
                        type=float, default=None)
    parser.add_argument('--std_noise', help='std for noise to add', type=float,
                        default=None)

    # monitor wrapper parameters
    parser.add_argument('--sv_fig',
                        help='indicates whether to save step-by-step figures',
                        type=bool, default=None)
    parser.add_argument('--sv_per',
                        help='number of trials to save behavioral data',
                        type=int, default=None)
    return parser


def update_dict(dict1, dict2):
    dict1.update((k, dict2[k]) for k in set(dict2).intersection(dict1))


def make_env(env_id, rank, seed=0, wrapps={}, **kwargs):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """
    def _init():
        env = gym.make(env_id, **kwargs)
        env.seed(seed + rank)
        for wrap in wrapps.keys():
            indx = wrap.find('_bis')
            if indx != -1:
                wrap = wrap[:indx]
            if not (wrap == 'MonitorExtended-v0' and rank != 0):
                env = apply_wrapper(env, wrap, wrapps[wrap])
        return env
    set_global_seeds(seed)
    return _init


def get_algo(alg):
    if alg == "A2C":
        from stable_baselines import A2C as algo
    elif alg == "ACER":
        from stable_baselines import ACER as algo
    elif alg == "ACKTR":
        from stable_baselines import ACKTR as algo
    elif alg == "PPO2":
        from stable_baselines import PPO2 as algo
    return algo


def run(task, task_kwargs, wrappers_kwargs, expl_params,
        rollout, num_trials, folder, n_lstm, rank,
        alpha_net=0.1, rerun=False, sigma_net=0.0, test_kwargs={}, num_retrains=10,
        seed=0, train_mode=None, sl_kwargs=None, sv_values=False,
        sv_activity=True):

    # reproducibility
    # random.seed(seed+rank*10)
    # torch.manual_seed(seed+rank*10)
    # np.random.seed(seed+rank*10)
    random.seed(int(time.time()))
    torch.manual_seed(random.randint(1, 4098))  # datetime.datetime.now())
    np.random.seed(int(time.time()))
    train_mode = train_mode or 'RL'
    files = []  # glob.glob(folder+'/*model*')
    vars_ = {'task': task,
             'task_kwargs': task_kwargs, 'wrappers_kwargs': wrappers_kwargs,
             'expl_params': expl_params, 'rollout': rollout,
             'folder': folder, 'num_trials': num_trials, 'n_lstm': n_lstm,
             'alpha_net': alpha_net, 'sigma_net': sigma_net}
    np.savez(folder + '/params.npz', **vars_)
    if len(files) == 0 or rerun:
        stps_ep = sl_kwargs['steps_per_epoch']
        lr = sl_kwargs['lr']
        wraps_sl = deepc(wrappers_kwargs)
        # env = make_env(env_id=task, rank=0, seed=seed, wrapps=wraps_sl,
        #                **task_kwargs)()
        env = make_env(env_id=task, rank=0, seed=random.randint(
            1, 4098), wrapps=wraps_sl, **task_kwargs)()
        obs_size = 1+1+1  # 1+1#1+n_ch+1#
        act_size = 1  # 1+n_ch
        # Instantiate the network and print information
        hidden_size = n_lstm
        input_size, output_size = obs_size, act_size
        noise_std = sigma_net  # 5e-2
        rank = rank

        wi_init, m_init, n_init, wo_init = None, None, None, None
        model_name = 'model'
        trained_files = glob.glob(folder+'/*'+model_name+'*')
        print('.....files', trained_files)
        if len(trained_files) > 0:
            sorted_models, last_model = order_by_sufix(trained_files)
            model_name = sorted_models[-1]
            print('Loading model: ')
            print(folder+'/'+model_name)
            weight_file = folder+'/'+model_name
            weights = np.load(weight_file)
            m, n, b, wi_full, wo_full = weights['m'], weights['n'], weights['b'],\
                weights['wi_full'], weights['wo_full']

            hs_, inps_ = wi_full.shape[1], wi_full.shape[0]
            hs_, outs_ = wo_full.shape[0], wo_full.shape[1]
            rank_ = m.shape[1]
            wi_init, m_init, n_init, wo_init =\
                np.zeros((input_size, hs_)), np.zeros((hs_, rank)),\
                np.zeros((hs_, rank)), np.zeros((hs_, output_size))
            # reduce to one dimensional inputs
            if inps_ > input_size:
                wi_init[0, :] = wi_full[0, :].copy()
                wi_init[1, :] = wi_full[1, :].copy()-wi_full[2, :].copy()
                wi_init[-1, :] = wi_full[-1, :].copy()
            elif inps_ < input_size:
                wi_init[0, :] = wi_full[0, :].copy()
                wi_init[1, :] = wi_full[1, :].copy()
                wi_init[2, :] = -wi_full[1, :].copy()
                wi_init[-1, :] = wi_full[-1, :].copy()
            else:
                wi_init = wi_full.copy()

            wo_init[:, :outs_] = wo_full.copy()
            for i in range(outs_, output_size):
                idxrand = np.random.choice(
                    len(wo_full.flatten()), hs_, replace=False)
                wo_init[:, i] = wo_full.flatten()[idxrand]

            m_init[:, :rank_] = m.copy()
            for i in range(rank_, rank):
                idxrand = np.random.choice(
                    len(m.flatten()), hs_, replace=False)
                m_init[:, i] = m.flatten()[idxrand]

            n_init[:, :rank_] = n.copy()
            for i in range(rank_, rank):
                idxrand = np.random.choice(
                    len(n.flatten()), hs_, replace=False)
                n_init[:, i] = n.flatten()[idxrand]
            wi_init = torch.from_numpy(wi_init).type(torch.cfloat)
            m_init = torch.from_numpy(m_init).type(torch.cfloat)
            n_init = torch.from_numpy(n_init).type(torch.cfloat)
            wo_init = torch.from_numpy(wo_init).type(torch.cfloat)
        # else:
        #     return

        model = LowRankRNN(input_size=input_size, hidden_size=hidden_size,
                           output_size=output_size, noise_std=noise_std,
                           alpha=alpha_net, rank=rank, train_wi=True,
                           train_wo=True, wi_init=wi_init, m_init=m_init,
                           n_init=n_init, wo_init=wo_init)
        print(model)
        # def scratch_model(xstim,noise_std,h,wi,m,n,b,wo,alpha,hs)
        m, n = model.m.clone(), model.n.clone()
        wi_full, wo_full = model.wi_full.clone(), model.wo_full.clone()
        b = model.b.clone()

        m, n = m.cpu(), n.cpu()
        wi_full = wi_full.cpu()
        wo_full = wo_full.cpu()
        b = b.cpu()

        m, n = m.detach().numpy(), n.detach().numpy()
        b = b.detach().numpy()
        wi_full, wo_full = wi_full.detach().numpy(), wo_full.detach().numpy()

        # print('vanilla rnn-- wi_full:', np.shape(wi_full), ', m:', np.shape(m),
        #       ', n:', np.shape(n), ', b:', np.shape(b),
        #       ', wo_full:', np.shape(wo_full))

        # recording training dataset
        sv_values = 1
        if sv_values:
            contexts = []  # @logYX
            observations = []
            obscategory = []
            ob_cum = []
            state_mat = []
            rewards = []
            actions = []
            prev_action = []
            actions_end_of_trial = []
            gt = []
            perf_ = []
            prev_perf = []
            info_vals = {}

        ob = env.reset()
        # To make sure that we start from the fixation time period of the trial.
        ob, rew, done, info = env.step(1)
        pre_start = ob[0]
        for i in range(rollout*2):
            ob, rew, done, info = env.step(1)
            start = ob[0]
            if pre_start == 0 and start == 1:
                print('start new trial')
                break
            pre_start = start

        ob_cum_temp = ob

        prev_act = -1
        prev_p = -1
        trial_count = 0
        perfs = []

        act_thred = 0.0  # 0.2#0#
        train_slot = 1
        # start and pre start

        numround = 2  # 6000#6000#
        # max_trial = 7 ### max trial length
        # max_trial = 21 ### max trial length
        max_trial = 20  # max trial length
        for iround in range(numround):
            # Analysis
            if train_slot:
                num_trial = (stps_ep+10)*rollout*sl_kwargs['btch_s']
                batch_trial = (stps_ep+10) * \
                    sl_kwargs['btch_s']*(rollout/max_trial)
            else:
                num_trial = rollout*100
                batch_trial = (stps_ep+10) * \
                    sl_kwargs['btch_s']*(rollout/max_trial)
            num_trial = int(num_trial)
            batch_trial = int(batch_trial)
            perf = 0
            trial_count_perf = 0
            dataset_ = {}
            dataset_['stimulus'] = []
            dataset_['gt'] = []
            dataset_['action_p'] = []

            hidden = None
            rate = None
            for i in range(num_trial):
                # current stimulus (obtained from the last step)
                ob = np.reshape(ob, (1, ob.shape[0]))
                all_zeros = not ob.copy()[0, :].any()
                if all_zeros:
                    batch_trial -= 1
                stim = np.zeros((1, ob.shape[1]-1))
                stim[0, 0], stim[0, 2] = ob[0, 0], ob[0, 3]
                stim[0, 1] = ob[0, 1]-ob[0, 2]
                # stim = np.zeros((1,ob.shape[1]))
                # stim = ob.copy()

                if hidden is None:
                    hidden = np.zeros((1, hidden_size))
                else:
                    hidden = state.copy()  # hidden.copy()
                # rate, state, action_p =\
                #     scratch_model_lr(stim[:,1:],noise_std,rate, hidden,
                #                      wi_full,m,n,b,wo_full,alpha_net,hidden_size)
                rate, state, action_p = scratch_model_lr(
                    stim[:, :], noise_std, rate, hidden, wi_full, m, n, b,
                    wo_full, alpha_net, hidden_size)
                # one-dimensional output
                if action_p[0, 0] > act_thred:
                    action = 2
                elif action_p[0, 0] < -act_thred:
                    action = 1
                else:
                    action = 0
                # now ob is the next stimulus, rew done info are for current
                ob, rew, done, info = env.step(action)
                if train_slot:
                    # generate dataset
                    if (len(dataset_['stimulus']) == 0):
                        obs_ = np.reshape(np.squeeze(stim.copy()), (1, -1))
                        dataset_['stimulus'] = obs_.copy()
                        dataset_['gt'] = np.ones(1)*info['gt']
                        dataset_['action_p'] = action_p[0, 0]
                    else:
                        obs_ = np.reshape(np.squeeze(stim.copy()), (1, -1))
                        dataset_['stimulus'] = np.vstack(
                            (dataset_['stimulus'], obs_))
                        dataset_['gt'] = np.append(dataset_['gt'], info['gt'])
                        dataset_['action_p'] = np.append(
                            dataset_['action_p'], action_p[0, 0])

                if info['gt'] > 0:
                    perf += rew
                    trial_count_perf += 1
                    if rew <= 0:
                        ob[-1] = -1
                hidden = state.copy()
                # recording training data
                if sv_values:
                    ob_cum_temp += ob
                    ob_cum.append(ob_cum_temp.copy())
                    if isinstance(info, (tuple, list)):
                        info = info[0]
                        ob_aux = ob[0]
                        rew = rew[0]
                        action = action[0]
                    else:
                        ob_aux = ob
                    if sv_activity:
                        state_mat.append(state)
                    observations.append(ob_aux)
                    if 'gt' in info.keys():
                        gt.append(info['gt'])
                    else:
                        gt.append(0)

                    rewards.append(rew)
                    actions.append(action)
                    prev_action.append(prev_act)
                    prev_perf.append(prev_p)
                    # @logYX
                    contexts.append(info['curr_block'])
                    obscategory.append(info['coh'])

                    for key in info:
                        if key not in info_vals.keys():
                            info_vals[key] = [info[key]]
                        else:
                            info_vals[key].append(info[key])
                    if info['new_trial']:
                        prev_act = action
                        prev_p = info['performance']
                        actions_end_of_trial.append(action)
                        perf_.append(info['performance'])
                        ob_cum_temp = np.zeros_like(ob_cum_temp)
                        trial_count += 1
                        if num_trials is not None and trial_count >= num_trials:
                            break
                    else:
                        actions_end_of_trial.append(-1)
                        perf_.append(-1)
                if batch_trial == 0:  # ensure that each loop start from 1,0,0,X
                    break
            perf /= trial_count_perf  # num_trial
            if len(perfs) == 0:
                perfs = np.ones(1)*perf
            else:
                perfs = np.append(perfs, perf)
            print('Round...#', iround, '...performance...',
                  perf, trial_count_perf)

            if train_slot:
                dataset = ngym_f.Dataset(env, dataset_, wrapps=wraps_sl,
                                         batch_size=sl_kwargs['btch_s'],
                                         seq_len=rollout, n_ch=n_ch,
                                         batch_first=True)
                # stages = iround//100
                n_epochs = 5  # 10#100
                # not keep best @YX-12May
                train(model, dataset, [], [], [], stps_ep, n_epochs,
                      lr=lr, n_ch=n_ch, batch_size=sl_kwargs['btch_s'],
                      clip_gradient=1., keep_best=False, cuda=False)
                # update the weights and bias for 'scratch_model_lr'
                m, n = model.m.clone(), model.n.clone()
                wi_full, wo_full = model.wi_full.clone(), model.wo_full.clone()
                b = model.b.clone()
                # ## from GPU to CPU
                # can't convert cuda:0 device type tensor to numpy.
                # Use Tensor.cpu() to copy the tensor to host memory first.
                m, n = m.cpu(), n.cpu()
                wi_full = wi_full.cpu()
                wo_full = wo_full.cpu()
                b = b.cpu()

                m, n = m.detach().numpy(), n.detach().numpy()
                b = b.detach().numpy()
                wi_full, wo_full =\
                    wi_full.detach().numpy(), wo_full.detach().numpy()
            train_slot = 1  # 1-train_slot

            if iround % 50 == 49:
                if sv_values:
                    # if model is not None and len(state_mat) > 0:
                    #     states = np.array(state_mat)
                    #     states = states[:,0,:]
                    # else:
                    #     states = None
                    data = {
                        'stimulus': np.array(observations),
                        'ob_cum': np.array(ob_cum),
                        'reward': np.array(rewards),
                        'choice': np.array(actions),
                        'perf': np.array(perf_),
                        'prev_choice': np.array(prev_action),
                        'prev_perf': np.array(prev_perf),
                        'actions_end_of_trial': actions_end_of_trial,
                        'gt': gt,
                        'contexts': contexts,
                        # 'states': states,
                        'info_vals': info_vals,
                        'obscategory': obscategory,
                    }
                    sv_folder = folder + '/'
                    # if data:
                    #     np.savez(sv_folder+'record_training_data'+
                    #              str(iround)+'.npz', **data)
                    # refreshing
                    contexts = []  # @logYX
                    observations = []
                    obscategory = []
                    ob_cum = []
                    state_mat = []
                    rewards = []
                    actions = []
                    prev_action = []
                    actions_end_of_trial = []
                    gt = []
                    prev_perf = []
                    info_vals = {}

                    trial_count = 0
                # reparameterize and orthogonization
                if iround == numround-1:
                    model.svd_reparametrization()

                m, n = model.m.clone(), model.n.clone()
                wi_full, wo_full = model.wi_full.clone(), model.wo_full.clone()
                b = model.b.clone()
                # from GPU to CPU
                # can't convert cuda:0 device type tensor to numpy.
                # Use Tensor.cpu() to copy the tensor to host memory first.
                m, n = m.cpu(), n.cpu()
                wi_full = wi_full.cpu()
                wo_full = wo_full.cpu()
                b = b.cpu()
                m, n = m.detach().numpy(), n.detach().numpy()
                b = b.detach().numpy()
                wi_full, wo_full =\
                    wi_full.detach().numpy(), wo_full.detach().numpy()

                data = {
                    'wi_full': np.array(wi_full),
                    'b': np.array(b),
                    'm': np.array(m),
                    'n': np.array(n),
                    'wo_full': np.array(wo_full),
                }

                # test_retrain = 'test'
                sv_folder = folder + '/'
                if data:
                    np.savez(sv_folder+'model_'+str(iround+000) +
                             '_rounds.npz', **data)
        if len(test_kwargs) != 0:
            for key in test_kwargs.keys():
                sv_folder = folder + key
                test_kwargs[key]['seed'] = seed
                # if train_mode == 'RL':
                if '_all' not in key:
                    data = ga.get_activity(
                        folder, alg, sv_folder, **test_kwargs[key])
                    if data:
                        np.savez(sv_folder+'data.npz', **data)
                else:
                    files = glob.glob(folder+'/model_*_steps.npz')
                    for f in files:
                        model_name = os.path.basename(f)
                        sv_f = folder+key+'_'+model_name[:-4]
                        data = ga.get_activity(folder, alg, sv_folder=sv_f,
                                               model_name=model_name,
                                               sigma_net=sigma_net,
                                               alpha_net=alpha_net,
                                               **test_kwargs[key])


if __name__ == "__main__":
    # Usage:
    # python bsc_dataaugment.py --folder '' --seed 1 --alg ACER --n_ch 2
    #                           --rollout 60 --alpha_net 0.1 --n_lstm 512 --rank 3
    # get params from call
    n_arg_parser = arg_parser()
    expl_params, unknown_args = n_arg_parser.parse_known_args(sys.argv)
    unkown_params = rest_arg_parser(unknown_args)
    if unkown_params:
        print('Unkown parameters: ', unkown_params)
    expl_params = vars(expl_params)
    expl_params = {k: expl_params[k] for k in expl_params.keys()
                   if expl_params[k] is not None}
    # folder /home/yshao/MLAuditory/results/vanilla_RNNs
    main_folder = expl_params['folder'] + '/'
    # main_folder = '/home/yshao/MLAuditory/results/vanilla_RNNs'
    name, _ = gncfd(expl_params)
    instance_folder = main_folder + name + '/'
    # this is done wo the monitor wrapper's parameter folder is updated
    expl_params['folder'] = instance_folder
    if not os.path.exists(instance_folder):
        os.makedirs(instance_folder)
    # load parameters
    print('Main folder: ', main_folder)
    sys.path.append(os.path.expanduser(main_folder))
    spec = importlib.util.spec_from_file_location("params",
                                                  main_folder+"/params.py")
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)
    # update general params
    gen_params = params.general_params
    update_dict(gen_params, expl_params)
    # update task params
    task_params = params.task_kwargs[gen_params['task']]
    update_dict(task_params, expl_params)
    # update wrappers params
    wrappers_kwargs = params.wrapps
    for wrap in wrappers_kwargs.keys():
        params_temp = wrappers_kwargs[wrap]
        update_dict(params_temp, expl_params)
    # get params
    tr_md = gen_params['train_mode'] if 'train_mode' in gen_params.keys(
    ) else 'RL'
    task = gen_params['task']
    seed = int(gen_params['seed'])
    num_trials = int(gen_params['num_trials'])
    rollout = int(gen_params['rollout'])
    rank = int(gen_params['rank'])  # @logYX
    n_lstm = int(gen_params['n_lstm'])
    alpha_net = float(gen_params['alpha_net'])
    sigma_net = float(gen_params['sigma_net'])
    task_kwargs = params.task_kwargs[gen_params['task']]
    # extra params
    test_kwargs = params.test_kwargs if hasattr(params, 'test_kwargs') else {}
    sl_kwargs = params.sl_kwargs if hasattr(params, 'sl_kwargs') else {}
    _ = run(task=task, task_kwargs=task_kwargs,
            wrappers_kwargs=params.wrapps, expl_params=expl_params,
            rollout=rollout, num_trials=num_trials,
            folder=instance_folder, n_lstm=n_lstm, rank=rank,
            test_kwargs=test_kwargs, seed=seed, train_mode=tr_md,
            sl_kwargs=sl_kwargs, alpha_net=alpha_net,
            sigma_net=sigma_net)
