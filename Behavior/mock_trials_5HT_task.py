#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:36:12 2020

@author: guido
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from ephys_functions import figure_style

TAU_VOLATILE = 3
TAU_STABLE = 6
BETA_A = 4
BETA_B = 4
N_TRIALS = 500

# Generate block structure for volatility
volatility = np.ones(90) * 0.5
while volatility.shape[0] < N_TRIALS:
    block_length = int(np.random.exponential(60))
    while (block_length < 20) | (block_length > 100):
        block_length = int(np.random.exponential(60))
    if (volatility.shape[0] == 90) & (np.random.choice([0, 1]) == 1):
        volatility = np.append(volatility, np.ones(block_length) * 1)
    elif (volatility.shape[0] > 90) & (volatility[-1] == 0):
        volatility = np.append(volatility, np.ones(block_length) * 1)
    elif (volatility.shape[0] > 90) & (volatility[-1] == 1):
        volatility = np.append(volatility, np.ones(block_length) * 0)
    else:
        volatility = np.append(volatility, np.ones(block_length) * 0)
volatility = volatility[:N_TRIALS]

# Generate stimuli
stim_size = np.empty(N_TRIALS)
for i in range(len(stim_size)):
    if volatility[i] == 0.5:
        stim_size[i] = np.random.choice([-1, 1])  # 50/50
    elif volatility[i] == 0:
        readout = np.sum(stim_size[:i] * np.exp(-np.arange(-i, 0) / -TAU_VOLATILE))
        prob = 1 / (1 + np.exp(-(1/TAU_VOLATILE) * readout))
        stim_size[i] = beta(BETA_A, BETA_B)
    elif volatility[i] == 1:
        readout = np.sum(stim_size[:i] * np.exp(-np.arange(-i, 0) / -TAU_STABLE))
        prob = 1 / (1 + np.exp(-(1/TAU_STABLE) * readout))
        stim_size[i] = bernoulli.rvs(prob)
stim_size[stim_size == 0] = -1

x = np.arange(-20, 0, 0.1)
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 8), dpi=100)

ax1.plot(x, np.exp(-x / -TAU_VOLATILE))

ax2.plot(x, np.exp(-x / -TAU_STABLE))

ax3.plot(stim_size, 'o')
sns.despine(trim=False)
plt.tight_layout()


 