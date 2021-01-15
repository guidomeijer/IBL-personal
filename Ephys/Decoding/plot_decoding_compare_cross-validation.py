#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020
Decode left/right block identity from all brain regions
@author: Guido Meijer
"""

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from my_functions import paths, figure_style, get_full_region_name, get_parent_region_name

# Settings
TARGET = 'block'
DECODER = 'bayes-multinomial'
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding', DECODER)
CHANCE_LEVEL = 'pseudo-blocks'
INCL_NEURONS = 'all'  # all or no_drift
INCL_SESSIONS = 'aligned-behavior'  # all or aligned
FULL_NAME = True
PARENT_REGIONS = False

# %% Plot
# Load in data
kfold = pd.read_pickle(join(SAVE_PATH, DECODER,
       ('%s_%s_%s_%s_%s_cells.p' % (TARGET, CHANCE_LEVEL, 'kfold',
                                    INCL_SESSIONS, INCL_NEURONS))))
kfold_interleaved = pd.read_pickle(join(SAVE_PATH, DECODER,
       ('%s_%s_%s_%s_%s_cells.p' % (TARGET, CHANCE_LEVEL, 'kfold-interleaved',
                                    INCL_SESSIONS, INCL_NEURONS))))
no_validation = pd.read_pickle(join(SAVE_PATH, DECODER,
       ('%s_%s_%s_%s_%s_cells.p' % (TARGET, CHANCE_LEVEL, 'none',
                                    INCL_SESSIONS, INCL_NEURONS))))

# Exclude root
kfold = kfold.reset_index(drop=True)
incl_regions = [i for i, j in enumerate(kfold['region']) if not j.islower()]
kfold = kfold.loc[incl_regions]
kfold_interleaved = kfold_interleaved.reset_index(drop=True)
incl_regions = [i for i, j in enumerate(kfold_interleaved['region']) if not j.islower()]
kfold_interleaved = kfold_interleaved.loc[incl_regions]
no_validation = no_validation.reset_index(drop=True)
incl_regions = [i for i, j in enumerate(no_validation['region']) if not j.islower()]
no_validation = no_validation.loc[incl_regions]

# Get decoding performance over chance
kfold['acc_over_chance'] = (kfold['accuracy']
                            - kfold['chance_accuracy']) * 100
kfold_interleaved['acc_over_chance'] = (kfold_interleaved['accuracy']
                                        - kfold_interleaved['chance_accuracy']) * 100
no_validation['acc_over_chance'] = (no_validation['accuracy']
                                    - no_validation['chance_accuracy']) * 100

# %%
figure_style(font_scale=2)
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30, 8), dpi=150, sharey=False)

ax1.hist(kfold['accuracy'] * 100, histtype='step', color=sns.color_palette('colorblind')[0],
         label='5-fold continuous', lw=3)
ax1.hist(kfold_interleaved['accuracy'] * 100, histtype='step', color=sns.color_palette('colorblind')[1],
         label='5-fold interleaved', lw=3)
ax1.hist(no_validation['accuracy'] * 100, histtype='step', color=sns.color_palette('colorblind')[2],
         label='No cross-validation', lw=3)
ax1.legend(frameon=False)
ax1.set(ylabel='Recordings split up by brain region', xlabel='Decoding accuracy (%)',
        ylim=[0, 400])

ax2.hist(kfold['chance_accuracy'] * 100, histtype='step', color=sns.color_palette('colorblind')[0],
         label='5-fold continuous', lw=3)
ax2.hist(kfold_interleaved['chance_accuracy'] * 100, histtype='step', color=sns.color_palette('colorblind')[1],
         label='5-fold interleaved', lw=3)
ax2.hist(no_validation['chance_accuracy'] * 100, histtype='step', color=sns.color_palette('colorblind')[2],
         label='No cross-validation', lw=3)
ax2.legend(frameon=False)
ax2.set(xlabel='Decoding accuracy on pseudo blocks (%)',
        ylim=[0, 400])

ax3.hist(kfold['p_accuracy'], histtype='step', color=sns.color_palette('colorblind')[0],
         label='5-fold continuous', lw=3)
ax3.hist(kfold_interleaved['p_accuracy'], histtype='step', color=sns.color_palette('colorblind')[1],
         label='5-fold interleaved', lw=3)
ax3.hist(no_validation['p_accuracy'], histtype='step', color=sns.color_palette('colorblind')[2],
         label='No cross-validation', lw=3)
ax3.legend(frameon=False)
ax3.set(xlabel='p-value',
        ylim=[0, 200])

ax4.hist(kfold['acc_over_chance'], histtype='step', color=sns.color_palette('colorblind')[0],
         label='5-fold continuous', lw=3)
ax4.hist(kfold_interleaved['acc_over_chance'], histtype='step', color=sns.color_palette('colorblind')[1],
         label='5-fold interleaved', lw=3)
ax4.hist(no_validation['acc_over_chance'], histtype='step', color=sns.color_palette('colorblind')[2],
         label='No cross-validation', lw=3)
ax4.legend(frameon=False)
ax4.set(xlabel='Decoding improvement over chance (%)',
        ylim=[0, 400])

plt.tight_layout(pad=2)
sns.despine()
plt.savefig(join(FIG_PATH, 'decode_compare_cross-validation'))

# %%

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8), dpi=150, sharey=False)
ax1.scatter(kfold.groupby('region')['acc_over_chance'].mean(),
            kfold_interleaved.groupby('region')['acc_over_chance'].mean())
ax1.set(xlabel='5-fold continuous', ylabel='5-flold interleaved')

ax2.scatter(no_validation.groupby('region')['acc_over_chance'].mean(),
            kfold_interleaved.groupby('region')['acc_over_chance'].mean())
ax2.set(xlabel='no cross-validation', ylabel='5-flold interleaved')

ax3.scatter(no_validation.groupby('region')['acc_over_chance'].mean(),
            kfold.groupby('region')['acc_over_chance'].mean())
ax3.set(xlabel='no cross-validation', ylabel='5-flold continuous')

plt.tight_layout()