#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:31:04 2020

@author: guido
"""

import numpy as np
import pandas as pd
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from ibllib import atlas
from brainbox.atlas import plot_atlas
from ephys_functions import paths, figure_style, combine_layers_cortex

# Settings
ML = -0.5  # in mm
AP = 2  # in mm
DV = -3  # in mm
MIN_REC = 1
MINMAX_F1 = 0.2
MINMAX_ACC = 0.15
DECODER = 'bayes'
_, FIG_PATH, SAVE_PATH = paths()
INCL_NEURONS = 'all'  # all or no_drift
INCL_SESSIONS = 'all'  # all or aligned

# %% Plot
# Load in data
decoding_block = pd.read_pickle(join(SAVE_PATH,
           ('decode_block_%s_%s_neurons_%s_sessions.p' % (DECODER, INCL_NEURONS, INCL_SESSIONS))))

# Exclude root
decoding_block = decoding_block.reset_index()
incl_regions = [i for i, j in enumerate(decoding_block['region']) if not j.islower()]
decoding_block = decoding_block.loc[incl_regions]

# Drop duplicates
decoding_block = decoding_block[~decoding_block.duplicated(subset=['region', 'eid', 'probe'])]

# Remove cortical layers from brain region map
ba = atlas.AllenAtlas(25)
all_regions = combine_layers_cortex(ba.regions.acronym)

# Get list of regions
regions_block = np.array(list((decoding_block['region'].value_counts() > MIN_REC).index))

# Create a list of decoding values
f1_block = np.empty(len(regions_block))
acc_block = np.empty(len(regions_block))
for i, region in enumerate(regions_block):
    f1_block[i] = np.mean((decoding_block.loc[decoding_block['region'] == region, 'f1']
                      - [i.mean() for i in decoding_block.loc[decoding_block['region'] == region,
                                                              'chance_f1']]))
    acc_block[i] = np.mean((decoding_block.loc[decoding_block['region'] == region, 'accuracy']
                      - [i.mean() for i in decoding_block.loc[decoding_block['region'] == region,
                                                              'chance_accuracy']]))

f, (axs1, axs2) = plt.subplots(2, 3, figsize=(30, 12))
figure_style(font_scale=2)
plot_atlas(regions_block, f1_block, ML, AP, DV, color_palette='RdBu_r',
           minmax=[-MINMAX_F1, MINMAX_F1], axs=axs1, custom_region_list=all_regions)

plot_atlas(regions_block, acc_block, ML, AP, DV, color_palette='RdBu_r',
           minmax=[-MINMAX_ACC, MINMAX_ACC], axs=axs2, custom_region_list=all_regions)

plt.savefig(join(FIG_PATH, 'WholeBrain',
                 'atlas_decode_block_%s_%s_neurons_%s_sessions_ML%.2f_AP%.2f_DV%.2f.png' % (
                        DECODER, INCL_NEURONS, INCL_SESSIONS, ML, AP, DV)))
