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
AP = 2.2  # in mm
DV = -3  # in mm
N_NEURONS = 15
MIN_REC = 2
MINMAX_BLOCK = 0.5
MINMAX_SURPRISE = 0.5
METRIC = 'f1'
SIDE = 'right'
COMBINE_LAYERS_CORTEX = True
_, FIG_PATH, SAVE_PATH = paths()

# %% Load in decoding results
if COMBINE_LAYERS_CORTEX:
    decoding_block = pd.read_csv(join(SAVE_PATH, ('decoding_block_combined_regions_%d_neurons.csv'
                                                  % N_NEURONS)))
else:
    decoding_block = pd.read_csv(join(SAVE_PATH, ('decoding_block_regions_%d_neurons.csv'
                                                  % N_NEURONS)))
if COMBINE_LAYERS_CORTEX:
    decoding_surprise = pd.read_csv(join(SAVE_PATH,
                                         'decoding_surprise_combined_regions_%d_neurons.csv'
                                         % N_NEURONS))
else:
    decoding_surprise = pd.read_csv(join(SAVE_PATH, 'decoding_surprise_regions_%d_neurons.csv'
                                         % N_NEURONS))
if 'f1_right_shuffle' in decoding_surprise.columns:
    decoding_surprise['f1_right_shuf'] = decoding_surprise['f1_right_shuffle']

# Exclude root
decoding_block = decoding_block.reset_index()
incl_regions = [i for i, j in enumerate(decoding_block['region']) if not j.islower()]
decoding_block = decoding_block.loc[incl_regions]
decoding_surprise = decoding_surprise.reset_index()
incl_regions = [i for i, j in enumerate(decoding_surprise['region']) if not j.islower()]
decoding_surprise = decoding_surprise.loc[incl_regions]

# Drop duplicates
decoding_block = decoding_block[~decoding_block.duplicated(subset=['region', 'eid', 'probe'])]
decoding_surprise = decoding_surprise[~decoding_surprise.duplicated(
                                                            subset=['region', 'eid', 'probe'])]

# Remove cortical layers from brain region map
ba = atlas.AllenAtlas(25)
if COMBINE_LAYERS_CORTEX:
    all_regions = combine_layers_cortex(ba.regions.acronym)
else: 
    all_regions = ba.regions.acronym    

# Get list of regions
regions_block = list((decoding_block['region'].value_counts() > MIN_REC).index)
regions_surprise = list((decoding_surprise['region'].value_counts() > MIN_REC).index)

# Create a list of decoding values
decod_block = np.empty(len(regions_block))
for i, region in enumerate(regions_block):
    decod_block[i] = (decoding_block.loc[decoding_block['region'] == region, METRIC]
                    - decoding_block.loc[decoding_block['region'] == region,
                                         '%s_shuffle' % METRIC]).mean() 
decod_surprise = np.empty(len(regions_surprise))
for i, region in enumerate(regions_surprise):
    decod_surprise[i] = (decoding_surprise.loc[decoding_surprise['region'] == region,
                                          '%s_%s' % (METRIC, SIDE)]
                    - decoding_surprise.loc[decoding_surprise['region'] == region,
                                         '%s_%s_shuf' % (METRIC, SIDE)]).mean()

f, (axs1, axs2) = plt.subplots(2, 3, figsize=(30, 12))
figure_style(font_scale=2)

plot_atlas(decoding_block['region'].unique(), decod_block, ML, AP, DV, color_palette='RdBu_r',
           minmax=[-MINMAX_BLOCK, MINMAX_BLOCK], axs=axs1, custom_region_list=all_regions)

plot_atlas(decoding_surprise['region'].unique(), decod_surprise, ML, AP, DV,
           color_palette='RdBu_r', minmax=[-MINMAX_SURPRISE, MINMAX_SURPRISE], axs=axs2,
           custom_region_list=all_regions)

if COMBINE_LAYERS_CORTEX:
    plt.savefig(join(FIG_PATH, 'WholeBrain',
                     'atlas_prior_combined_%s_%dneurons_ML%.2f_AP%.2f_DV%.2f.png' % (
                            METRIC, N_NEURONS, ML, AP, DV)))
else:
    plt.savefig(join(FIG_PATH, 'WholeBrain',
                     'atlas_prior_%s_%dneurons_ML%.2f_AP%.2f_DV%.2f.png' % (
                            METRIC, N_NEURONS, ML, AP, DV)))
