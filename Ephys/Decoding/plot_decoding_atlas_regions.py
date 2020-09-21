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
from ephys_functions import paths, figure_style, combine_layers_cortex
from ibllib import atlas

# Settings
ML_COORDINATE = -1  # in mm
AP_COORDINATE = 3  # in mm
DV_COORDINATE = -1  # in mm
N_NEURONS = 10
CRANGE_BLOCK = 0.1
CRANGE_SURPRISE = 0.3
METRIC = 'accuracy'
SIDE = 'right'
COMBINE_LAYERS_CORTEX = True
_, FIG_PATH, SAVE_PATH = paths()


def get_slice(coordinate, orientation, fill_values):
    if orientation == 'sagittal':
        axis = 0
    elif orientation == 'coronal':
        axis = 1
    elif orientation == 'horizontal':
        axis = 2
    index = ba.bc.xyz2i(np.array([coordinate / 1000] * 3))[axis]
    imlabel = ba.label.take(index, axis=ba.xyz2dims[axis])
    im_unique, ilabels, iim = np.unique(imlabel, return_index=True, return_inverse=True)
    _, ir_unique, _ = np.intersect1d(ba.regions.id, im_unique, return_indices=True)
    im = np.squeeze(np.reshape(fill_values[ir_unique[iim]], (*imlabel.shape, 1)))
    return im
    

def get_slice_boundaries(coordinate, orientation, boundaries):
    if orientation == 'sagittal':
        axis = 0
    elif orientation == 'coronal':
        axis = 1
    elif orientation == 'horizontal':
        axis = 2
    index = ba.bc.xyz2i(np.array([coordinate / 1000] * 3))[axis]
    im_boundaries = boundaries.take(index, axis=ba.xyz2dims[axis])
    return im_boundaries
    

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
        
decoding_surprise['f1_right_shuf'] = decoding_surprise['f1_right_shuffle']

# Exclude root
decoding_block = decoding_block.reset_index()
incl_regions = [i for i, j in enumerate(decoding_block['region']) if not j.islower()]
decoding_block = decoding_block.loc[incl_regions]
decoding_surprise = decoding_surprise.reset_index()
incl_regions = [i for i, j in enumerate(decoding_surprise['region']) if not j.islower()]
decoding_surprise = decoding_surprise.loc[incl_regions]

# Get the atlas
ba = atlas.AllenAtlas(25)

# Calculate region boundaries volume
boundaries = np.diff(ba.label, axis=0, append=0)
boundaries = boundaries + np.diff(ba.label, axis=1, append=0)
boundaries = boundaries + np.diff(ba.label, axis=2, append=0)
boundaries[boundaries != 0] = 1

# Remove cortical layers from brain region map
if COMBINE_LAYERS_CORTEX:
    regions = combine_layers_cortex(ba.regions.acronym)
else: 
    regions = ba.regions.acronym    

# Create a map of decoding performances
decod_block = np.zeros(ba.regions.id.shape)
for i, region in enumerate(decoding_block['region'].unique()):
    decod_result = (decoding_block.loc[decoding_block['region'] == region, METRIC]
                    - decoding_block.loc[decoding_block['region'] == region,
                                         '%s_shuffle' % METRIC]).mean()
    decod_block[regions == region] = decod_result
    
decod_surprise = np.zeros(ba.regions.id.shape)
for i, region in enumerate(decoding_surprise['region'].unique()):
    decod_result = (decoding_surprise.loc[decoding_surprise['region'] == region,
                                          '%s_%s' % (METRIC, SIDE)]
                    - decoding_surprise.loc[decoding_surprise['region'] == region,
                                         '%s_%s_shuf' % (METRIC, SIDE)]).mean()
    decod_surprise[regions == region] = decod_result
    
# Get a saggital slice with regions colored by classificaion performance
im_sag_block = get_slice(ML_COORDINATE, 'sagittal', decod_block)
im_cor_block = get_slice(AP_COORDINATE, 'coronal', decod_block)
im_hor_block = get_slice(DV_COORDINATE, 'horizontal', decod_block)
im_sag_surprise = get_slice(ML_COORDINATE, 'sagittal', decod_surprise)
im_cor_surprise = get_slice(AP_COORDINATE, 'coronal', decod_surprise)
im_hor_surprise = get_slice(DV_COORDINATE, 'horizontal', decod_surprise)

# Get slice boundaries
im_sag_boundaries = get_slice_boundaries(ML_COORDINATE, 'sagittal', boundaries)
im_cor_boundaries = get_slice_boundaries(AP_COORDINATE, 'coronal', boundaries)
im_hor_boundaries = get_slice_boundaries(DV_COORDINATE, 'horizontal', boundaries)

# Apply boundaries to slices
im_sag_block[im_sag_boundaries == 1] = -1
im_cor_block[im_cor_boundaries == 1] = -1
im_hor_block[im_hor_boundaries == 1] = -1
im_sag_surprise[im_sag_boundaries == 1] = -1
im_cor_surprise[im_cor_boundaries == 1] = -1
im_hor_surprise[im_hor_boundaries == 1] = -1

# Construct color map
color_map = sns.diverging_palette(220, 20, n=1000)
color_map.append((0.8, 0.8, 0.8))
color_map.insert(0, (0.8, 0.8, 0.8))
color_map.insert(501, (1, 1, 1))

# %% Plot


f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(30, 12))
figure_style(font_scale=2)
             
sns.heatmap(np.rot90(im_sag_block, 3), cmap=color_map, cbar=True,
            vmin=-CRANGE_BLOCK, vmax=CRANGE_BLOCK, ax=ax1)
ax1.set(title='Block identity, ML: %.1f mm' % ML_COORDINATE)
plt.axis('off')
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)

sns.heatmap(np.rot90(im_cor_block, 3), cmap=color_map, cbar=True,
            vmin=-CRANGE_BLOCK, vmax=CRANGE_BLOCK, ax=ax2)
ax2.set(title='Block identity, AP: %.1f mm' % AP_COORDINATE)
plt.axis('off')
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)

sns.heatmap(np.rot90(im_hor_block, 3), cmap=color_map, cbar=True,
            vmin=-CRANGE_BLOCK, vmax=CRANGE_BLOCK, ax=ax3)
ax3.set(title='Block identity, DV: %.1f mm' % DV_COORDINATE)
plt.axis('off')
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)

sns.heatmap(np.rot90(im_sag_surprise, 3), cmap=color_map, cbar=True,
            vmin=-CRANGE_SURPRISE, vmax=CRANGE_SURPRISE, ax=ax4)
ax4.set(title='Consistent stimulus, ML: %.1f mm' % ML_COORDINATE)
plt.axis('off')
ax4.get_xaxis().set_visible(False)
ax4.get_yaxis().set_visible(False)

sns.heatmap(np.rot90(im_cor_surprise, 3), cmap=color_map, cbar=True,
            vmin=-CRANGE_SURPRISE, vmax=CRANGE_SURPRISE, ax=ax5)
ax5.set(title='Consistent stimulus, AP: %.1f mm' % AP_COORDINATE)
plt.axis('off')
ax5.get_xaxis().set_visible(False)
ax5.get_yaxis().set_visible(False)

sns.heatmap(np.rot90(im_hor_surprise, 3), cmap=color_map, cbar=True,
            vmin=-CRANGE_SURPRISE, vmax=CRANGE_SURPRISE, ax=ax6)
ax6.set(title='Consistent stimulus, DV: %.1f mm' % DV_COORDINATE)
plt.axis('off')
ax6.get_xaxis().set_visible(False)
ax6.get_yaxis().set_visible(False)

if COMBINE_LAYERS_CORTEX:
    plt.savefig(join(FIG_PATH, 'WholeBrain',
                     'atlas_prior_combined_%s_%dneurons_ML%.2f_AP%.2f_DV%.2f.png' % (
                            METRIC, N_NEURONS, ML_COORDINATE, AP_COORDINATE, DV_COORDINATE)))
else:
    plt.savefig(join(FIG_PATH, 'WholeBrain',
                     'atlas_prior_%s_%dneurons_ML%.2f_AP%.2f_DV%.2f.png' % (
                            METRIC, N_NEURONS, ML_COORDINATE, AP_COORDINATE, DV_COORDINATE)))
