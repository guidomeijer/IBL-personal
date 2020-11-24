"""
Created on Thu Feb  6 10:56:57 2020
Decode left/right block identity from all brain regions
@author: Guido Meijer
"""

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ibllib import atlas
from brainbox.atlas import plot_atlas
from ephys_functions import paths, figure_style, combine_layers_cortex

# Settings
TARGET = 'block'
DECODER = 'bayes'
MIN_REC = 2
MINMAX = 12
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding')
INCL_NEURONS = 'all'  # all or no_drift
INCL_SESSIONS = 'aligned_behavior'

ML = [-0.5, -3]  # in mm
AP = [1.5, -2.5]  # in mm
DV = [-2, -3.5]  # in mm


# %% Plot
# Load in data
decoding_pre = pd.read_pickle(join(SAVE_PATH,
       ('decode_%s_%s_%s_neurons_%s_sessions.p' % ('block', DECODER,
                                                   INCL_NEURONS, INCL_SESSIONS))))
decoding_stim = pd.read_pickle(join(SAVE_PATH,
        ('decode_%s_%s_%s_neurons_%s_sessions.p' % ('block_stim', DECODER,
                                                    INCL_NEURONS, INCL_SESSIONS))))

# Exclude root
decoding_pre = decoding_pre.reset_index(drop=True)
incl_regions = [i for i, j in enumerate(decoding_pre['region']) if not j.islower()]
decoding_pre = decoding_pre.loc[incl_regions]
decoding_stim = decoding_stim.reset_index(drop=True)
incl_regions = [i for i, j in enumerate(decoding_stim['region']) if not j.islower()]
decoding_stim = decoding_stim.loc[incl_regions]

# Drop duplicates
decoding_stim = decoding_stim[decoding_stim.duplicated(subset=['region', 'eid', 'probe'])
                              == False]
decoding_pre = decoding_pre[decoding_pre.duplicated(
                                            subset=['region', 'eid', 'probe']) == False]

# Get decoding performance over chance
decoding_stim['acc_over_chance'] = (decoding_stim['accuracy']
                                    - decoding_stim['chance_accuracy']) * 100
decoding_pre['acc_over_chance'] = (decoding_pre['accuracy']
                                   - decoding_pre['chance_accuracy']) * 100

# Remove cortical layers from brain region map
ba = atlas.AllenAtlas(25)
all_regions = combine_layers_cortex(ba.regions.acronym)

# Calculate average decoding performance per region
regions_stim = []
accuracy_stim = []
for i, region in enumerate(decoding_stim['region'].unique()):
    if np.sum(decoding_stim['region'] == region) >= MIN_REC:
        regions_stim.append(region)
        accuracy_stim.append(decoding_stim.loc[decoding_stim['region'] == region,
                                               'acc_over_chance'].mean())
regions_pre = []
accuracy_pre = []
for i, region in enumerate(decoding_pre['region'].unique()):
    if np.sum(decoding_pre['region'] == region) >= MIN_REC:
        regions_pre.append(region)
        accuracy_pre.append(decoding_pre.loc[decoding_pre['region'] == region,
                                            'acc_over_chance'].mean())

for i in range(len(ML)):
    f, (axs1, axs2) = plt.subplots(2, 3, figsize=(30, 12))
    figure_style(font_scale=2)
    plot_atlas(np.array(regions_pre), np.array(accuracy_pre), ML[i], AP[i], DV[i],
               color_palette='RdBu_r', minmax=[-MINMAX, MINMAX], axs=axs1,
               custom_region_list=all_regions)
    axs1[0].set(title='')
    axs1[1].set(title='Pre-stimulus')
    axs1[2].set(title='')
    plot_atlas(np.array(regions_stim), np.array(accuracy_stim), ML[i], AP[i], DV[i],
               color_palette='RdBu_r', minmax=[-MINMAX, MINMAX], axs=axs2,
               custom_region_list=all_regions)
    axs2[0].set(title='')
    axs2[1].set(title='Stimulus evoked')
    axs2[2].set(title='')
    f.suptitle('Decoding of stimulus prior')
    plt.savefig(join(FIG_PATH, 'atlas_decode_pre_vs_stim_%s_ML%.2f_AP%.2f_DV%.2f.png' % (
                            DECODER, ML[i], AP[i], DV[i])))
