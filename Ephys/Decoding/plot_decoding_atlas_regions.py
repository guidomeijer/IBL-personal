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
TARGET = 'reward'
DECODER = 'bayes'
MIN_REC = 1
MINMAX = 30
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding')
INCL_NEURONS = 'all'  # all or no_drift
INCL_SESSIONS = 'behavior_crit'  # all or aligned
ML = -2  # in mm
AP = 2  # in mm
DV = -3  # in mm


# %% Plot
# Load in data
decoding_result = pd.read_pickle(join(SAVE_PATH,
       ('decode_%s_%s_%s_neurons_%s_sessions.p' % (TARGET, DECODER, INCL_NEURONS, INCL_SESSIONS))))

# Exclude root
decoding_result = decoding_result.reset_index()
incl_regions = [i for i, j in enumerate(decoding_result['region']) if not j.islower()]
decoding_result = decoding_result.loc[incl_regions]

# Drop duplicates
decoding_result = decoding_result[~decoding_result.duplicated(subset=['region', 'eid', 'probe'])]

# Calculate accuracy over chance
decoding_result['acc_over_chance'] = (decoding_result['accuracy']
                                      - decoding_result['chance_accuracy']) * 100

# Remove cortical layers from brain region map
ba = atlas.AllenAtlas(25)
all_regions = combine_layers_cortex(ba.regions.acronym)

# Calculate average decoding performance per region
decode_regions = []
accuracy = []
for i, region in enumerate(decoding_result['region'].unique()):
    if np.sum(decoding_result['region'] == region) >= MIN_REC:
        decode_regions.append(region)
        accuracy.append(decoding_result.loc[decoding_result['region'] == region,
                                            'acc_over_chance'].mean())

f, axs1 = plt.subplots(1, 3, figsize=(30, 6))
figure_style(font_scale=2)
plot_atlas(np.array(decode_regions), np.array(accuracy), ML, AP, DV, color_palette='RdBu_r',
           minmax=[-MINMAX, MINMAX], axs=axs1, custom_region_list=all_regions)

if TARGET == 'stim_side':
    f.suptitle('Decoding of stimulus side')
elif TARGET == 'block':
    f.suptitle('Decoding of stimulus prior from pre-stim activity')
elif TARGET == 'blank':
    f.suptitle('Decoding of stimulus prior from blank trials')
elif TARGET == 'block_stim':
    f.suptitle('Decoding of stimulus prior from stimulus period')
elif TARGET == 'reward':
    f.suptitle('Decoding of reward or ommission')
elif TARGET == 'choice':
    f.suptitle('Decoding of choice')

plt.savefig(join(FIG_PATH, 'atlas_decode_%s_%s_%s_neurons_%s_sessions_ML%.2f_AP%.2f_DV%.2f.png' % (
                        TARGET, DECODER, INCL_NEURONS, INCL_SESSIONS, ML, AP, DV)))


