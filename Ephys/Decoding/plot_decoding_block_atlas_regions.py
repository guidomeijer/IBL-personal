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
from ephys_functions import paths
from ibllib import atlas

# Settings
ML_COORDINATE = 0
N_NEURONS = 20
_, FIG_PATH, SAVE_PATH = paths()

# Load in decoding results
decoding_result = pd.read_csv(join(SAVE_PATH, 'decoding_block_all_regions_%d_neurons' % N_NEURONS))

# Exclude root
decoding_result = decoding_result.reset_index()
incl_regions = [i for i, j in enumerate(decoding_result['region']) if not j.islower()]
decoding_result = decoding_result.loc[incl_regions]

# Get the atlas
ba = atlas.AllenAtlas(25)

# Get region boundaries
boundaries = np.diff(ba.label, axis=0)
boundaries = boundaries + np.diff(ba.label, axis=1)
boundaries = boundaries + np.diff(ba.label, axis=2)

# Create a color map of RGB colors per region according to decoding performance
colors = np.full(ba.regions.rgb.shape, 255, dtype='uint8')
acronyms = ba.regions.acronym
color_map = sns.color_palette('icefire', 100)
for i, region in enumerate(decoding_result['region'].unique()):
    f1_score = decoding_result.loc[decoding_result['region'] == region, 'f1'].mean()
    colors[acronyms == region, :] = np.round(np.array(color_map[int(np.round(f1_score * 100))])
                                             * 255)
    
# Get a saggital slice with regions colored by classificaion performance
index = ba.bc.xyz2i(np.array([ml_coordinate] * 3))[0]
imlabel = ba.label.take(index, axis=ba.xyz2dims[0])
im_unique, ilabels, iim = np.unique(imlabel, return_index=True, return_inverse=True)
_, ir_unique, _ = np.intersect1d(ba.regions.id, im_unique, return_indices=True)
imrgb = np.reshape(colors[ir_unique[iim], :], (*imlabel.shape, 3))

plt.imshow(imrgb)