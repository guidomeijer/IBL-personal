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
REGION = 'ACAv'
ML = -0.5  # in mm
AP = 2  # in mm
DV = -2  # in mm
COMBINE_LAYERS_CORTEX = True

# Remove cortical layers from brain region map
ba = atlas.AllenAtlas(25)
if COMBINE_LAYERS_CORTEX:
    all_regions = combine_layers_cortex(ba.regions.acronym)
else:
    all_regions = ba.regions.acronym

# Highlight region
fill_region = np.zeros(all_regions.shape)
fill_region[all_regions == REGION] = 0.5

# Plot atlas
plot_atlas(all_regions, fill_region, ML, AP, DV, color_palette='bright',
           minmax=None, axs=None, custom_region_list=all_regions)
