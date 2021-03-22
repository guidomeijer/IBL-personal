#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:24:30 2021

@author: guido
"""
import vedo
from my_functions import get_children_region_names
import pandas as pd
from iblviewer import atlas_controller

nan_color = [0.5, 0.5, 0.5]
nan_alpha = 1
VMIN = -0.3
VMAX = 0.3

# Initialize
resolution = 25  # units = um
mapping = 'Beryl'
controller = atlas_controller.AtlasController()
controller.initialize(resolution, mapping, embed_ui=True, jupyter=False)

# Load in data
file_path = '/home/guido/Data/Ephys/Decoding/linear-regression/prior-prevaction_pseudo_kfold-interleaved_aligned-behavior_all_cells_beryl-atlas.p'
df = pd.read_pickle(file_path)

# Duplicate entries for all children
filtered_df = df.groupby('region')['r_prior'].median()
children = get_children_region_names(filtered_df.index.values)
for i, acronym in enumerate(filtered_df.index.values):
    if type(children[i]) is not str:
        filtered_df = filtered_df.append(pd.Series(
                    index=children[i], data=[filtered_df[acronym]] * children[i].shape[0]))
filtered_df = filtered_df.groupby(filtered_df.index).first()

scalar_map = {}
for acronym, value in filtered_df.iteritems():
    region_id, row_id = controller.model.get_region_and_row_id(acronym)
    if row_id == 0:
        # We ignore void acronym
        continue
    scalar_map[int(row_id)] = float(value)

# Get color mapping
rgb, alpha = [], []

# Initialize all on grey
for r_id in range(len(controller.model.metadata)):
    rgb.append([r_id, nan_color])
    a = nan_alpha if r_id > 0 else 0.0
    alpha.append([r_id, a])

# Map colors to regions
for row_id in scalar_map:
      value = scalar_map[row_id]
      rgb[row_id] = [row_id, list(vedo.colorMap(value, 'twilight_shifted',
                                                vmin=VMIN,
                                                vmax=VMAX))]
      alpha[row_id] = [row_id, 1.0]

controller.add_transfer_function(scalar_map, rgb, alpha, make_current=False)

controller.render()