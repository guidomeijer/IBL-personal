#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:21:49 2021

@author: guido
"""

import pandas as pd
import numpy as np
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from my_functions import paths, get_full_region_name, figure_style
from ibllib.atlas import BrainRegions
from oneibl.one import ONE
one = ONE()
br = BrainRegions()
fig_path = join(paths()[1], '5HT', 'switch_neurons_brainwide')
save_path = join(paths()[2], '5HT')

MIN_REC = 5
SHOW_REGIONS = 12

# Load in results
results_df = pd.read_csv(join(save_path, 'switch_neurons.csv'))
results_df['full_region'] = get_full_region_name(results_df['region'].values)

# Calculate average decoding performance per region
for i, region in enumerate(results_df['region'].unique()):
    results_df.loc[results_df['region'] == region, 'perc_median'] = results_df.loc[
                            results_df['region'] == region, 'percentage'].median()
    results_df.loc[results_df['region'] == region, 'n_rec'] = np.sum(results_df['region'] == region)

# Drop areas with too few recordings and sort
plot_df = results_df[results_df['n_rec'] >= MIN_REC].copy()
plot_df = plot_df.sort_values('perc_median', ascending=False)
plot_df = plot_df[plot_df['perc_median'] >= plot_df['perc_median'].unique()[SHOW_REGIONS - 1]]

figure_style(font_scale=1.8)
f, ax1 = plt.subplots(1, 1, figsize=(10, 10), dpi=300)
sort_regions = plot_df.groupby('full_region').max().sort_values(
                        'perc_median', ascending=False).reset_index()['full_region']
sns.stripplot(x='percentage', y='full_region', data=plot_df,
            order=sort_regions, s=6, ax=ax1)
sns.pointplot(x='percentage', y='full_region', data=plot_df,
            order=sort_regions, ci=68, join=False, estimator=np.median, color='k', ax=ax1)
ax1.set(xlabel='Percentage switch neurons', ylabel='')
plt.tight_layout()
