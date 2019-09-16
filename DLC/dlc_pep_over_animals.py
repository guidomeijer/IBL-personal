#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:54:54 2019

@author: guido
"""

from oneibl.one import ONE
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
sys.path.insert(0, '/home/guido/Projects/ibllib/ibllib/dlc_analysis')
from dlc_basis_functions import load_dlc_training, load_event_times, load_events, px_to_mm
from dlc_plotting_functions import peri_plot
from dlc_analysis_functions import pupil_features

# Query sessions with available DLC data using ONE
one = ONE()
dtypes = ['_ibl_leftCamera.dlc', '_iblrig_taskData.raw', 'trials.feedback_times',
          'trials.feedbackType', 'trials.stimOn_times', 'trials.choice', '_iblrig_leftCamera.raw']
eids = one.search(dataset_types=dtypes)

# Initialize dataframes
pupil_stim_on = pd.DataFrame(columns=['eid', 'timepoint', 'trace'])
pupil_reward = pd.DataFrame(columns=['eid', 'timepoint', 'trace'])
paw_left = pd.DataFrame(columns=['eid', 'timepoint', 'trace'])
paw_right = pd.DataFrame(columns=['eid', 'timepoint', 'trace'])
tongue_reward = pd.DataFrame(columns=['eid', 'timepoint', 'trace'])
tongue_no_reward = pd.DataFrame(columns=['eid', 'timepoint', 'trace'])

# Loop over sessions
for i, eid in enumerate(eids):
    # Download data to local disk
    print('Processing DLC data of session %d of %d..' % (i+1, len(eids)))
    d = one.load(eid, dataset_types=dtypes, download_only=True, dclass_output=True)

    # Load in DLC traces and behavioral event timestamps
    folder_path = str(Path(d.local_path[0]).parent.parent)
    dlc_dict = load_dlc_training(folder_path)
    dlc_dict = px_to_mm(dlc_dict)
    stim_on_times, feedback_times = load_event_times(folder_path)
    choice, feedback_type = load_events(folder_path)

    # Fit pupil and get pupil traces
    pupil_x, pupil_y, diameter = pupil_features(dlc_dict)
    pstim_df = peri_plot(diameter, dlc_dict['timestamps'], stim_on_times,
                         None, [-1, 1], 'baseline')
    pstim_avg_df = pstim_df.groupby('timepoint').mean().reset_index()
    pstim_avg_df['eid'] = eid
    pupil_stim_on = pd.concat([pupil_stim_on, pstim_avg_df], ignore_index=True, sort=True)

    rew_df = peri_plot(diameter, dlc_dict['timestamps'], feedback_times[feedback_type == 1],
                       None, [-1, 1], 'baseline')
    prew_avg_df = rew_df.groupby('timepoint').mean().reset_index()
    prew_avg_df['eid'] = eid
    pupil_reward = pd.concat([pupil_reward, prew_avg_df], ignore_index=True, sort=True)

    # Get paw position
    this_pep_df = peri_plot(dlc_dict['middle_finger_r_x'], dlc_dict['timestamps'],
                            feedback_times[(choice == -1) & (feedback_type == 1)],
                            None, [-1, 1], 'baseline')
    this_avg_df = this_pep_df.groupby('timepoint').mean().reset_index()
    this_avg_df['eid'] = eid
    paw_left = pd.concat([paw_left, this_avg_df], ignore_index=True, sort=True)

    this_pep_df = peri_plot(dlc_dict['middle_finger_r_x'], dlc_dict['timestamps'],
                            feedback_times[(choice == 1) & (feedback_type == 1)],
                            None, [-1, 1], 'baseline')
    this_avg_df = this_pep_df.groupby('timepoint').mean().reset_index()
    this_avg_df['eid'] = eid
    paw_right = pd.concat([paw_right, this_avg_df], ignore_index=True, sort=True)

    # Get tongue position
    this_pep_df = peri_plot(dlc_dict['tongue_end_l_x'], dlc_dict['timestamps'],
                            feedback_times[feedback_type == 1],
                            None, [-1, 1], 'baseline')
    this_avg_df = this_pep_df.groupby('timepoint').mean().reset_index()
    this_avg_df['eid'] = eid
    tongue_reward = pd.concat([tongue_reward, this_avg_df], ignore_index=True, sort=True)

    this_pep_df = peri_plot(dlc_dict['tongue_end_l_x'], dlc_dict['timestamps'],
                            feedback_times[feedback_type == -1],
                            None, [-1, 1], 'baseline')
    this_avg_df = this_pep_df.groupby('timepoint').mean().reset_index()
    this_avg_df['eid'] = eid
    tongue_no_reward = pd.concat([tongue_no_reward, this_avg_df], ignore_index=True, sort=True)

# Plot output
sns.set(style="ticks", context="paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(16, 10))
plt.tight_layout(pad=4)

sns.lineplot(x='timepoint', y='trace', data=pupil_stim_on, ci=68, ax=ax1)
ax1.set(ylabel='Baseline subtracted pupil diameter (mm)', xlabel='Time (s)',
        title='Stimulus onset')
ax1.plot([0, 0], ax1.get_ylim(), 'r')

sns.lineplot(x='timepoint', y='trace', data=pupil_reward, ci=68, ax=ax2)
ax2.set(ylabel='Baseline subtracted pupil diameter (mm)', xlabel='Time (s)',
        title='Reward delivery')
ax2.plot([0, 0], ax2.get_ylim(), 'r')

sns.lineplot(x='timepoint', y='trace', data=paw_left, ci=68, ax=ax3)
ax3.set(ylabel='Baseline subtracted paw position (mm)', xlabel='Time (s)',
        title='Left rewarded trials')
ax3.plot([0, 0], ax3.get_ylim(), 'r')

sns.lineplot(x='timepoint', y='trace', data=paw_right, ci=68, ax=ax4)
ax4.set(ylabel='Baseline subtracted paw position (mm)', xlabel='Time (s)',
        title='Right rewarded trials')
ax4.plot([0, 0], ax4.get_ylim(), 'r')

sns.lineplot(x='timepoint', y='trace', data=tongue_reward, ci=68, ax=ax5)
ax5.set(ylabel='Baseline subtracted tongue position (mm)', xlabel='Time (s)',
        title='Rewarded trials')
ax5.plot([0, 0], ax5.get_ylim(), 'r')

sns.lineplot(x='timepoint', y='trace', data=tongue_no_reward, ci=68, ax=ax6)
# units='eid', estimator=None, hue='eid', legend=False)
ax6.set(ylabel='Baseline subtracted tongue position (mm)', xlabel='Time (s)',
        title='Unrewarded trials', ylim=ax5.get_ylim())
ax6.plot([0, 0], ax6.get_ylim(), 'r')
