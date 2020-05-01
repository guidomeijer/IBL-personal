#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020

Decode left/right block identity from all superficial recordings

@author: guido
"""

from os.path import join
import alf.io as ioalf
import matplotlib.pyplot as plt
import numpy as np
from brainbox.population import decode
import pandas as pd
import seaborn as sns
import alf.io
from sklearn.utils import shuffle
from brainbox.io.one import load_spike_sorting
from ephys_functions import paths, figure_style
from oneibl.one import ONE
one = ONE()

# Settings
DOWNLOAD = False
OVERWRITE = False
PRE_TIME = 0.6
POST_TIME = -0.1
N_NEURONS = 75
MIN_TRIALS = 300
ITERATIONS = 1000
DECODER = 'bayes'  # bayes, regression or forest
VALIDATION = 'kfold'
NUM_SPLITS = 5
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')

# Get list of recordings
eids, ses_info = one.search(dataset_types=['spikes.times', 'probes.trajectory'],
                            task_protocol='_iblrig_tasks_ephysChoiceWorld', details=True)

decoding_result = pd.DataFrame()
for i, eid in enumerate(eids):

    # Load in data
    print('Processing session %d of %d' % (i+1, len(eids)))
    try:
        spikes, clusters = load_spike_sorting(eid, one=one)
        trials = one.load_object(eid, 'trials')
        probes = one.load_object(eid, 'probes')
    except:
        continue

    # Check data integrity
    if ((not hasattr(trials, 'stimOn_times'))
            or (len(trials.feedback_times) != len(trials.feedbackType))
            or (len(trials.stimOn_times) != len(trials.probabilityLeft))
            or (not hasattr(probes, 'trajectory'))):
        continue

    for p, probe in enumerate(spikes.keys()):

        # Select shallow penetrations
        if probes['trajectory'][p]['phi'] == 180:

            # Only use good single units
            spikes[probe].times = spikes[probe].times[np.isin(
                    spikes[probe].clusters, clusters[probe].metrics.cluster_id[
                        clusters[probe].metrics.ks2_label == 'good'])]
            spikes[probe].clusters = spikes[probe].clusters[np.isin(
                    spikes[probe].clusters, clusters[probe].metrics.cluster_id[
                        clusters[probe].metrics.ks2_label == 'good'])]

            # Get trial vectors
            incl_trials = (trials.probabilityLeft > 0.55) | (trials.probabilityLeft < 0.45)
            trial_times = trials.stimOn_times[incl_trials]
            probability_left = trials.probabilityLeft[incl_trials]
            trial_blocks = (trials.probabilityLeft[incl_trials] > 0.55).astype(int)

            # Check for number of neurons and trials
            if ((np.unique(spikes[probe].clusters).shape[0] < N_NEURONS)
                    or (trial_times.shape[0] < MIN_TRIALS)):
                continue

            # Decode block identity
            decode_result = decode(spikes[probe].times, spikes[probe].clusters,
                                   trial_times, trial_blocks,
                                   pre_time=PRE_TIME, post_time=POST_TIME,
                                   classifier=DECODER, cross_validation=VALIDATION,
                                   num_splits=NUM_SPLITS, n_neurons=N_NEURONS,
                                   iterations=ITERATIONS)

            # Shuffle
            shuffle_result = decode(spikes[probe].times, spikes[probe].clusters,
                                    trial_times, shuffle(trial_blocks),
                                    pre_time=PRE_TIME, post_time=POST_TIME,
                                    classifier=DECODER, cross_validation=VALIDATION,
                                    num_splits=NUM_SPLITS, n_neurons=N_NEURONS,
                                    iterations=ITERATIONS, shuffle=True)

            # Add to dataframe
            nickname = ses_info[i]['subject']
            ses_date = ses_info[i]['start_time'][:10]
            decoding_result = decoding_result.append(pd.DataFrame(
                index=[0], data={'subject': nickname, 'date': ses_date, 'eid': eid,
                                 'f1': decode_result['f1'].mean() - shuffle_result['f1'].mean(),
                                 'accuracy': (decode_result['accuracy'].mean()
                                              - shuffle_result['accuracy'].mean()),
                                 'auroc': (decode_result['auroc'].mean()
                                           - shuffle_result['auroc'].mean()),
                                 'ML': probes.trajectory[p]['x'],
                                 'AP': probes.trajectory[p]['y'],
                                 'DV': probes.trajectory[p]['z'],
                                 'phi': probes.trajectory[p]['phi'],
                                 'theta': probes.trajectory[p]['theta'],
                                 'depth': probes.trajectory[p]['depth'],
                                 'probe': probes.description[p]['label']}))

decoding_result.to_csv(join(SAVE_PATH, 'decoding_block_all_cortex'))

# %% Plot
Y_LIM = [-6000, 4000]
X_LIM = [-5000, 5000]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
plot_h = sns.scatterplot(x='ML', y='AP', data=decoding_result.sort_values(by='f1'), hue='f1',
                         palette='RdBu_r', s=100, hue_norm=(-0.3, 0.3), ax=ax1)

# Fix legend
leg = plot_h.legend(loc=(0.75, 0.5))
leg.texts[0].set_text('F1')


plot_h = sns.scatterplot(x='ML', y='AP', data=decoding_result.sort_values(by='accuracy'),
                         hue='accuracy', palette='RdBu_r', s=100, hue_norm=(-0.15, 0.15), ax=ax2)

# Fix legend
leg = plot_h.legend(loc=(0.75, 0.5))
leg.texts[0].set_text('F1')

# figure_style()
plt.savefig(join(FIG_PATH, 'block_decode_all_cortex'))
