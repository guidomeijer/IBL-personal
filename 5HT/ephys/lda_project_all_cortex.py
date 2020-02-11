#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020

@author: guido
"""

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import alf.io
import seaborn as sns
import brainbox as bb
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from functions_5HT import paths, plot_settings, one_session_path
from oneibl.one import ONE
one = ONE()

# Settings
DOWNLOAD = False
OVERWRITE = False
PRE_TIME = 1
POST_TIME = 0
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')

# Get list of recordings
eids, ses_info = one.search(dataset_types='spikes.times', details=True)

decoding_result = pd.DataFrame()
for i, eid in enumerate(eids):

    # Load in data
    print('Processing session %d of %d' % (i+1, len(eids)))
    session_path = one_session_path(eid)
    trials = one.load_object(eid, 'trials')
    if (not hasattr(trials, 'stimOn_times')
            or (trials.stimOn_times.shape[0] != trials.probabilityLeft.shape[0])):
        continue

    probes = one.load_object(eid, 'probes', download_only=False)
    for p in range(len(probes['trajectory'])):
        # Select shallow penetrations
        if (probes['trajectory'][p]['theta'] == 15) and (probes['trajectory'][p]['depth'] < 4500):
            probe_path = session_path.joinpath('alf', probes['description'][p]['label'])
            try:
                spikes = alf.io.load_object(probe_path, object='spikes')
                clusters = alf.io.load_object(probe_path, object='clusters')
            except Exception:
                continue

            # Only use good single units
            clusters_to_use = clusters.metrics.ks2_label == 'good'
            if clusters_to_use.sum() < 4:
                continue
            spikes.times = spikes.times[
                    np.isin(spikes.clusters, clusters.metrics.cluster_id[clusters_to_use])]
            spikes.clusters = spikes.clusters[
                    np.isin(spikes.clusters, clusters.metrics.cluster_id[clusters_to_use])]
            clusters.depths = clusters.depths[clusters_to_use]
            cluster_ids = clusters.metrics.cluster_id[clusters_to_use]

            # Get spike counts for all trials
            trial_times = trials.stimOn_times[(trials.probabilityLeft > 0.55)
                                              | (trials.probabilityLeft < 0.45)]
            times = np.column_stack(((trial_times - PRE_TIME), (trial_times + POST_TIME)))
            spike_counts, cluster_ids = bb.task._get_spike_counts_in_bins(spikes.times,
                                                                          spikes.clusters, times)
            trial_blocks = (trials.probabilityLeft[
                                    (((trials.probabilityLeft > 0.55)
                                      | (trials.probabilityLeft < 0.45)))] > 0.55).astype(int)

            # Transform to LDA
            lda = LDA(n_components=1)
            lda_transform = lda.fit_transform(np.rot90(spike_counts), trial_blocks)

            # Correlate probability left with lda score
            r = stats.pearsonr(np.rot90(lda_transform)[0],
                               trials.probabilityLeft[
                                   (trials.probabilityLeft > 0.55)
                                   | (trials.probabilityLeft < 0.45)])[0]

            # Add to dataframe
            nickname = ses_info[i]['subject']
            ses_date = ses_info[i]['start_time'][:10]
            decoding_result = decoding_result.append(pd.DataFrame(
                index=[0], data={'subject': nickname, 'date': ses_date, 'eid': eid,
                                 'r': r,
                                 'ML': probes.trajectory[p]['x'],
                                 'AP': probes.trajectory[p]['y'],
                                 'DV': probes.trajectory[p]['z'],
                                 'phi': probes.trajectory[p]['phi'],
                                 'theta': probes.trajectory[p]['theta'],
                                 'depth': probes.trajectory[p]['depth']}))

decoding_result.to_csv(join(DATA_PATH, 'lda_block_all_cortex'))

fig, ax = plt.subplots(1, 1)
sns.scatterplot(x='ML', y='AP', data=decoding_result, size=10, hue='r', palette='YlOrRd')
plot_settings()
plt.savefig(join(FIG_PATH, 'LDA_all_cortex'))
