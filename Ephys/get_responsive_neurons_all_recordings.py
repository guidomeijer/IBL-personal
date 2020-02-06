#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:28:36 2020

@author: guido
"""

from os import mkdir
from os.path import join, isdir
from glob import glob
import matplotlib.pyplot as plt
from pathlib import Path
import alf.io
import pandas as pd
import brainbox as bb
import shutil
from scipy import stats
import numpy as np
from ephys_functions import paths
from oneibl.one import ONE
one = ONE()

OVERWRITE = False


def one_session_path(eid):
    ses = one.alyx.rest('sessions', 'read', id=eid)
    return Path(one._par.CACHE_DIR, ses['lab'], 'Subjects', ses['subject'],
                ses['start_time'][:10], str(ses['number']).zfill(3))


# Get list of recordings
eids, ses_info = one.search(dataset_types='spikes.times', details=True)

# Set path to save plots
DATA_PATH, FIG_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')

resp = pd.DataFrame()
for i, eid in enumerate(eids):

    # Load in data
    session_path = one_session_path(eid)
    spikes = one.load_object(eid, 'spikes', download_only=True)
    trials = one.load_object(eid, 'trials')
    if (len(spikes) != 0) & (hasattr(trials, 'stimOn_times')):
        probes = one.load_object(eid, 'probes', download_only=False)
        for p in range(len(probes['trajectory'])):
            probe_path = session_path.joinpath('alf', probes['description'][p]['label'])
            try:
                spikes = alf.io.load_object(probe_path, object='spikes')
                clusters = alf.io.load_object(probe_path, object='clusters')
            except Exception:
                continue

            # Only use single units
            spikes.times = spikes.times[np.isin(
                    spikes.clusters, clusters.metrics.cluster_id[
                        clusters.metrics.ks2_label == 'good'])]
            spikes.clusters = spikes.clusters[np.isin(
                    spikes.clusters, clusters.metrics.cluster_id[
                        clusters.metrics.ks2_label == 'good'])]

            # Get session info
            nickname = ses_info[i]['subject']
            ses_date = ses_info[i]['start_time'][:10]

            # Get number of responsive neurons
            sig_stim = bb.task.responsive_units(spikes.times, spikes.clusters,
                                                trials.stimOn_times, alpha=0.01)[0]
            sig_rew = bb.task.responsive_units(spikes.times, spikes.clusters,
                                               trials.feedback_times[trials.feedbackType == 1],
                                               alpha=0.01)[0]
            sig_omit = bb.task.responsive_units(spikes.times, spikes.clusters,
                                                trials.feedback_times[trials.feedbackType == -1],
                                                alpha=0.01)[0]

            # Get neurons that differentiate between blocks
            trial_times = trials.goCue_times[((trials.probabilityLeft > 0.55)
                                              | (trials.probabilityLeft < 0.55))]
            trial_blocks = (trials.probabilityLeft[
                            (((trials.probabilityLeft > 0.55)
                              | (trials.probabilityLeft < 0.55)))] > 0.55).astype(int)
            time_bins = np.arange(0, 1, 0.2)
            for j, time in enumerate(time_bins):
                sig_units, _, p_values, cluster_ids = bb.task.differentiate_units(
                        spikes.times, spikes.clusters, trial_times, trial_blocks,
                        pre_time=time+0.2, post_time=-time, test='ranksums', alpha=0.05)
                if j == 0:
                    all_p_values = p_values
                else:
                    all_p_values = np.column_stack((all_p_values, p_values))
            p_values = np.zeros(all_p_values.shape[0])
            for j in range(all_p_values.shape[0]):
                _, p_values[j] = stats.combine_pvalues(all_p_values[j, :])
            diff_units = cluster_ids[p_values < 0.05]

            resp = resp.append(pd.DataFrame(index=[0],
                                            data={'subject': nickname,
                                                  'date': ses_date,
                                                  'eid': eid,
                                                  'stim': sig_stim.shape[0]/p_values.shape[0],
                                                  'reward': sig_rew.shape[0]/p_values.shape[0],
                                                  'omit': sig_omit.shape[0]/p_values.shape[0],
                                                  'blocks': (diff_units.shape[0]
                                                             / cluster_ids.shape[0]),
                                                  'ML': probes.trajectory[p]['x'],
                                                  'AP': probes.trajectory[p]['y'],
                                                  'DV': probes.trajectory[p]['z'],
                                                  'phi': probes.trajectory[p]['phi'],
                                                  'theta': probes.trajectory[p]['theta'],
                                                  'depth': probes.trajectory[p]['depth']}))
