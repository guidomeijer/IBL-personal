# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

@author: guido
"""

from os import listdir, mkdir
from os.path import join, isdir
import alf.io as ioalf
import matplotlib.pyplot as plt
import shutil
import brainbox as bb
import numpy as np
from scipy import stats
from functions_5HT import download_data, paths, sessions

download = False
overwrite = True
frontal_control = 'Frontal'

if frontal_control == 'Frontal':
    sessions, _ = sessions()
elif frontal_control == 'Control':
    _, sessions = sessions()

DATA_PATH, FIG_PATH, _ = paths()
FIG_PATH = join(FIG_PATH, 'PSTH', 'Blocks')
for i in range(sessions.shape[0]):
    # Download data if required
    if download is True:
        download_data(sessions.loc[i, 'subject'], sessions.loc[i, 'date'])

    # Get paths
    ses_nr = listdir(join(DATA_PATH, sessions.loc[i, 'lab'], 'Subjects',
                          sessions.loc[i, 'subject'], sessions.loc[i, 'date']))[0]
    session_path = join(DATA_PATH, sessions.loc[i, 'lab'], 'Subjects',
                        sessions.loc[i, 'subject'], sessions.loc[i, 'date'], ses_nr)
    alf_path = join(DATA_PATH, sessions.loc[i, 'lab'], 'Subjects', sessions.loc[i, 'subject'],
                    sessions.loc[i, 'date'], ses_nr, 'alf')
    probe_path = join(DATA_PATH, sessions.loc[i, 'lab'], 'Subjects', sessions.loc[i, 'subject'],
                      sessions.loc[i, 'date'], ses_nr, 'alf', 'probe%s' % sessions.loc[i, 'probe'])

    # Load in data
    spikes = ioalf.load_object(probe_path, 'spikes')
    clusters = ioalf.load_object(probe_path, 'clusters')
    trials = ioalf.load_object(alf_path, '_ibl_trials')

    # Only use single units
    spikes.times = spikes.times[np.isin(
            spikes.clusters, clusters.metrics.cluster_id[clusters.metrics.ks2_label == 'good'])]
    spikes.clusters = spikes.clusters[np.isin(
            spikes.clusters, clusters.metrics.cluster_id[clusters.metrics.ks2_label == 'good'])]

    # Calculate whether neuron discriminates
    trial_times = trials.goCue_times[((trials.probabilityLeft > 0.55)
                                      | (trials.probabilityLeft < 0.55))]
    trial_blocks = (trials.probabilityLeft[(((trials.probabilityLeft > 0.55)
                                            | (trials.probabilityLeft < 0.55)))] > 0.55).astype(
                                                                                            int)
    """
    auc_roc, cluster_ids = bb.task.calculate_roc(spikes.times, spikes.clusters,
                                                 trial_times, trial_blocks,
                                                 pre_time=0.5, post_time=0)
    roc_units = cluster_ids[(auc_roc < 0.4) | (auc_roc > 0.6)]
    """

    time_bins = np.arange(0, 1, 0.2)
    for j, time in enumerate(time_bins):
        sig_units, _, p_values, cluster_ids = bb.task.differentiate_units(spikes.times,
                                                                          spikes.clusters,
                                                                          trial_times,
                                                                          trial_blocks,
                                                                          pre_time=time+0.2,
                                                                          post_time=-time,
                                                                          test='ranksums',
                                                                          alpha=0.05)
        if j == 0:
            all_p_values = p_values
        else:
            all_p_values = np.column_stack((all_p_values, p_values))
    p_values = np.zeros(all_p_values.shape[0])
    for j in range(all_p_values.shape[0]):
        _, p_values[j] = stats.combine_pvalues(all_p_values[j, :])

    diff_units = cluster_ids[p_values < 0.05]
    # diff_units = roc_units

    # Make directories
    if (isdir(join(FIG_PATH, frontal_control, '%s_%s' % (sessions.loc[i, 'subject'],
                                                         sessions.loc[i, 'date'])))
            and (overwrite is True)):
        shutil.rmtree(join(FIG_PATH, frontal_control, '%s_%s' % (sessions.loc[i, 'subject'],
                                                                 sessions.loc[i, 'date'])))
    if not isdir(join(FIG_PATH, frontal_control, '%s_%s' % (sessions.loc[i, 'subject'],
                                                            sessions.loc[i, 'date']))):
        mkdir(join(FIG_PATH, frontal_control, '%s_%s' % (sessions.loc[i, 'subject'],
                                                         sessions.loc[i, 'date'])))
        for n, cluster in enumerate(diff_units):
            fig, ax = plt.subplots(1, 1)
            bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters,
                                              trials.stimOn_times[(trials.probabilityLeft > 0.55)],
                                              cluster, t_before=1, t_after=2,
                                              error_bars='sem', ax=ax)
            y_lim_1 = ax.get_ylim()
            bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters,
                                              trials.stimOn_times[(trials.probabilityLeft < 0.45)],
                                              cluster, t_before=1, t_after=2, error_bars='sem',
                                              pethline_kwargs={'color': 'red', 'lw': 2},
                                              errbar_kwargs={'color': 'red', 'alpha': 0.5}, ax=ax)
            y_lim_2 = ax.get_ylim()
            if y_lim_1[1] > y_lim_2[1]:
                ax.set(ylim=y_lim_1)
            plt.legend(['Left block', 'Right block'])
            plt.title('Stimulus Onset')
            plt.savefig(join(FIG_PATH, frontal_control,
                             '%s_%s' % (sessions.loc[i, 'subject'], sessions.loc[i, 'date']),
                             'p%s_d%s_n%s' % (sessions.loc[i, 'probe'],
                                              int(clusters.depths[
                                                  clusters.metrics.cluster_id == cluster][0]),
                                              cluster)))
            plt.close(fig)
