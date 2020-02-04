# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

@author: guido
"""

from os import mkdir, rmdir
from os.path import join, isdir
import matplotlib.pyplot as plt
import brainbox as bb
import numpy as np
from ephys_functions import paths
from oneibl.one import ONE
one = ONE()

overwrite = False

eids, ses_info = one.search(user='guido', dataset_types='spikes.times', details=True)
dtypes = ['spikes.times',
          'spikes.clusters',
          'clusters.channels',
          'clusters.metrics',
          'probes.trajectory',
          'trials.choice',
          'trials.contrastLeft',
          'trials.contrastRight',
          'trials.feedback_times',
          'trials.goCue_times',
          'trials.feedbackType',
          'trials.response_times',
          'trials.stimOn_times']

DATA_PATH, FIG_PATH = paths()
FIG_PATH = join(FIG_PATH, 'PSTH')
for i, eid in enumerate(eids):

    (spike_times, spike_clusters, cluster_channels, cluster_metrics,
     probe_traj, choice, contrast_l, contrast_r, feedback_times,
     gocue_times, feedback_type, response_times, stimon_times) = one.load(eid,
                                                                          dataset_types=dtypes)

    # Only use single units
    spike_times = spike_times[np.isin(
            spike_clusters, cluster_metrics.cluster_id[cluster_metrics.ks2_label == 'good'])]
    spike_clusters = spike_clusters[np.isin(
            spike_clusters, cluster_metrics.cluster_id[cluster_metrics.ks2_label == 'good'])]

    # Get session info
    nickname = ses_info[i]['subject']
    ses_date = ses_info[i]['start_time'][:10]

    # Stimulus onset PSTH
    sig_units, p_values, _ = bb.task.responsive_units(spike_times, spike_clusters, stimon_times)
    print('%d out of %d units are significantly responsive to stimulus onset' % (
                                len(sig_units), len(np.unique(spike_clusters))))
    if (isdir(join(FIG_PATH, 'StimOn', '%s_%s' % (nickname, ses_date)))
            and (overwrite is True)):
        rmdir(join(FIG_PATH, 'StimOn', '%s_%s' % (nickname, ses_date)))
    if not isdir(join(FIG_PATH, 'StimOn', '%s_%s' % (nickname, ses_date))):
        mkdir(join(FIG_PATH, 'StimOn', '%s_%s' % (nickname, ses_date)))
        for n, cluster in enumerate(sig_units):
            fig, ax = plt.subplots(1, 1)
            bb.plot.peri_event_time_histogram(spike_times, spike_clusters, stimon_times,
                                              cluster, t_before=1, t_after=2,
                                              error_bars='sem', ax=ax)
            plt.title('Stimulus Onset')
            plt.savefig(join(FIG_PATH, 'StimOn', '%s_%s' % (nickname, ses_date),
                             'c%s_n%s' % (cluster_channels[n], cluster)))
            plt.close(fig)

    # Reward PSTH
    sig_units, p_values, _ = bb.task.responsive_units(spike_times, spike_clusters,
                                                      feedback_times[feedback_type == 1],
                                                      alpha=0.01)
    print('%d out of %d units are significantly responsive to reward delivery' % (
                                len(sig_units), len(np.unique(spike_clusters))))
    if (isdir(join(FIG_PATH, 'Reward', '%s_%s' % (nickname, ses_date)))
            and (overwrite is True)):
        rmdir(join(FIG_PATH, 'Reward', '%s_%s' % (nickname, ses_date)))
    if not isdir(join(FIG_PATH, 'Reward', '%s_%s' % (nickname, ses_date))):
        mkdir(join(FIG_PATH, 'Reward', '%s_%s' % (nickname, ses_date)))
        for n, cluster in enumerate(sig_units):
            fig, ax = plt.subplots(1, 1)
            bb.plot.peri_event_time_histogram(spike_times, spike_clusters,
                                              feedback_times[feedback_type == 1],
                                              cluster, t_before=1, t_after=2,
                                              error_bars='sem', ax=ax)
            plt.title('Reward delivery')
            plt.savefig(join(FIG_PATH, 'Reward', '%s_%s' % (nickname, ses_date),
                             'c%s_n%s' % (cluster_channels[n], cluster)))
            plt.close(fig)

    # Reward ommission PSTH
    sig_units, p_values, _ = bb.task.responsive_units(spike_times, spike_clusters,
                                                      feedback_times[feedback_type == -1],
                                                      alpha=0.01)
    print('%d out of %d units are significantly responsive to reward omission' % (
                                len(sig_units), len(np.unique(spike_clusters))))
    if (isdir(join(FIG_PATH, 'RewardOmission', '%s_%s' % (nickname, ses_date)))
            and (overwrite is True)):
        rmdir(join(FIG_PATH, 'RewardOmission', '%s_%s' % (nickname, ses_date)))
    if not isdir(join(FIG_PATH, 'RewardOmission', '%s_%s' % (nickname, ses_date))):
        mkdir(join(FIG_PATH, 'RewardOmission', '%s_%s' % (nickname, ses_date)))
        for n, cluster in enumerate(sig_units):
            fig, ax = plt.subplots(1, 1)
            bb.plot.peri_event_time_histogram(spike_times, spike_clusters,
                                              feedback_times[feedback_type == -1],
                                              cluster, t_before=1, t_after=2,
                                              error_bars='sem', ax=ax)
            plt.title('Reward omission')
            plt.savefig(join(FIG_PATH, 'RewardOmission', '%s_%s' % (nickname, ses_date),
                             'c%s_n%s' % (cluster_channels[n], cluster)))
            plt.close(fig)
