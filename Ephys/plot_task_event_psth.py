# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

@author: guido
"""

from os import mkdir
from os.path import join, isdir
import matplotlib.pyplot as plt
from pathlib import Path
import alf.io
import brainbox as bb
import shutil
import numpy as np
from ephys_functions import paths
from oneibl.one import ONE
one = ONE()

OVERWRITE = True


def one_session_path(eid):
    ses = one.alyx.rest('sessions', 'read', id=eid)
    return Path(one._par.CACHE_DIR, ses['lab'], 'Subjects', ses['subject'],
                ses['start_time'][:10], str(ses['number']).zfill(3))


# Get list of recordings
eids, ses_info = one.search(user='guido', dataset_types='spikes.times', details=True)

# Set path to save plots
DATA_PATH, FIG_PATH = paths()
FIG_PATH = join(FIG_PATH, 'PSTH')

# Loop over recordings
for i, eid in enumerate(eids):

    # Load in data
    session_path = one_session_path(eid)
    trials = one.load_object(eid, 'trials')
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

        # Stimulus onset PSTH
        sig_units = bb.task.responsive_units(spikes.times, spikes.clusters,
                                             trials.stimOn_times, alpha=0.01)[0]
        print('%d out of %d units are significantly responsive to stimulus onset' % (
                                    len(sig_units), len(np.unique(spikes.clusters))))
        if (isdir(join(FIG_PATH, 'StimOn', '%s_%s' % (nickname, ses_date)))
                and (OVERWRITE is True)):
            shutil.rmtree(join(FIG_PATH, 'StimOn', '%s_%s' % (nickname, ses_date)))
        if not isdir(join(FIG_PATH, 'StimOn', '%s_%s' % (nickname, ses_date))):
            mkdir(join(FIG_PATH, 'StimOn', '%s_%s' % (nickname, ses_date)))
            for n, cluster in enumerate(sig_units):
                fig, ax = plt.subplots(1, 1)
                bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters,
                                                  trials.stimOn_times, cluster,
                                                  t_before=1, t_after=2,
                                                  error_bars='sem', ax=ax)
                plt.title('Stimulus Onset')
                plt.savefig(join(FIG_PATH, 'StimOn', '%s_%s' % (nickname, ses_date),
                                 'p0%d_d%s_n%s' % (
                                     p, int(clusters.depths[
                                         clusters.metrics.cluster_id == cluster][0]),
                                     cluster)))
                plt.close(fig)

        # Reward PSTH
        sig_units = bb.task.responsive_units(spikes.times, spikes.clusters,
                                             trials.feedback_times[trials.feedbackType == 1],
                                             alpha=0.01)[0]
        print('%d out of %d units are significantly responsive to reward delivery' % (
                                    len(sig_units), len(np.unique(spikes.clusters))))
        if (isdir(join(FIG_PATH, 'Reward', '%s_%s' % (nickname, ses_date)))
                and (OVERWRITE is True)):
            shutil.rmtree(join(FIG_PATH, 'Reward', '%s_%s' % (nickname, ses_date)))
        if not isdir(join(FIG_PATH, 'Reward', '%s_%s' % (nickname, ses_date))):
            mkdir(join(FIG_PATH, 'Reward', '%s_%s' % (nickname, ses_date)))
            for n, cluster in enumerate(sig_units):
                fig, ax = plt.subplots(1, 1)
                bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters,
                                                  trials.feedback_times[
                                                      trials.feedbackType == 1],
                                                  cluster, t_before=1, t_after=2,
                                                  error_bars='sem', ax=ax)
                plt.title('Reward delivery')
                plt.savefig(join(FIG_PATH, 'Reward', '%s_%s' % (nickname, ses_date),
                                 'p0%d_d%s_n%s' % (
                                     p, int(
                                         clusters.depths[
                                             clusters.metrics.cluster_id == cluster][0]),
                                     cluster)))
                plt.close(fig)

        # Reward ommission PSTH
        sig_units = bb.task.responsive_units(spikes.times, spikes.clusters,
                                             trials.feedback_times[trials.feedbackType == -1],
                                             alpha=0.01)[0]
        print('%d out of %d units are significantly responsive to reward omission' % (
                                    len(sig_units), len(np.unique(spikes.clusters))))
        if (isdir(join(FIG_PATH, 'RewardOmission', '%s_%s' % (nickname, ses_date)))
                and (OVERWRITE is True)):
            shutil.rmtree(join(FIG_PATH, 'RewardOmission', '%s_%s' % (nickname, ses_date)))
        if not isdir(join(FIG_PATH, 'RewardOmission', '%s_%s' % (nickname, ses_date))):
            mkdir(join(FIG_PATH, 'RewardOmission', '%s_%s' % (nickname, ses_date)))
            for n, cluster in enumerate(sig_units):
                fig, ax = plt.subplots(1, 1)
                bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters,
                                                  trials.feedback_times[
                                                      trials.feedbackType == -1],
                                                  cluster, t_before=1, t_after=2,
                                                  error_bars='sem', ax=ax)
                plt.title('Reward omission')
                plt.savefig(join(FIG_PATH, 'RewardOmission', '%s_%s' % (nickname, ses_date),
                                 'p0%d_d%s_n%s' % (
                                     p, int(clusters.depths[
                                         clusters.metrics.cluster_id == cluster][0]),
                                     cluster)))
                plt.close(fig)
