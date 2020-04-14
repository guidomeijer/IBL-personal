# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

@author: guido
"""

from os import mkdir
from os.path import join, isdir
import matplotlib.pyplot as plt
import alf.io
import brainbox as bb
import shutil
import numpy as np
from ephys_functions import paths, one_session_path
from oneibl.one import ONE
one = ONE()

OVERWRITE = True

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
    if not hasattr(trials, 'stimOn_times'):
        continue
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

        event_times = trials.stimOn_times[(trials.choice == -1) | (trials.choice == 1)]
        event_choices = (trials.choice[
                            (trials.choice == -1) | (trials.choice == 1)] == 1).astype(int)

        # Stimulus onset PSTH
        sig_units = bb.task.differentiate_units(spikes.times, spikes.clusters,
                                                event_times, event_choices, alpha=0.01)[0]
        print('%d out of %d units are significantly responsive to choice' % (
                                    len(sig_units), len(np.unique(spikes.clusters))))
        if (isdir(join(FIG_PATH, 'Choice', '%s_%s' % (nickname, ses_date)))
                and (OVERWRITE is True)):
            shutil.rmtree(join(FIG_PATH, 'Choice', '%s_%s' % (nickname, ses_date)))
        if not isdir(join(FIG_PATH, 'Choice', '%s_%s' % (nickname, ses_date))):
            mkdir(join(FIG_PATH, 'Choice', '%s_%s' % (nickname, ses_date)))
            for n, cluster in enumerate(sig_units):
                fig, ax = plt.subplots(1, 1)
                bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters,
                                                  event_times[event_choices == 0], cluster,
                                                  t_before=1, t_after=2,
                                                  error_bars='sem', ax=ax)
                y_lim_1 = ax.get_ylim()
                bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters,
                                                  event_times[event_choices == 1], cluster,
                                                  t_before=1, t_after=2,
                                                  error_bars='sem', ax=ax,
                                                  pethline_kwargs={'color': 'red', 'lw': 2},
                                                  errbar_kwargs={'color': 'red', 'alpha': 0.5})
                y_lim_2 = ax.get_ylim()
                if y_lim_1[1] > y_lim_2[1]:
                    ax.set(ylim=y_lim_1)
                plt.legend(['Choice left', 'Choice right'])
                plt.title('Stimulus Onset')
                plt.savefig(join(FIG_PATH, 'Choice', '%s_%s' % (nickname, ses_date),
                                 'p0%d_d%s_n%s' % (
                                     p, int(clusters.depths[
                                         clusters.metrics.cluster_id == cluster][0]),
                                     cluster)))
                plt.close(fig)
