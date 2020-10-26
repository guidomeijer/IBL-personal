#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:58:04 2020

@author: guido
"""

from os import mkdir
from os.path import join, isdir
import brainbox as bb
import alf
import numpy as np
import matplotlib.pyplot as plt
from ephys_functions import paths, check_trials
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

# Settings
REGION = 'MD'
TEST_PRE_TIME = 0.6
TEST_POST_TIME = -0.1
PLOT_PRE_TIME = 0.5
PLOT_POST_TIME = 1
ALPHA = 0.05
FIG_PATH = paths()[1]

# Query sessions with at least one channel in the region of interest
ses = one.alyx.rest('sessions', 'list', atlas_acronym=REGION,
                    task_protocol='_iblrig_tasks_ephysChoiceWorld',
                    project='ibl_neuropixel_brainwide')

# Make directory for region
if not isdir(join(FIG_PATH, 'PSTH', 'Block', REGION)):
    mkdir(join(FIG_PATH, 'PSTH', 'Block', REGION))

# Loop over sessions
for i, eid in enumerate([j['url'][-36:] for j in ses]):
    print('Processing session %d of %d' % (i+1, len(ses)))

    # Load in data
    try:
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
        ses_path = one.path_from_eid(eid)
        trials = alf.io.load_object(join(ses_path, 'alf'), 'trials')
    except:
        continue

    # Check data integrity
    if check_trials(trials) is False:
        continue

    # Get trial indices
    trial_times = trials.goCue_times[
        ((trials.probabilityLeft == 0.8) | (trials.probabilityLeft == 0.2))]
    trial_blocks = (trials.probabilityLeft[
        (((trials.probabilityLeft == 0.8) | (trials.probabilityLeft == 0.2)))] == 0.8).astype(int)

    # Loop over probes
    for p, probe in enumerate(spikes.keys()):

        # Get clusters in region of interest
        clusters_in_region = clusters[probe].metrics.cluster_id[
                        [i for i, j in enumerate(clusters[probe]['acronym']) if REGION in j]]
        if len(clusters_in_region) == 0:
            continue

        # Select spikes and clusters
        spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
        clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_in_region)]

        # Get block neurons
        sig_block = bb.task.differentiate_units(spks_region, clus_region,
                                                trial_times, trial_blocks,
                                                pre_time=TEST_PRE_TIME, post_time=TEST_POST_TIME,
                                                alpha=ALPHA)[0]

        for c, cluster_ind in enumerate(sig_block):

            fig, ax = plt.subplots(1, 1)
            bb.plot.peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                              trials.stimOn_times[trials.probabilityLeft == 0.8],
                                              cluster_ind, t_before=PLOT_PRE_TIME,
                                              t_after=PLOT_POST_TIME, error_bars='sem', ax=ax)
            y_lim_1 = ax.get_ylim()
            bb.plot.peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                              trials.stimOn_times[trials.probabilityLeft == 0.2],
                                              cluster_ind, t_before=PLOT_PRE_TIME,
                                              t_after=PLOT_POST_TIME, error_bars='sem',
                                              pethline_kwargs={'color': 'red', 'lw': 2},
                                              errbar_kwargs={'color': 'red', 'alpha': 0.5}, ax=ax)
            y_lim_2 = ax.get_ylim()
            if y_lim_1[1] > y_lim_2[1]:
                ax.set(ylim=y_lim_1)
            plt.legend(['Left', 'Right'], frameon=False)
            plt.title('Stim. onset')
            plt.tight_layout()
            plt.savefig(join(FIG_PATH, 'PSTH', 'Block', REGION,
                             '%s_%s_%s_%s' % (ses[i]['subject'], ses[i]['start_time'][:10],
                                              clusters[probe]['acronym'][cluster_ind].replace(
                                                                                        '/', '-'),
                                              str(cluster_ind))))
            plt.close(fig)



