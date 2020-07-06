#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:58:04 2020

@author: guido
"""

from os.path import join
import brainbox as bb
import matplotlib.pyplot as plt
from ephys_functions import paths
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

# Settings
REGION = 'ACA'
MIN_CONTRAST = 0.1
PRE_TIME = 0.5
POST_TIME = 1
FIG_PATH = paths()[1]

# Query sessions with at least one channel in the region of interest
ses = one.alyx.rest('sessions', 'list', atlas_acronym=REGION,
                    task_protocol='_iblrig_tasks_ephysChoiceWorld',
                    project='ibl_neuropixel_brainwide')

# Loop over sessions
for i, eid in enumerate([j['url'][-36:] for j in ses]):
    print('Processing session %d of %d' % (i+1, len(ses)))
    
    # Load in data
    try:
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
        trials = one.load_object(eid, 'trials')
    except:
        continue
        
    # Get trial indices
    r_in_l_block = trials.stimOn_times[((trials.probabilityLeft == 0.8)
                                        & (trials.contrastRight > MIN_CONTRAST))]
    r_in_r_block = trials.stimOn_times[((trials.probabilityLeft == 0.2)
                                        & (trials.contrastRight > MIN_CONTRAST))]
    l_in_r_block = trials.stimOn_times[((trials.probabilityLeft == 0.2)
                                        & (trials.contrastLeft > MIN_CONTRAST))]
    l_in_l_block = trials.stimOn_times[((trials.probabilityLeft == 0.8)
                                        & (trials.contrastLeft > MIN_CONTRAST))]
        
    # Loop over probes
    for p, probe in enumerate(spikes.keys()):
        
        # Get clusters in region of interest
        region_clusters = [ind for ind, s in enumerate(clusters[probe]['acronym']) if REGION in s]
        
        for c, cluster_ind in enumerate(region_clusters):
            
            # Right stimulus
            fig, ax = plt.subplots(1, 1)
            bb.plot.peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                              r_in_r_block, cluster_ind,
                                              t_before=PRE_TIME, t_after=POST_TIME,
                                              error_bars='sem', ax=ax)
            y_lim_1 = ax.get_ylim()
            bb.plot.peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                              r_in_l_block, cluster_ind,
                                              t_before=PRE_TIME, t_after=POST_TIME,
                                              error_bars='sem',
                                              pethline_kwargs={'color': 'red', 'lw': 2},
                                              errbar_kwargs={'color': 'red', 'alpha': 0.5}, ax=ax)
            y_lim_2 = ax.get_ylim()
            if y_lim_1[1] > y_lim_2[1]:
                ax.set(ylim=y_lim_1)
            plt.legend(['Consistent', 'Inconsistent'])
            plt.title('Stimulus Onset (right side)')
            plt.tight_layout()
            plt.savefig(join(FIG_PATH, 'PSTH', 'Surprise',
                             '%s_%s_%s_%s_r' % (ses[i]['subject'], ses[i]['start_time'][:10],
                                              clusters[probe]['acronym'][cluster_ind],
                                              str(cluster_ind))))
            plt.close(fig)
            
            # Left stimulus
            fig, ax = plt.subplots(1, 1)
            bb.plot.peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                              l_in_l_block, cluster_ind,
                                              t_before=PRE_TIME, t_after=POST_TIME,
                                              error_bars='sem', ax=ax)
            y_lim_1 = ax.get_ylim()
            bb.plot.peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                              l_in_r_block, cluster_ind,
                                              t_before=PRE_TIME, t_after=POST_TIME,
                                              error_bars='sem',
                                              pethline_kwargs={'color': 'red', 'lw': 2},
                                              errbar_kwargs={'color': 'red', 'alpha': 0.5}, ax=ax)
            y_lim_2 = ax.get_ylim()
            if y_lim_1[1] > y_lim_2[1]:
                ax.set(ylim=y_lim_1)
            plt.legend(['Consistent', 'Inconsistent'])
            plt.title('Stimulus Onset (right side)')
            plt.tight_layout()
            plt.savefig(join(FIG_PATH, 'PSTH', 'Surprise',
                             '%s_%s_%s_%s_r' % (ses[i]['subject'], ses[i]['start_time'][:10],
                                              clusters[probe]['acronym'][cluster_ind],
                                              str(cluster_ind))))
            plt.close(fig)
    
    
