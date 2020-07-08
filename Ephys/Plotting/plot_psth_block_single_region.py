#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:58:04 2020

@author: guido
"""

from os.path import join
import brainbox as bb
import numpy as np
import matplotlib.pyplot as plt
from ephys_functions import paths, check_trials
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
        region_clusters = [ind for ind, s in enumerate(clusters[probe]['acronym']) if REGION in s]
        region_clusters = np.array(region_clusters)
        
        # Calculate significant units
        sig_units = bb.task.differentiate_units(spikes[probe].times, spikes[probe].clusters,
                                                trial_times, trial_blocks,
                                                pre_time=0.6, post_time=-0.1, alpha=0.05)[0]
                
        for c, cluster_ind in enumerate(region_clusters[np.isin(region_clusters, sig_units)]):
            
            fig, ax = plt.subplots(1, 1)
            bb.plot.peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                              trials.stimOn_times[trials.probabilityLeft == 0.8],
                                              cluster_ind, t_before=PRE_TIME, t_after=POST_TIME,
                                              error_bars='sem', ax=ax)
            y_lim_1 = ax.get_ylim()
            bb.plot.peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                              trials.stimOn_times[trials.probabilityLeft == 0.2],
                                              cluster_ind, t_before=PRE_TIME, t_after=POST_TIME,
                                              error_bars='sem',
                                              pethline_kwargs={'color': 'red', 'lw': 2},
                                              errbar_kwargs={'color': 'red', 'alpha': 0.5}, ax=ax)
            y_lim_2 = ax.get_ylim()
            if y_lim_1[1] > y_lim_2[1]:
                ax.set(ylim=y_lim_1)
            plt.legend(['Left', 'Right'])
            plt.title('Stimulus Onset (right side)')
            plt.tight_layout()
            plt.savefig(join(FIG_PATH, 'PSTH', 'Block',
                             '%s_%s_%s_%s' % (ses[i]['subject'], ses[i]['start_time'][:10],
                                              clusters[probe]['acronym'][cluster_ind],
                                              str(cluster_ind))))
            plt.close(fig)
         
   
    
