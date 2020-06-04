# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

@author: guido
"""

from brainbox.population import _get_spike_counts_in_bins as spike_bins
import brainbox.io.one as bbone
import numpy as np
from oneibl.one import ONE
one = ONE()

# Settings
PRE_TIME = 0.1
POST_TIME = 0.1
SUBJECT = 'ZM_2240'
DATE = '2020-01-23'
PROBE = '00'

# Load in data
eids = one.search(subject=SUBJECT, date_range=DATE)
spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eids[0], one=one)
trials = one.load_object(eids[0], 'trials')

# Only use single units
probe = 'probe' + PROBE
spikes[probe].times = spikes[probe].times[np.isin(
        spikes[probe].clusters, clusters[probe].metrics.cluster_id[
            clusters[probe].metrics.ks2_label == 'good'])]
spikes[probe].clusters = spikes[probe].clusters[np.isin(
        spikes[probe].clusters, clusters[probe].metrics.cluster_id[
            clusters[probe].metrics.ks2_label == 'good'])]

# Get bin around go cue
times = np.column_stack(((trials.goCue_times - PRE_TIME), (trials.goCue_times + POST_TIME)))
pop_vector, cluster_ids = spike_bins(spikes[probe].times, spikes[probe].clusters, times)
