#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:21:49 2021

@author: guido
"""

import numpy as np
from my_functions import (paths, query_sessions, check_trials, combine_layers_cortex, load_trials,
                          remap)
import brainbox.io.one as bbone
from ibllib.atlas import BrainRegions
from oneibl.one import ONE
one = ONE()
br = BrainRegions()

# Settings
INCL_NEURONS = 'pass-QC'
INCL_SESSIONS = 'aligned-behavior'
ATLAS = 'beryl-atlas'
MIN_NEURONS = 5
MIN_CONTRAST = 0.1

# Query session list
eids, probes = query_sessions(selection=INCL_SESSIONS)

for i in range(len(eids)):
    print('\nProcessing session %d of %d' % (i+1, len(eids)))

    # Load in data
    eid = eids[i]
    try:
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
                                                                    eid, aligned=True, one=one)
        ses_path = one.path_from_eid(eid)
        trials = load_trials(eid)
    except Exception as error_message:
        print(error_message)
        continue

    # Check data integrity
    if check_trials(trials) is False:
        continue

    # Extract session data
    ses_info = one.get_details(eid)
    subject = ses_info['subject']
    date = ses_info['start_time'][:10]
    probes_to_use = probes[i]

    # Process trials
    trials = trials.loc[trials['probabilityLeft'] != 0.5].reset_index(drop=True)  # Exclude 50/50 block
    transitions = np.array(np.where(np.diff(trials['probabilityLeft']) != 0)[0]) + 1

    # Decode per brain region
    for p, probe in enumerate(probes_to_use):
        print('Processing %s (%d of %d)' % (probe, p + 1, len(probes_to_use)))

        # Check if data is available for this probe
        if probe not in clusters.keys():
            continue

        # Check if histology is available for this probe
        if not hasattr(clusters[probe], 'acronym'):
            continue

        # Check if cluster metrics are available
        if 'metrics' not in clusters[probe]:
            continue

        # Get list of brain regions
        if ATLAS == 'beryl-atlas':
            mapped_br = br.get(ids=remap(clusters[probe]['atlas_id']))
            clusters_regions = mapped_br['acronym']
        elif ATLAS == 'allen-atlas':
            clusters_regions = combine_layers_cortex(clusters[probe]['acronym'])

        # Get list of neurons that pass QC
        if INCL_NEURONS == 'pass-QC':
            clusters_pass = np.where(clusters[probe]['metrics']['label'] == 1)[0]
        elif INCL_NEURONS == 'all':
            clusters_pass = np.arange(clusters[probe]['metrics'].shape[0])

        # Decode per brain region
        for r, region in enumerate(np.unique(clusters_regions)):

            # Skip region if any of these conditions apply
            if region.islower():
                continue

            print('Processing region %s (%d of %d)' % (region, r + 1, len(np.unique(clusters_regions))))

            # Get clusters in this brain region
            clusters_in_region = [x for x, y in enumerate(clusters_regions)
                                  if (region == y) and (x in clusters_pass)]

            # Select spikes and clusters
            spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
            clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters,
                                                         clusters_in_region)]

            # Check if there are enough neurons in this brain region
            if np.unique(clus_region).shape[0] < MIN_NEURONS:
                continue

            # Process
            for t, trans in enumerate(transitions):
                if trials.loc[trans, 'probabilityLeft'] == 0.2:



