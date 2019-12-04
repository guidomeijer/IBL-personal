#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:34:22 2019

@author: guido
"""

from oneibl.one import ONE
one = ONE()

eids, ses_info = one.search(users='guido', dataset_types='spikes.times', details=True)
dtypes = ['spikes.amps', 'spikes.clusters', 'spikes.depths', 'spikes.samples', 'spikes.templates',
          'spikes.times',
          '_spikeglx_sync.channels', '_spikeglx_sync.polarities',
          '_spikeglx_sync.times', '_iblrig_RFMapStim.raw',
          '_iblrig_codeFiles.raw',
          '_iblrig_taskSettings.raw',
          'clusters.amps', 'clusters.channels', 'clusters.depths', 'clusters.metrics',
          'clusters.peakToTrough',
          'clusters.probes',
          'clusters.uuids',
          'clusters.waveforms',
          'clusters.waveformsChannels', 'channels.localCoordinates',
          'channels._phy_ids',
          'channels.probes',
          'channels.rawInd',]

for i, eid in enumerate(eids):
    d = one.load(eid, dataset_types=dtypes, download_only=True, dclass_output=True)