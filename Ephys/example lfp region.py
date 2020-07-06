#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:31:34 2020

@author: guido
"""

from ibllib.io import spikeglx
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

eid = 'c9fec76e-7a20-4da4-93ad-04510a89473b'
region = 'VPL'

spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)


channels = bbone.load_channel_locations(eid, one=one)
channel_select = channels['probe00']['acronym'] == region   
    
lf_paths = one.load(eid, dataset_types=['ephysData.raw.lf', 'ephysData.raw.meta',
                                        'ephysData.raw.ch'], download_only=True)
raw = spikeglx.Reader(lf_paths[0])
signal = raw.read(nsel=slice(None, 100000, None), csel=channel_select)[0]