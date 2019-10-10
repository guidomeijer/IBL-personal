#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:54:54 2019

@author: guido
"""

from oneibl.one import ONE
import numpy as np

# Query sessions with available DLC data using ONE
one = ONE()
dtypes = ['camera.dlc', 'camera.times', '_iblrig_Camera.raw']
eids = one.search(dataset_types=dtypes)

# Loop over sessions
for i, eid in enumerate(eids):
    if np.mod(i+1, 5) == 0:
        print('Downloading video of session %d of %d' % (i+1, len(eids)))

    # Load in data
    d = one.load(eid, dataset_types=dtypes, download_only=True, dclass_output=True)
