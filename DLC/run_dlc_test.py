#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:31:11 2019

@author: guido
"""


from pathlib import Path
import alf.io
from oneibl.one import ONE

# Query sessions with available DLC data using ONE
one = ONE()
dtypes = ['camera.dlc', 'camera.times', '_iblrig_Camera.raw']
eids = one.search(dataset_types=dtypes)

# Loop over sessions
for i, eid in enumerate(eids):
    print('Subject %d of %d' % (i+1, len(eids)))
    dlc_data = one.load(eid, dataset_types=dtypes, download_only=True, dclass_output=True)
