#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 12:18:30 2019

@author: guido
"""

from pathlib import Path
import alf.io
from oneibl.one import ONE

# Query sessions with available DLC data using ONE
one = ONE()
dtypes = ['camera.dlc', 'camera.times']
eids = one.search(dataset_types=dtypes)

# Loop over sessions
for i, eid in enumerate(eids):

    d = one.load(eid, dataset_types=dtypes, download_only=True, dclass_output=True)



    """
    , download_only=True, dclass_output=True)
    ses_path = Path(d.local_path[0]).parent
    segments = alf.io.load_object(ses_path, '_ibl_leftCamera', short_keys=True)


    dlc, timestamps = one.load(eid, dataset_types=dtypes)
    """