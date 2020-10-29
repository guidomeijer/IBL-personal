#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 08:57:24 2020

@author: guido
"""


from oneibl.one import ONE
one = ONE()
probe_id = one.alyx.rest('insertions', 'list', subject='SWC_014', date='2019-12-10', name='probe01')[0]['id']

traj = one.alyx.rest('trajectories', 'list', probe_insertion=probe_id, provenance='Ephys aligned histology track')[0]
alignment_keys = traj['json'].keys()
print(alignment_keys)

from ibllib.qc.alignment_qc import AlignmentQC

align_key = "2020-06-15T10:03:50_guido"  # change this to your chosen alignment key
align_qc = AlignmentQC(probe_id, one=one)
align_qc.resolve_manual(align_key)
