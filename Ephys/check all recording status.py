#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:11:21 2020

@author: guido
"""

from ibl_pipeline import acquisition, behavior, subject
from ibl_pipeline.analyses import behavior as behavior_analyses
import datajoint as dj
import pandas as pd
from uuid import UUID

ephys = dj.create_virtual_module('ephys', 'ibl_ephys')

from oneibl.one import ONE
one = ONE()

# grab insertions and behavioral QC from datajoint
sessions = (acquisition.Session & ephys.DefaultCluster & 
            (acquisition.SessionProject & 'session_project like "%brainwide%"'))
    
insertions_qc = subject.Subject * subject.SubjectLab * ephys.ProbeInsertion * sessions * (behavior_analyses.SessionTrainingStatus)
insertions_qc = insertions_qc.proj('session_uuid', 'session_lab', 
                                   'subject_nickname', behavior_qc_passed='good_enough_for_brainwide_map')
insertions = insertions_qc.fetch(format='frame').reset_index()
insertions['eid'] = [eid.urn[9:] for eid in insertions.session_uuid] # convert to string
insertions.shape

insertions['alyx_qc'] = 'empty'
qcs = ['critical', 'error', 'warning','not_set']
for qc in qcs:
    # grab those eids
    sess = list(one.alyx.rest('sessions', 'list', qc=qc))
    eids = [s['url'] for s in sess]
    eids = [e.split('/')[-1] for e in eids]
    insertions.loc[insertions['eid'].isin(eids), 'alyx_qc'] = qc

insertions.groupby(['alyx_qc', 'behavior_qc_passed', 'session_lab'])['session_start_time'].count()