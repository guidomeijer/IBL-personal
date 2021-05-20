#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:00:29 2021

@author: guido
"""

from iblvideo import run_session
from oneibl.one import ONE
one = ONE()

# session = one.search(task_protocol='_iblrig_NPH_tasks_trainingChoiceWorld')
session = one.search(task_protocol='_iblrig_tasks_opto_biasedChoiceWorld', subject='ZFM-01867')

for eid in session:
    print(f'Processing session {eid}')
    status = run_session(eid, machine='guido', cams=['left'], one=one, frames=10000)

