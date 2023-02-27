#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 16:31:39 2021
By: Guido Meijer
"""

# Data exploration GUI
from data_exploration_gui.data_explore_gui import viewer as data_viewer
from atlaselectrophysiology.alignment_with_easyqc import viewer as alignment_viewer
from one.api import ONE
one = ONE()
pid = 'eeb27b45-5b85-4e5c-b6ff-f639ca5687de'
dv2 = data_viewer(pid, one=one, spike_collection='ks2_preproc_tests', title='new')
dv = data_viewer(pid, one=one, title='original')

av = alignment_viewer(pid, one=one)
av2 = alignment_viewer(pid, spike_collection='ks2_preproc_tests')