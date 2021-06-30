#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 11:35:59 2021
By: Guido Meijer
"""

from oneibl.one import ONE
from atlaselectrophysiology.alignment_with_easyqc import viewer
one = ONE()

pid = "af2a0072-e17e-4368-b80b-1359bf6d4647"  # ZFM-01937 / 2021-04-08 / probe00

av = viewer(pid, one=one)