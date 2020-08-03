#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 14:43:03 2020

@author: guido
"""


from ibllib.qc.bpodqc_metrics import BpodQC
from oneibl.one import ONE
import pandas as pd
from ibllib.qc.qcplots import boxplot_metrics, barplot_passed
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
one = ONE()
session_path = Path(r'/home/guido/Flatiron/mainenlab/Subjects/test/2020-07-31/002')
bpodqc = BpodQC(session_path, one=one, ensure_data=False, lazy=False)
session_name = "/".join(Path(bpodqc.session_path).parts[-3:])
with open(session_path / 'bpodqc_metrics.p', 'wb') as f:
    pickle.dump(bpodqc.metrics, f)
with open(session_path / 'bpodqc_passed.p', 'wb') as f:
    pickle.dump(bpodqc.passed, f)
bpodqc.metrics.pop('_bpod_wheel_integrity')
bpodqc.passed.pop('_bpod_wheel_integrity')
df_metric = pd.DataFrame.from_dict(bpodqc.metrics)
df_passed = pd.DataFrame.from_dict(bpodqc.passed)
boxplot_metrics(df_metric, title=session_name)
barplot_passed(df_passed, title=session_name, save_path=session_path)
plt.show()