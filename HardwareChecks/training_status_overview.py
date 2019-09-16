#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:52:40 2019

@author: guido
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:30:25 2018

@author: guido
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from os.path import join
import seaborn as sns
import datajoint as dj
from ibl_pipeline import subject, acquisition, action, behavior, reference
from ibl_pipeline.analyses import behavior as behavior_analyses
import training_criteria_schemas as criteria_urai 

# Settings
path = '/home/guido/Figures/Behavior/'

# Query list of subjects
all_sub = subject.Subject * subject.SubjectLab & 'subject_birth_date > "2018-09-01"' & 'subject_line IS NULL OR subject_line="C57BL/6J"'
subjects = pd.DataFrame(all_sub)
        
mice = pd.DataFrame(0, index=np.unique(subjects['lab_name']), columns=['trained','in training','untrainable'])
for i, nickname in enumerate(subjects['subject_nickname']):
    print('Processing subject %s'%nickname)
    subj = subject.Subject & 'subject_nickname="%s"'%nickname
    session_start, training_status = (behavior_analyses.SessionTrainingStatus & subj).fetch('session_start_time', 'training_status')
    
    if np.sum(training_status == 'trained') > 0:
        mice.loc[subjects.iloc[i]['lab_name'],'trained'] = mice.loc[subjects.iloc[i]['lab_name'],'trained'] + 1
    elif np.sum(training_status == 'untrainable') > 0:
        mice.loc[subjects.iloc[i]['lab_name'],'untrainable'] = mice.loc[subjects.iloc[i]['lab_name'],'untrainable'] + 1
    elif np.sum(training_status == 'training in progress') > 1:
        mice.loc[subjects.iloc[i]['lab_name'],'in training'] = mice.loc[subjects.iloc[i]['lab_name'],'in training'] + 1
    else:
        continue

mice = mice.drop('hoferlab')

pos = list(range(len(mice['trained']))) 
width = 0.25 
fig, ax = plt.subplots(figsize=(10,5))
plt.bar(pos, mice['trained'], width, color='#228B22', label='trained') 
plt.bar([p + width for p in pos], mice['in training'], width, color='#F78F1E', label='in training') 
plt.bar([p + width*2 for p in pos], mice['untrainable'], width, color='#EE3224', label='untrainable') 
ax.set_ylabel('Number of mice')
ax.set_xticks([p + 1.5 * width for p in pos])
ax.set_xticklabels(mice.index)
for item in ax.get_xticklabels():
    item.set_rotation(60)
plt.xlim(min(pos)-width, max(pos)+width*4)
plt.legend()
plt.tight_layout()

plt.savefig(join(path, 'training_status.pdf'), dpi=300)
plt.savefig(join(path, 'training_status.png'), dpi=300)