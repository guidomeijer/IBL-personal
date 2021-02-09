#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 09:03:38 2021

@author: guido
"""

from functions_pharmacology import fit_running_avg_model
import pandas as pd

sessions = pd.read_csv('pharmacology_sessions.csv', header=1)
sessions = sessions[sessions['Week'] == 1]

results_df = pd.DataFrame()
for i in range(len(sessions)):
    tau_pre = fit_running_avg_model(sessions.loc[i, 'Nickname'], sessions.loc[i, 'Pre-vehicle'])
    tau_drug = fit_running_avg_model(sessions.loc[i, 'Nickname'], sessions.loc[i, 'Drug'])
    tau_post = fit_running_avg_model(sessions.loc[i, 'Nickname'], sessions.loc[i, 'Post-vehicle'])
    results_df = results_df.append(pd.DataFrame(data={'tau': [tau_pre, tau_drug, tau_post],
                                                      'Subject': sessions.loc[i, 'Nickname'],
                                                      'condition': ['Pre-vehicle', 'Drug', 'Post-vehicle']}))