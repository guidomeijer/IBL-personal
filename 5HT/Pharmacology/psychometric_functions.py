#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:56:30 2020

@author: guido
"""

import numpy as np


def plot_psychometric_data(trials, ax):
    """Plot psychometric data """

    # Prepare dataframe
    trials = trials.loc[trials['trial_response_choice'] != 'No Go']
    trials['signed_contrast'] = (trials['trial_stim_contrast_left']
                                 - trials['trial_stim_contrast_right']) * 100
    trials.loc[trials['trial_response_choice'] == 'CCW', 'trial_response_choice'] = -1
    trials.loc[trials['trial_response_choice'] == 'CW', 'trial_response_choice'] = 1

    # Count "left" and "right" responses for each signed contrast level
    resp = pd.Series(index=np.unique(trials['signed_contrast']))
    left_resp = trials[(trials['trial_response_choice'] == -1)].groupby(
                                            ['signed_contrast']).count()['trial_id'].values
    right_resp = trials[(trials['trial_response_choice'] == 1)].groupby(
                                            ['signed_contrast']).count()['trial_id'].values

    # Plot data
    frac_resp = right_resp / (left_resp + right_resp)
    err_bar = np.sqrt(frac_resp*(1-frac_resp) / (left_resp + right_resp))
    ax.errorbar(x=left_resp.index, y=frac_resp, yerr=err_bar)
