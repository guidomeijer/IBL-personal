# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:22:01 2020

@author: guido
"""

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from os.path import join, dirname
import pandas as pd
import pathlib
from paths import DATA_PATH, FIG_PATH


def paths():
    """
    Make a file in the root of the repository called 'paths.py' with in it:

    DATA_PATH = '/path/to/Flatiron/data'
    FIG_PATH = '/path/to/save/figures'

    """
    save_path = join(pathlib.Path(__file__).parent.absolute(), 'Data')
    return DATA_PATH, FIG_PATH, save_path


def figure_style(font_scale=2, despine=False, trim=True):
    """
    Set style for plotting figures
    """
    sns.set(style="ticks", context="paper", font_scale=font_scale)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    if despine is True:
        sns.despine(trim=trim)
        plt.tight_layout()


def check_trials(trials):

    if trials is None:
        print('trials Bunch is None type')
        return False
    if trials.probabilityLeft is None:
        print('trials.probabilityLeft is None type')
        return False
    if len(trials.probabilityLeft[0].shape) > 0:
        print('trials.probabilityLeft is an array of tuples')
        return False
    if trials.probabilityLeft[0] != 0.5:
        print('trials.probabilityLeft does not start with 0.5')
        return False
    if ((not hasattr(trials, 'stimOn_times'))
            or (len(trials.feedback_times) != len(trials.feedbackType))
            or (len(trials.stimOn_times) != len(trials.probabilityLeft))):
        print('stimOn_times or feedback_times don not match with probabilityLeft')
        return False
    return True


def query_sessions(selection='all'):
    from oneibl.one import ONE
    one = ONE()

    if selection == 'all':
        # Query all ephysChoiceWorld sessions with histology
        sessions = one.alyx.rest('sessions', 'list',
                                 task_protocol='_iblrig_tasks_ephysChoiceWorld',
                                 project='ibl_neuropixel_brainwide',
                                 dataset_types = ['spikes.times', 'trials.probabilityLeft'],
                                 histology=True)
    elif selection == 'resolved':
        # Query all sessions with resolved alignment
        sessions = one.alyx.rest('insertions', 'list',
                                 django='json__extended_qc__alignment_resolved,True')
    return sessions


def sessions_with_region(brain_region):
    from oneibl.one import ONE
    one = ONE()

    # Query sessions with at least one channel in the specified region
    sessions = one.alyx.rest('sessions', 'list', atlas_acronym=brain_region,
                        task_protocol='_iblrig_tasks_ephysChoiceWorld',
                        dataset_types = ['spikes.times', 'trials.probabilityLeft'],
                        project='ibl_neuropixel_brainwide')
    return sessions


def combine_layers_cortex(regions, delete_duplicates=False):
    remove = ['1', '2', '3', '4', '5', '6a', '6b', '/']
    for i, region in enumerate(regions):
        for j, char in enumerate(remove):
            regions[i] = regions[i].replace(char, '')
    if delete_duplicates:
        regions = list(set(regions))
    return regions

