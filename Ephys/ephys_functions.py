# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:22:01 2020

@author: guido
"""

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from os.path import join
import pathlib
from ibllib.atlas import regions_from_allen_csv
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
            or (len(trials.stimOn_times) != len(trials.probabilityLeft))):
        print('stimOn_times do not match with probabilityLeft')
        return False
    return True


def classify(population_activity, trial_labels, classifier, cross_validation=None):
    """
    Classify trial identity (e.g. stim left/right) from neural population activity.

    Parameters
    ----------
    population_activity : 2D array (trials x neurons)
        population activity of all neurons in the population for each trial.
    trial_labels : 1D or 2D array
        identities of the trials, can be any number of groups, accepts integers and strings
    classifier : scikit-learn object
        which decoder to use, for example Gaussian with Multinomial likelihood:
                    from sklearn.naive_bayes import MultinomialNB
                    classifier = MultinomialNB()
    cross_validation : None or scikit-learn object
        which cross-validation method to use, for example 5-fold:
                    from sklearn.model_selection import KFold
                    cross_validation = KFold(n_splits=5)

    Returns
    -------

    """

    # Check input
    assert population_activity.shape[0] == trial_labels.shape[0]

    if cross_validation is None:
        # Fit the model on all the data
        classifier.fit(population_activity, trial_labels)
        pred = classifier.predict(population_activity)
        prob = classifier.predict_proba(population_activity)
        prob = prob[:, 1]
    else:
        pred = np.empty(trial_labels.shape[0])
        prob = np.empty(trial_labels.shape[0])
        for train_index, test_index in cross_validation.split(population_activity):
            # Fit the model to the training data and predict the held-out test data
            classifier.fit(population_activity[train_index], trial_labels[train_index])
            pred[test_index] = classifier.predict(population_activity[test_index])
            proba = classifier.predict_proba(population_activity[test_index])
            prob[test_index] = proba[:, 1]

    # Calcualte accuracy
    accuracy = accuracy_score(trial_labels, pred)
    return accuracy, pred, prob


def query_sessions(selection='all'):
    from oneibl.one import ONE
    one = ONE()

    if selection == 'all':
        # Query all ephysChoiceWorld sessions
        ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50')
    elif selection == 'aligned':
        # Query all sessions with at least one alignment
        ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50,'
                               'json__extended_qc__alignment_count__gt,0')
    elif selection == 'resolved':
        # Query all sessions with resolved alignment
         ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50,'
                               'json__extended_qc__alignment_resolved,True')
    elif selection == 'aligned-behavior':
        # Query sessions with at least one alignment and that meet behavior criterion
        ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50,'
                               'json__extended_qc__alignment_count__gt,0,'
                               'session__extended_qc__behavior,1')
    elif selection == 'resolved-behavior':
        # Query sessions with resolved alignment and that meet behavior criterion
        ins = one.alyx.rest('insertions', 'list',
                        django='session__project__name__icontains,ibl_neuropixel_brainwide_01,'
                               'session__qc__lt,50,'
                               'json__extended_qc__alignment_resolved,True,'
                               'session__extended_qc__behavior,1')
    else:
        ins = []

    # Get list of eids and probes
    all_eids = np.array([i['session'] for i in ins])
    all_probes = np.array([i['name'] for i in ins])
    eids = np.unique(all_eids)
    probes = []
    for i, eid in enumerate(eids):
        probes.append(all_probes[[s == eid for s in all_eids]])
    return eids, probes


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


def get_parent_region_name(acronyms):
    brainregions = regions_from_allen_csv()
    parent_region_names = []
    for i, acronym in enumerate(acronyms):
        try:
            regid = brainregions.id[np.argwhere(brainregions.acronym == acronym)]
            ancestors = brainregions.ancestors(regid)
            targetlevel = 6
            if sum(ancestors.level == targetlevel) == 0:
                parent_region_names.append(ancestors.name[-1])
            else:
                parent_region_names.append(ancestors.name[np.argwhere(
                                                ancestors.level == targetlevel)[0, 0]])
        except IndexError:
            parent_region_names.append(acronym)
    if len(parent_region_names) == 1:
        return parent_region_names[0]
    else:
        return parent_region_names


def get_full_region_name(acronyms):
    brainregions = regions_from_allen_csv()
    full_region_names = []
    for i, acronym in enumerate(acronyms):
        try:
            regname = brainregions.name[np.argwhere(brainregions.acronym == acronym).flatten()][0]
            full_region_names.append(regname)
        except IndexError:
            full_region_names.append(acronym)
    if len(full_region_names) == 1:
        return full_region_names[0]
    else:
        return full_region_names


def get_eid_list():
    eids = np.array([
            '465c44bd-2e67-4112-977b-36e1ac7e3f8c', 'db4df448-e449-4a6f-a0e7-288711e7a75a',
            '158d5d35-a2ab-4a76-87b0-51048c5d5283', '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b',
            'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', '36280321-555b-446d-9b7d-c2e17991e090',
            'bb6a5aae-2431-401d-8f6a-9fdd6de655a9', 'a71175be-d1fd-47a3-aa93-b830ea3634a1',
            '2199306e-488a-40ab-93cb-2d2264775578', '4d8c7767-981c-4347-8e5e-5d5fffe38534',
            '30e5937e-e86a-47e6-93ae-d2ae3877ff8e', 'e535fb62-e245-4a48-b119-88ce62a6fe67',
            '8435e122-c0a4-4bea-a322-e08e8038478f', 'b39752db-abdb-47ab-ae78-e8608bbf50ed',
            'fa704052-147e-46f6-b190-a65b837e605e', '3dd347df-f14e-40d5-9ff2-9c49f84d2157',
            'e5fae088-ed96-4d9b-82f9-dfd13c259d52', 'e49d8ee7-24b9-416a-9d04-9be33b655f40',
            'f9860a11-24d3-452e-ab95-39e199f20a93', 'b658bc7d-07cd-4203-8a25-7b16b549851b',
            '7622da34-51b6-4661-98ae-a57d40806008', 'bd456d8f-d36e-434a-8051-ff3997253802'])
    return eids

