
"""
Created on Thu Nov 26 10:10:54 2020
Decode left/right block identity from stimulus period
@author: Guido Meijer
"""

from os.path import join
import numpy as np
from brainbox.task import generate_pseudo_session
from brainbox.population import _get_spike_counts_in_bins
import pandas as pd
import alf
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from ephys_functions import paths, query_sessions, check_trials, combine_layers_cortex
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

# Settings
MIN_NEURONS = 5  # min neurons per region
DECODER = 'bayes-multinomial'
VALIDATION = 'kfold-interleaved'
INCL_NEURONS = 'all'  # all or no_drift
INCL_SESSIONS = 'aligned-behavior'  # all, aligned, resolved, aligned-behavior or resolved-behavior
NUM_SPLITS = 5
CHANCE_LEVEL = 'pseudo-sessions'  # pseudo-blocks, phase-rand, shuffle or none
ITERATIONS = 100  # for null distribution estimation
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')
DOWNLOAD_TRIALS = False
PRE_TIME = 0
POST_TIME = 0.3

# %% Initialize

# Query session list
sessions = query_sessions(selection=INCL_SESSIONS)

# Detect data format
if type(sessions) == pd.DataFrame:
    ses_type = 'datajoint'
elif 'model' in sessions[0]:
    ses_type = 'insertions'
else:
    ses_type = 'sessions'

# Initialize decoder
clf = MultinomialNB()


def decode(pop_vector, trial_ids, num_splits, interleaved):

    # Cross-validation
    if interleaved:
        cv = KFold(n_splits=num_splits, shuffle=True).split(pop_vector)
    else:
        cv = KFold(n_splits=num_splits, shuffle=False).split(pop_vector)

    # Loop over the splits into train and test
    y_pred = np.zeros(trial_ids.shape)
    y_probs = np.zeros(trial_ids.shape)
    for train_index, test_index in cv:

        # Fit the model to the training data
        clf.fit(pop_vector[train_index], trial_ids[train_index])

        # Predict the test data
        y_pred[test_index] = clf.predict(pop_vector[test_index])

        # Get the probability of the prediction for ROC analysis
        probs = clf.predict_proba(pop_vector[test_index])
        y_probs[test_index] = probs[:, 1]  # keep positive only

    return y_pred, y_probs


def balanced_trial_set(trials):
    # A balanced random subset of left and right trials from the two blocks
    left_stim = (trials.contrastLeft > 0) & (trials.probabilityLeft == 0.2)
    left_stim[np.random.choice(np.where(((trials.contrastLeft > 0)
                                         & (trials.probabilityLeft == 0.8)))[0],
                               size=np.sum(left_stim), replace=False)] = True
    right_stim = (trials.contrastRight > 0) & (trials.probabilityLeft == 0.8)
    right_stim[np.random.choice(np.where(((trials.contrastRight > 0)
                                         & (trials.probabilityLeft == 0.2)))[0],
                               size=np.sum(right_stim), replace=False)] = True
    incl_trials = (left_stim | right_stim
                   | (((trials.contrastLeft == 0) | (trials.contrastRight == 0))
                      & (trials.probabilityLeft != 0.5)))
    return incl_trials


# %% MAIN
decoding_result = pd.DataFrame()
for i in range(len(sessions)):
    print('\nProcessing session %d of %d' % (i+1, len(sessions)))

    # Extract eid based on data format
    if ses_type == 'insertions':
        eid = sessions[i]['session']
    elif ses_type == 'datajoint':
        eid = sessions.loc[i, 'session_eid']
    elif ses_type == 'sessions':
        eid = sessions[i]['url'][-36:]

    # Load in data
    try:
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
                                                                    eid, aligned=True, one=one)
        ses_path = one.path_from_eid(eid)
        if DOWNLOAD_TRIALS:
            _ = one.load(eid, dataset_types=['trials.stimOn_times', 'trials.probabilityLeft',
                                             'trials.contrastLeft', 'trials.contrastRight',
                                             'trials.feedbackType', 'trials.choice',
                                             'trials.feedback_times'],
                         download_only=True, clobber=True)
        trials = alf.io.load_object(join(ses_path, 'alf'), 'trials')
    except Exception as error_message:
        print(error_message)
        continue

    # Check data integrity
    if check_trials(trials) is False:
        continue

    # Extract session data depending on whether input is a list of sessions or insertions
    if ses_type == 'insertions':
        subject = sessions[i]['session_info']['subject']
        date = sessions[i]['session_info']['start_time'][:10]
        probes_to_use = [sessions[i]['name']]
    elif ses_type == 'datajoint':
        subject = sessions.loc[i, 'subject_nickname']
        date = str(sessions.loc[i, 'session_end_time'].date())
        probes_to_use = spikes.keys()
    else:
        subject = sessions[i]['subject']
        date = sessions[i]['start_time'][:10]
        probes_to_use = spikes.keys()


    # Decode per brain region
    for p, probe in enumerate(probes_to_use):
        print('Processing %s (%d of %d)' % (probe, p + 1, len(probes_to_use)))

        # Check if histology is available for this probe
        if not hasattr(clusters[probe], 'acronym'):
            continue

        # Get brain regions and combine cortical layers
        regions = combine_layers_cortex(np.unique(clusters[probe]['acronym']))

        # Decode per brain region
        for r, region in enumerate(np.unique(regions)):
            print('Decoding region %s (%d of %d)' % (region, r + 1, len(np.unique(regions))))

            # Get clusters in this brain region
            region_clusters = combine_layers_cortex(clusters[probe]['acronym'])
            clusters_in_region = clusters[probe].metrics.cluster_id[region_clusters == region]

            # Select spikes and clusters
            spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
            clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters,
                                                         clusters_in_region)]

            # Check if there are enough neurons in this brain region
            if np.unique(clus_region).shape[0] < MIN_NEURONS:
                continue

            # Get matrix of all neuronal responses
            times = np.column_stack(((trials.stimOn_times - PRE_TIME),
                                     (trials.stimOn_times + POST_TIME)))
            pop_vector, cluster_ids = _get_spike_counts_in_bins(spks_region, clus_region, times)
            pop_vector = pop_vector.T

            # Subselect trials to balance stimulus sides
            decode_result = pd.DataFrame()
            decode_chance = pd.DataFrame()
            for k in range(ITERATIONS):

                # Select balanced trial set
                incl_trials = balanced_trial_set(trials)

                # Select activity matrix and trial ids for this iteration
                this_pop_vector = pop_vector[incl_trials]
                trial_ids = (trials.probabilityLeft[incl_trials] == 0.2).astype(int)

                # Decode
                if VALIDATION[6:] == 'interleaved':
                    y_pred, y_probs = decode(this_pop_vector, trial_ids, NUM_SPLITS, True)
                else:
                    y_pred, y_probs = decode(this_pop_vector, trial_ids, NUM_SPLITS, False)

                # Calculate performance metrics and confusion matrix
                decode_result.loc[k, 'accuracy'] = accuracy_score(trial_ids, y_pred)
                decode_result.loc[k, 'f1'] = f1_score(trial_ids, y_pred)
                decode_result.loc[k, 'auroc'] = roc_auc_score(trial_ids[~np.isnan(y_probs)],
                                                              y_probs[~np.isnan(y_probs)])

                # Decode pseudo session
                pseudo_trials = generate_pseudo_session(trials)
                incl_pseudo_trials = balanced_trial_set(pseudo_trials)

                # Select activity matrix and trial ids for this iteration
                this_pseudo_pop_vector = pop_vector[incl_pseudo_trials]
                pseudo_trial_ids = (trials.probabilityLeft[incl_pseudo_trials] == 0.2).astype(int)

                if VALIDATION[6:] == 'interleaved':
                    y_pred, y_probs = decode(this_pseudo_pop_vector, pseudo_trial_ids,
                                             NUM_SPLITS, True)
                else:
                    y_pred, y_probs = decode(this_pseudo_pop_vector, pseudo_trial_ids,
                                             NUM_SPLITS, False)

                # Calculate performance metrics and confusion matrix
                decode_chance.loc[k, 'accuracy'] = accuracy_score(pseudo_trial_ids, y_pred)
                decode_chance.loc[k, 'f1'] = f1_score(pseudo_trial_ids, y_pred)
                decode_chance.loc[k, 'auroc'] = roc_auc_score(pseudo_trial_ids[~np.isnan(y_probs)],
                                                              pseudo_trial_ids[~np.isnan(y_probs)])

            # Calculate p-values
            p_accuracy = (np.sum(decode_chance['accuracy'] > decode_result['accuracy'].mean())
                          / decode_chance['accuracy'].shape[0])
            p_f1 = (np.sum(decode_chance['f1'] > decode_result['f1'].mean())
                          / decode_chance['f1'].shape[0])
            p_auroc = (np.sum(decode_chance['auroc'] > decode_result['auroc'].mean())
                          / decode_chance['auroc'].shape[0])

            # Add to dataframe
            decoding_result = decoding_result.append(pd.DataFrame(
                index=[decoding_result.shape[0] + 1], data={'subject': subject,
                                 'date': date,
                                 'eid': eid,
                                 'probe': probe,
                                 'region': region,
                                 'f1': decode_result['f1'].mean(),
                                 'accuracy': decode_result['accuracy'].mean(),
                                 'auroc': decode_result['auroc'].mean(),
                                 'chance_accuracy': decode_chance['accuracy'].mean(),
                                 'chance_f1': decode_chance['f1'].mean(),
                                 'chance_auroc': decode_chance['auroc'].mean(),
                                 'p_accuracy': p_accuracy,
                                 'p_f1': p_f1,
                                 'p_auroc': p_auroc,
                                 'n_trials': trial_ids.shape[0],
                                 'n_neurons': np.unique(clus_region).shape[0]}))

    decoding_result.to_pickle(join(SAVE_PATH, DECODER,
                    ('block-stim_%s_%s_%s_%s_cells.p' % (CHANCE_LEVEL, VALIDATION,
                                                             INCL_SESSIONS, INCL_NEURONS))))
