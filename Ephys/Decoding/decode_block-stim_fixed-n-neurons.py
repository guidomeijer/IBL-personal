
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
from my_functions import paths, query_sessions, check_trials, combine_layers_cortex, classify
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

# Settings
MIN_NEURONS = 15  # min neurons per region
N_NEURONS = 15  # number of neurons to use for decoding
DECODER = 'bayes-multinomial'
VALIDATION = 'kfold-interleaved'
INCL_SESSIONS = 'aligned-behavior'  # all, aligned, resolved, aligned-behavior or resolved-behavior
NUM_SPLITS = 5
CHANCE_LEVEL = 'pseudo-session'  # pseudo-blocks, phase-rand, shuffle or none
ITERATIONS = 1  # for null distribution estimation
N_TRIAL_PICK = 50  # number of times to randomly subselect trials
N_NEURON_PICK = 50  # number of times to randomly subselect neurons
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')
DOWNLOAD_TRIALS = False
PRE_TIME = 0
POST_TIME = 0.3

# %% Initialize

# Query session list
eids, probes = query_sessions(selection=INCL_SESSIONS)

# Initialize decoder
clf = MultinomialNB()
cv = KFold(n_splits=NUM_SPLITS, shuffle=True)


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
for i in range(len(eids)):
    print('\nProcessing session %d of %d' % (i+1, len(eids)))

    # Load in data
    eid = eids[i]
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

    # Extract session data
    ses_info = one.get_details(eid)
    subject = ses_info['subject']
    date = ses_info['start_time'][:10]
    probes_to_use = probes[i]

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
            if region.islower():
                continue
            if 'metrics' not in clusters[probe]:
                continue
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

            decode_subselects = np.empty(N_NEURON_PICK)
            decode_subselects_null = np.empty(N_NEURON_PICK)
            for n in range(N_NEURON_PICK):

                # Subselect neurons
                use_neurons = np.random.choice(clusters_in_region, N_NEURONS, replace=False)

                decode_trials = np.empty(N_TRIAL_PICK)
                decode_null = np.empty(N_TRIAL_PICK)
                for t in range(N_TRIAL_PICK):

                    # Select balanced trial set
                    incl_trials = balanced_trial_set(trials)
                    trial_ids = (trials.probabilityLeft[incl_trials] == 0.2).astype(int)

                    # Decode
                    decode_trials[t] = classify(
                        pop_vector[np.ix_(incl_trials, np.isin(cluster_ids, use_neurons))],
                        trial_ids, clf, cv)[0]

                    this_decode_null = np.empty(ITERATIONS)
                    for j in range(ITERATIONS):

                        # Decode pseudo session
                        try:
                            pseudo_trials = generate_pseudo_session(trials)
                            incl_pseudo_trials = balanced_trial_set(pseudo_trials)
                            pseudo_trial_ids = (trials.probabilityLeft[
                                                        incl_pseudo_trials] == 0.2).astype(int)

                            # Decode
                            this_decode_null[j] = classify(
                                pop_vector[np.ix_(incl_pseudo_trials, np.isin(cluster_ids,
                                                                              use_neurons))],
                                pseudo_trial_ids, clf, cv)[0]
                        except:
                            this_decode_null[j] = np.nan

                    decode_null[t] = np.nanmean(this_decode_null)

            # Get means
            decode_subselects[n] = np.mean(decode_trials)
            decode_subselects_null[n] = np.mean(decode_null)

            # Add to dataframe
            decoding_result = decoding_result.append(pd.DataFrame(
                index=[decoding_result.shape[0] + 1], data={'subject': subject,
                                 'date': date,
                                 'eid': eid,
                                 'probe': probe,
                                 'region': region,
                                 'accuracy': decode_subselects.mean(),
                                 'chance_accuracy': decode_subselects_null.mean(),
                                 'n_trials': np.sum(incl_trials),
                                 'n_neurons': N_NEURONS,
                                 'n_trial_pick': N_TRIAL_PICK,
                                 'iterations': ITERATIONS}))

    decoding_result.to_pickle(join(SAVE_PATH, DECODER,
                    ('block-stim_%s_%s_%s_%s_cells.p' % (CHANCE_LEVEL, VALIDATION,
                                                         INCL_SESSIONS, N_NEURONS))))

# Drop duplicates
decoding_result = decoding_result.reset_index()
decoding_result = decoding_result[~decoding_result.duplicated(subset=['region', 'eid', 'probe'])]
