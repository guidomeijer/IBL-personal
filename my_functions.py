# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:22:01 2020

@author: guido
"""

import numpy as np
import seaborn as sns
import matplotlib
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import alf
from os.path import join
from brainbox.numerical import ismember
from ibllib.atlas import BrainRegions
from oneibl.one import ONE
one = ONE()


def paths():
    """
    Make a file in the root of the repository called 'paths.py' with in it:

    DATA_PATH = '/path/to/Flatiron/data'
    FIG_PATH = '/path/to/save/figures'
    SAVE_PATH = '/path/to/save/data'

    """
    from paths import DATA_PATH, FIG_PATH, SAVE_PATH
    return DATA_PATH, FIG_PATH, SAVE_PATH


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


def load_trials(eid, laser_stimulation=False, invert_choice=False):
    trials = pd.DataFrame()
    if laser_stimulation:
        (trials['stimOn_times'], trials['feedback_times'], trials['goCue_times'],
         trials['probabilityLeft'], trials['contrastLeft'], trials['contrastRight'],
         trials['feedbackType'], trials['choice'], trials['laser_stimulation'],
         trials['laser_probability']) = one.load(
                             eid, dataset_types=['trials.stimOn_times', 'trials.feedback_times',
                                                 'trials.goCue_times', 'trials.probabilityLeft',
                                                 'trials.contrastLeft', 'trials.contrastRight',
                                                 'trials.feedbackType', 'trials.choice',
                                                 '_ibl_trials.laser_stimulation',
                                                 '_ibl_trials.laser_probability'])
        if trials.loc[0, 'laser_stimulation'] is None:
            trials = trials.drop(columns=['laser_stimulation'])
        if trials.loc[0, 'laser_probability'] is None:
            trials = trials.drop(columns=['laser_probability'])
    else:
       (trials['stimOn_times'], trials['feedback_times'], trials['goCue_times'],
         trials['probabilityLeft'], trials['contrastLeft'], trials['contrastRight'],
         trials['feedbackType'], trials['choice']) = one.load(
                             eid, dataset_types=['trials.stimOn_times', 'trials.feedback_times',
                                                 'trials.goCue_times', 'trials.probabilityLeft',
                                                 'trials.contrastLeft', 'trials.contrastRight',
                                                 'trials.feedbackType', 'trials.choice'])
    trials['signed_contrast'] = trials['contrastRight']
    trials.loc[trials['signed_contrast'].isnull(), 'signed_contrast'] = -trials['contrastLeft']
    trials['correct'] = trials['feedbackType']
    trials.loc[trials['correct'] == -1, 'correct'] = 0
    trials['right_choice'] = -trials['choice']
    trials.loc[trials['right_choice'] == -1, 'right_choice'] = 0
    trials['stim_side'] = (trials['signed_contrast'] > 0).astype(int)
    trials.loc[trials['stim_side'] == 0, 'stim_side'] = -1
    trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastLeft'].isnull()),
               'stim_side'] = 1
    trials.loc[(trials['signed_contrast'] == 0) & (trials['contrastRight'].isnull()),
               'stim_side'] = -1
    if invert_choice:
        trials['choice'] = trials['choice'] + 2
        trials.loc[trials['choice'] == 3] = -1
    return trials


def check_trials(trials):

    if trials is None:
        print('trials is None type')
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
    if (('stimOn_times' not in trials.columns)
            or (len(trials.stimOn_times) != len(trials.probabilityLeft))):
        print('stimOn_times do not match with probabilityLeft')
        return False
    return True


def remap(ids, source='Allen', dest='Beryl', output='acronym'):
    br = BrainRegions()
    _, inds = ismember(ids, br.id[br.mappings[source]])
    ids = br.id[br.mappings[dest][inds]]
    if output == 'id':
        return br.id[br.mappings[dest][inds]]
    elif output == 'acronym':
        return br.get(br.id[br.mappings[dest][inds]])['acronym']


def query_sessions(selection='all', return_subjects=False):

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
    all_subjects = np.array([i['session_info']['subject'] for i in ins])
    eids, ind_unique = np.unique(all_eids, return_index=True)
    subjects = all_subjects[ind_unique]
    probes = []
    for i, eid in enumerate(eids):
        probes.append(all_probes[[s == eid for s in all_eids]])
    if return_subjects:
        return eids, probes, subjects
    else:
        return eids, probes


def sessions_with_region(acronym):
    from oneibl.one import ONE
    one = ONE()

    traj = one.alyx.rest('trajectories', 'list',
                        django='channels__brain_region__acronym,CP')


    # Query sessions with at least one channel in the specified region
    sessions = one.alyx.rest('sessions', 'list', atlas_acronym=brain_region,
                        task_protocol='_iblrig_tasks_ephysChoiceWorld',
                        dataset_types = ['spikes.times', 'trials.probabilityLeft'],
                        project='ibl_neuropixel_brainwide')
    return sessions


def combine_layers_cortex(regions):
    """
    Combine all layers of cortex
    """
    remove = ['1', '2', '3', '4', '5', '6a', '6b', '/']
    for i, region in enumerate(regions):
        if region[:2] == 'CA':
            continue
        for j, char in enumerate(remove):
            regions[i] = regions[i].replace(char, '')
    return regions


def get_parent_region_name(acronyms):
    brainregions = BrainRegions()
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
    brainregions = BrainRegions()
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


def criteria_opto_eids(eids, max_lapse=0.2, max_bias=0.3, min_trials=200):
    use_eids = []
    for j, eid in enumerate(eids):
        try:
            trials = load_trials(eid, laser_stimulation=True)
            lapse_l = 1 - (np.sum(trials.loc[trials['signed_contrast'] == -1, 'choice'] == 1)
                           / trials.loc[trials['signed_contrast'] == -1, 'choice'].shape[0])
            lapse_r = 1 - (np.sum(trials.loc[trials['signed_contrast'] == 1, 'choice'] == -1)
                           / trials.loc[trials['signed_contrast'] == 1, 'choice'].shape[0])
            bias = np.abs(0.5 - (np.sum(trials.loc[trials['signed_contrast'] == 0, 'choice'] == 1)
                                 / np.shape(trials.loc[trials['signed_contrast'] == 0, 'choice'] == 1)[0]))
            details = one.get_details(eid)
            if ((lapse_l < max_lapse) & (lapse_r < max_lapse) & (trials.shape[0] > min_trials)
                    & (bias < max_bias) & ('laser_stimulation' in trials.columns)):
                use_eids.append(eid)
            elif 'laser_stimulation' not in trials.columns:
                print('No laser_stimulation data for %s %s' % (details['subject'], details['start_time'][:10]))
            else:
                print('%s %s excluded (n_trials: %d, lapse_l: %.2f, lapse_r: %.2f, bias: %.2f)'
                      % (details['subject'], details['start_time'][:10], trials.shape[0], lapse_l, lapse_r, bias))
        except Exception:
            print('Could not load session %s' % eid)
    return use_eids


def fit_psychfunc(stim_levels, n_trials, proportion):
    # Fit a psychometric function with two lapse rates
    #
    # Returns vector pars with [threshold, bias, lapselow, lapsehigh]
    from ibl_pipeline.utils import psychofit as psy
    assert(stim_levels.shape == n_trials.shape == proportion.shape)

    pars, _ = psy.mle_fit_psycho(np.vstack((stim_levels, n_trials, proportion)),
                                 P_model='erf_psycho_2gammas',
                                 parstart=np.array([0, 20, 0.05, 0.05]),
                                 parmin=np.array([-100, 2, 0, 0]),
                                 parmax=np.array([100, 100., 1, 1]))
    return pars


def plot_psychometric(trials, ax, **kwargs):
    from ibl_pipeline.utils import psychofit as psy
    if trials['signed_contrast'].max() <= 1:
        trials['signed_contrast'] = trials['signed_contrast'] * 100

    stim_levels = np.sort(trials['signed_contrast'].unique())
    pars = fit_psychfunc(stim_levels, trials.groupby('signed_contrast').size(),
                         trials.groupby('signed_contrast').mean()['right_choice'])

    # plot psychfunc
    sns.lineplot(x=np.arange(-27, 27), y=psy.erf_psycho_2gammas(pars, np.arange(-27, 27)),
                 ax=ax, **kwargs)

    # plot psychfunc: -100, +100
    sns.lineplot(x=np.arange(-36, -31), y=psy.erf_psycho_2gammas(pars, np.arange(-103, -98)),
                 ax=ax, **kwargs)
    sns.lineplot(x=np.arange(31, 36), y=psy.erf_psycho_2gammas(pars, np.arange(98, 103)),
                 ax=ax, **kwargs)

    # now break the x-axis
    trials['signed_contrast'].replace(-100, -35)
    trials['signed_contrast'].replace(100, 35)

    # plot datapoints with errorbars on top
    sns.lineplot(x=trials['signed_contrast'], y=trials['right_choice'], ax=ax,
                     **{**{'err_style':"bars",
                     'linewidth':0, 'linestyle':'None', 'mew':0.5,
                     'marker':'o', 'ci':68}, **kwargs})

    ax.set(xticks=[-35, -25, -12.5, 0, 12.5, 25, 35], xlim=[-40, 40], ylim=[0, 1.02],
           yticks=[0, 0.25, 0.5, 0.75, 1], yticklabels=['0', '25', '50', '75', '100'],
           ylabel='Right choices', xlabel='Contrast (%)')
    ax.set_xticklabels(['-100', '-25', '-12.5', '0', '12.5', '25', '100'],
                       size='small')
    break_xaxis()


def break_xaxis(y=-0.004, **kwargs):

    # axisgate: show axis discontinuities with a quick hack
    # https://twitter.com/StevenDakin/status/1313744930246811653?s=19
    # first, white square for discontinuous axis
    plt.text(-30, y, '-', fontsize=14, fontweight='bold',
             horizontalalignment='center', verticalalignment='center',
             color='w')
    plt.text(30, y, '-', fontsize=14, fontweight='bold',
             horizontalalignment='center', verticalalignment='center',
             color='w')

    # put little dashes to cut axes
    plt.text(-30, y, '/ /', horizontalalignment='center',
             verticalalignment='center', fontsize=12, fontweight='bold')
    plt.text(30, y, '/ /', horizontalalignment='center',
             verticalalignment='center', fontsize=12, fontweight='bold')


def fit_prob_choice_model(trials, previous_trials=6):

    data = trials.copy()
    data = data[['choice', 'correct', 'signed_contrast', 'probabilityLeft']]

    # Rewardeded choices:
    data.loc[(data['choice'] == 0) & (data['correct'].isnull()), 'rchoice'] = 0  # NoGo trials
    data.loc[(data['choice'] == -1) & (data['correct'] == 0), 'rchoice'] = 0
    data.loc[(data['choice'] == -1) & (data['correct'] == 1), 'rchoice'] = 1
    data.loc[(data['choice'] == 1) & (data['correct'] == 1), 'rchoice'] = -1
    data.loc[(data['choice'] == 0) & (data['correct'].isnull()), 'rchoice'] = 0  # NoGo trials
    data.loc[(data['choice'] == 1) & (data['correct'] == 0), 'rchoice'] = 0

    # Unrewarded choices:
    data.loc[(data['choice'] == 0) & (data['correct'].isnull()), 'uchoice'] = 0  # NoGo trials
    data.loc[(data['choice'] == -1) & (data['correct'] == 0), 'uchoice'] = 1
    data.loc[(data['choice'] == -1) & (data['correct'] == 1), 'uchoice'] = 0
    data.loc[(data['choice'] == 1) & (data['correct'] == 1), 'uchoice'] = 0
    data.loc[(data['choice'] == 0) & (data['correct'].isnull()), 'uchoice'] = 0  # NoGo trials
    data.loc[(data['choice'] == 1) & (data['correct'] == 0), 'uchoice'] = -1

    # Shift rewarded and unrewarded predictors by one
    for i in range(previous_trials):
        data.loc[:, 'rchoice-%s' % str(i+1)] = data['rchoice'].shift(
                                                    periods=i+1, fill_value=0).to_numpy()
        data.loc[:, 'uchoice-%s' % str(i+1)] = data['uchoice'].shift(
                                                    periods=i+1, fill_value=0).to_numpy()

    # Drop any nan trials
    data.dropna(inplace=True)

    # Make sensory predictors (no 0 predictor)
    contrasts = [.25, 1, .125, .0625]
    for i in contrasts:
        data.loc[(data['signed_contrast'].abs() == i), i] = np.sign(
                    data.loc[(data['signed_contrast'].abs() == i), 'signed_contrast'].to_numpy())
        data[i].fillna(0,  inplace=True)

    # Add block probability predictor
    data.loc[(data['probabilityLeft'] == 0.5), 'block'] = 0
    data.loc[(data['probabilityLeft'] == 0.2), 'block'] = 1
    data.loc[(data['probabilityLeft'] == 0.8), 'block'] = -1

    # Make choice in between 0 and 1 -> 1 for right and 0 for left
    data.loc[data['choice'] == 1, 'choice'] = 0
    data.loc[data['choice'] == -1, 'choice'] = 1

    # Create predictor matrix
    exog = data.copy()
    exog.drop(columns=['correct', 'signed_contrast', 'choice', 'probabilityLeft',
                       'rchoice', 'uchoice'],
              inplace=True)
    exog = sm.add_constant(exog)

    # Fit model
    logit_model = sm.Logit(data['choice'], exog)
    result = logit_model.fit()
    weights = result.params.rename(index=str)
    return weights


def fit_psytrack(trials, previous_trials=0):
    from psytrack.hyperOpt import hyperOpt

    # Load data
    contrast_l = trials['contrastLeft'].values
    contrast_r = trials['contrastRight'].values
    prob_l = trials['probabilityLeft'].values
    correct = trials['correct'].values
    choice = trials['choice'].values
    day_length = trials.groupby('probabilityLeft').size().values

    # Change values to what the model input
    choice[choice == 1] = 2
    choice[choice == -1] = 1
    correct[correct == -1] = 0
    contrast_l[np.isnan(contrast_l)] = 0
    contrast_r[np.isnan(contrast_r)] = 0

    # Transform visual contrast
    p = 3.5
    contrast_l_transform = np.tanh(contrast_l * p) / np.tanh(p)
    contrast_r_transform = np.tanh(contrast_r * p) / np.tanh(p)

    # Reformat the stimulus vectors to matrices which include previous trials
    s1_trans = contrast_l_transform
    s2_trans = contrast_r_transform
    for i in range(1, 10):
        s1_trans = np.column_stack((s1_trans, np.append([contrast_l_transform[0]]*(i+i),
                                                        contrast_l_transform[i:-i])))
        s2_trans = np.column_stack((s2_trans, np.append([contrast_r_transform[0]]*(i+i),
                                                        contrast_r_transform[i:-i])))

    # Create input dict
    D = {'name': '',
         'y': choice,
         'correct': correct,
         'dayLength': day_length,
         'inputs': {'s1': s1_trans, 's2': s2_trans}
         }

    # Model parameters
    weights = {'bias': 1,
               's1': previous_trials+1,
               's2': previous_trials+1}
    K = np.sum([weights[i] for i in weights.keys()])
    hyper = {'sigInit': 2**4.,
             'sigma': [2**-4.]*K,
             'sigDay': [2**-4.]*K}
    optList = ['sigInit', 'sigma', 'sigDay']

    # Fit model
    print('Fitting model..')
    hyp, evd, wMode, hess = hyperOpt(D, hyper, weights, optList)

    return wMode, prob_l, hyp
