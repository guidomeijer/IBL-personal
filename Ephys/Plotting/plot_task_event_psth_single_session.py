# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

@author: guido
"""

from os import mkdir
from os.path import join, isdir
import matplotlib.pyplot as plt
import brainbox as bb
import seaborn as sns
import shutil
import numpy as np
from ephys_functions import paths
from brainbox.io.one import load_spike_sorting, load_channel_locations
from oneibl.one import ONE
one = ONE()

# Settings
ALPHA = 0.05
SUBJECT = 'ZM_1897'
DATE = '2019-12-02'
PROBE = '01'
OVERWRITE = True

# Set path to save plots
DATA_PATH, FIG_PATH, _ = paths()
FIG_PATH = join(FIG_PATH, 'PSTH')

# Load in data
eids = one.search(subject=SUBJECT, date_range=DATE)
channels = load_channel_locations(eids[0], one=one)
spikes, clusters = load_spike_sorting(eids[0], one=one)
trials = one.load_object(eids[0], 'trials')

# Only use single units
probe = 'probe' + PROBE
spikes[probe].times = spikes[probe].times[np.isin(
        spikes[probe].clusters, clusters[probe].metrics.cluster_id[
            clusters[probe].metrics.ks2_label == 'good'])]
spikes[probe].clusters = spikes[probe].clusters[np.isin(
        spikes[probe].clusters, clusters[probe].metrics.cluster_id[
            clusters[probe].metrics.ks2_label == 'good'])]

# Make directory
if (isdir(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe)))
        and (OVERWRITE is True)):
    shutil.rmtree(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe)))
if not isdir(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe))):
    mkdir(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe)))

# Stimulus onset PSTH
sig_units = bb.task.responsive_units(spikes[probe].times, spikes[probe].clusters,
                                     trials.stimOn_times, alpha=ALPHA)[0]
print('%d out of %d units are significantly responsive to stimulus onset' % (
                            len(sig_units), len(np.unique(spikes[probe].clusters))))

if not isdir(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe), 'StimOn')):
    mkdir(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe), 'StimOn'))
    for n, cluster in enumerate(sig_units):
        # Get brain region of neuron
        region = channels[probe].acronym[clusters[probe].channels[
                             clusters[probe].metrics.cluster_id == cluster][0]]
        region = region.replace('/', '-')

        fig, ax = plt.subplots(1, 1)
        sns.set(style="ticks", context="paper", font_scale=2)
        bb.plot.peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                          trials.stimOn_times, cluster,
                                          t_before=1, t_after=2,
                                          pethline_kwargs={'color': sns.color_palette()[0],
                                                           'lw': 2},
                                          errbar_kwargs={'color': sns.color_palette()[0],
                                                         'alpha': 0.5},
                                          eventline_kwargs={'color': 'black',
                                                            'linestyle': 'dashed',
                                                            'alpha': 0.75},
                                          error_bars='sem', ax=ax)
        # plt.title('Stimulus Onset')
        plt.ylabel('Firing rate (spikes/s)')
        plt.xlabel('Time (s) from stimulus onset')
        plt.tight_layout()
        plt.savefig(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe), 'StimOn',
                         '%s_n%d' % (region, cluster)))
        plt.close(fig)

# Reward PSTH
sig_units = bb.task.responsive_units(spikes[probe].times, spikes[probe].clusters,
                                     trials.feedback_times[trials.feedbackType == 1],
                                     alpha=ALPHA)[0]
print('%d out of %d units are significantly responsive to reward delivery' % (
                            len(sig_units), len(np.unique(spikes[probe].clusters))))

if not isdir(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe), 'Reward')):
    mkdir(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe), 'Reward'))
    for n, cluster in enumerate(sig_units):
        # Get brain region of neuron
        region = channels[probe].acronym[clusters[probe].channels[
                             clusters[probe].metrics.cluster_id == cluster][0]]
        region = region.replace('/', '-')

        fig, ax = plt.subplots(1, 1)
        sns.set(style="ticks", context="paper", font_scale=2)
        bb.plot.peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                          trials.feedback_times[
                                              trials.feedbackType == 1],
                                          cluster, t_before=1, t_after=2,
                                          pethline_kwargs={'color': sns.color_palette()[1],
                                                           'lw': 2},
                                          errbar_kwargs={'color': sns.color_palette()[1],
                                                         'alpha': 0.5},
                                          eventline_kwargs={'color': 'black',
                                                            'linestyle': 'dashed',
                                                            'alpha': 0.75},
                                          error_bars='sem', ax=ax)
        # plt.title('Reward delivery')
        plt.ylabel('Firing rate (spikes/s)')
        plt.xlabel('Time (s) from reward delivery')
        plt.tight_layout()
        plt.savefig(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe), 'Reward',
                         '%s_n%d' % (region, cluster)))
        plt.close(fig)

# Reward ommission PSTH
sig_units = bb.task.responsive_units(spikes[probe].times, spikes[probe].clusters,
                                     trials.feedback_times[trials.feedbackType == -1],
                                     alpha=ALPHA)[0]
print('%d out of %d units are significantly responsive to reward omission' % (
                            len(sig_units), len(np.unique(spikes[probe].clusters))))

if not isdir(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe), 'Omission')):
    mkdir(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe), 'Omission'))
    for n, cluster in enumerate(sig_units):
        # Get brain region of neuron
        region = channels[probe].acronym[clusters[probe].channels[
                             clusters[probe].metrics.cluster_id == cluster][0]]
        region = region.replace('/', '-')

        fig, ax = plt.subplots(1, 1)
        sns.set(style="ticks", context="paper", font_scale=2)
        bb.plot.peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                          trials.feedback_times[
                                              trials.feedbackType == -1],
                                          cluster, t_before=1, t_after=2,
                                          pethline_kwargs={'color': sns.color_palette()[2],
                                                           'lw': 2},
                                          errbar_kwargs={'color': sns.color_palette()[2],
                                                         'alpha': 0.5},
                                          eventline_kwargs={'color': 'black',
                                                            'linestyle': 'dashed',
                                                            'alpha': 0.75},
                                          error_bars='sem', ax=ax)
        # plt.title('Reward omission')
        plt.ylabel('Firing rate (spikes/s)')
        plt.xlabel('Time (s) from reward omission')
        plt.tight_layout()
        plt.savefig(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe), 'Omission',
                         '%s_n%d' % (region, cluster)))
        plt.close(fig)
