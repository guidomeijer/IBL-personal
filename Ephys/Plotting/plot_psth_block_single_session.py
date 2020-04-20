# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

@author: guido
"""

from os import mkdir
from os.path import join, isdir, expanduser
import matplotlib.pyplot as plt
import shutil
import brainbox as bb
import seaborn as sns
import numpy as np
from brainbox.io.one import load_spike_sorting, load_channel_locations
from oneibl.one import ONE
one = ONE()

# Settings
ALPHA = 0.05
SUBJECT = 'DY_011'
DATE = '2020-01-30'
PROBE = '00'
OVERWRITE = True

# Set path to save plots
FIG_PATH = join(expanduser('~'), 'Figures', 'Ephys', 'PSTH')

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

# Make directories
if (isdir(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe)))
        and (OVERWRITE is True)):
    shutil.rmtree(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe)))
if not isdir(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe))):
    mkdir(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe)))
if not isdir(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe), 'Blocks')):
    mkdir(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe), 'Blocks'))

# Calculate whether neuron discriminates blocks
trial_times = trials.goCue_times[
                    ((trials.probabilityLeft > 0.55)
                     | (trials.probabilityLeft < 0.45))]
trial_blocks = (trials.probabilityLeft[
        (((trials.probabilityLeft > 0.55)
          | (trials.probabilityLeft < 0.45)))] > 0.55).astype(int)

diff_units = bb.task.differentiate_units(spikes[probe].times, spikes[probe].clusters,
                                         trial_times, trial_blocks,
                                         pre_time=1, post_time=0, alpha=0.05)[0]

print('%d out of %d neurons differentiate between blocks' % (
                                len(diff_units), len(np.unique(spikes[probe].clusters))))

for n, cluster in enumerate(diff_units):
    # Get brain region of neuron
    if channels[probe].acronym.shape[0] == 0:
        region = 'NoHist'
    else:
        region = channels[probe].acronym[clusters[probe].channels[
                    clusters[probe].metrics.cluster_id == cluster][0]]
        region = region.replace('/', '-')

    fig, ax = plt.subplots(1, 1)
    sns.set(style="ticks", context="paper", font_scale=2)
    bb.plot.peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                      trials.stimOn_times[trials.probabilityLeft > 0.55],
                                      cluster, t_before=1, t_after=2,
                                      error_bars='sem', ax=ax)
    y_lim_1 = ax.get_ylim()
    bb.plot.peri_event_time_histogram(spikes[probe].times, spikes[probe].clusters,
                                      trials.stimOn_times[trials.probabilityLeft < 0.45],
                                      cluster, t_before=1, t_after=2, error_bars='sem',
                                      pethline_kwargs={'color': 'red', 'lw': 2},
                                      errbar_kwargs={'color': 'red', 'alpha': 0.5}, ax=ax)
    y_lim_2 = ax.get_ylim()
    if y_lim_1[1] > y_lim_2[1]:
        ax.set(ylim=y_lim_1)
    plt.legend(['Left block', 'Right block'])
    plt.title('Stimulus onset')
    plt.savefig(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe), 'Blocks',
                     '%s_n%d' % (region, cluster)))
    plt.close(fig)
