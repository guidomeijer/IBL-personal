# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

@author: guido
"""

from os import listdir
from os.path import join, isdir
from os import mkdir
import alf.io as ioalf
import matplotlib.pyplot as plt
import brainbox as bb
from ephys_functions import paths, download_data
import numpy as np

# Settings
SUBJECT = 'ZM_1897'
SESSION = '2019-12-06'
PROBE = '00'
DOWNLOAD = False
DATA_PATH, FIG_PATH = paths()

# Download data if required
if DOWNLOAD is True:
    download_data(SUBJECT, SESSION)

# Get paths
ses_nr = listdir(join(DATA_PATH, SUBJECT, SESSION))[0]
session_path = join(DATA_PATH, SUBJECT, SESSION, ses_nr)
alf_path = join(DATA_PATH, SUBJECT, SESSION, ses_nr, 'alf')
probe_path = join(DATA_PATH, SUBJECT, SESSION, ses_nr, 'alf', 'probe%s' % PROBE)

# Load in data
spikes = ioalf.load_object(probe_path, 'spikes')
clusters = ioalf.load_object(probe_path, 'clusters')
trials = ioalf.load_object(alf_path, '_ibl_trials')

# Only use single units
spikes.times = spikes.times[np.isin(
    spikes.clusters, clusters.metrics.cluster_id[clusters.metrics.ks2_label == 'good'])]
spikes.clusters = spikes.clusters[np.isin(
    spikes.clusters, clusters.metrics.cluster_id[clusters.metrics.ks2_label == 'good'])]

# Make folders
if not isdir(join(FIG_PATH, 'PSTH', '%s' % SUBJECT)):
    mkdir(join(FIG_PATH, 'PSTH', '%s' % SUBJECT))
if not isdir(join(FIG_PATH, 'PSTH', '%s' % SUBJECT, '%s' % SESSION)):
    mkdir(join(FIG_PATH, 'PSTH', '%s' % SUBJECT, '%s' % SESSION))

for n, cluster in enumerate(spikes.clusters):
    fig = plt.figure()
    bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters,
                                      trials.stimOn_times[trials.contrastLeft == 0],
                                      cluster, t_before=1, t_after=2, error_bars='sem',
                                      include_raster=True)
    plt.title('Stimulus onset')
    plt.savefig(join(FIG_PATH, 'PSTH', '%s' % SUBJECT, '%s' % SESSION,
                     'p%s_n%s' % (PROBE, cluster)))
    plt.close()
