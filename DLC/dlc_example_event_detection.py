#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 10:54:54 2019

@author: guido
"""

from oneibl.one import ONE
import alf.io
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from os.path import join
import seaborn as sns
import cv2
import sys
sys.path.insert(0, '/home/guido/Projects/ibllib/ibllib/dlc_analysis')
from dlc_basis_functions import px_to_mm
from dlc_plotting_functions import peri_plot
from dlc_analysis_functions import pupil_features, lick_times, sniff_times


def butter_filter(data, cutoff, fs, ftype='lowpass', order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=ftype, analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def get_video_frame(video_path, frame_number):
    """
    Obtain numpy array corresponding to a particular video frame in video_path
    :param video_path: local path to mp4 file
    :param frame_number: video frame to be returned
    :return: numpy array corresponding to frame of interest.  Dimensions are (1024, 1280, 3)
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_number)  # 0-based index of the frame to be decoded/captured next.
    ret, frame_image = cap.read()
    cap.release()
    return frame_image


def get_xy_frame(dlc_dict, frame_number):
    """
    Obtain numpy array with all x and y coordinates of a certain frame
    :param dlc_dict: dictionary with DLC results
    :param frame_number: video frame to be returned
    :return: x and y numpy arrays
    """
    x = np.empty(0)
    y = np.empty(0)
    for key in list(dlc_dict.keys()):
        if key[-1] == 'x':
            x = np.append(x, dlc_dict[key][frame_number])
        if key[-1] == 'y':
            y = np.append(y, dlc_dict[key][frame_number])
    return x, y


FRAME_NR = 5000
FIG_PATH = '/home/guido/Figures/DLC/ExampleApplications/'

# Query sessions with available DLC data using ONE
one = ONE()
dtypes = ['camera.dlc', 'camera.times']
eids, ses_info = one.search(dataset_types=dtypes, details=True)

for i, eid in enumerate(eids):
    if np.mod(i+1, 5) == 0:
        print('Running session %d of %d' % (i+1, len(eids)))

    # Load in data
    d = one.load(eid, dataset_types=dtypes, download_only=True, dclass_output=True)
    ses_path = Path(d.local_path[0]).parent
    dlc_dict = alf.io.load_object(ses_path, '_ibl_leftCamera', short_keys=True)
    video_path = Path(ses_path.parent, 'raw_video_data', '_iblrig_leftCamera.raw.mp4')
    fs = np.mean(np.diff(dlc_dict['times']))

    # Get DLC and video data
    if ((np.isnan(dlc_dict['pupil_top_r_x'][0]))
        or (np.size(dlc_dict['pupil_top_r_x']) < 100)
            or (np.size(dlc_dict['times']) != np.size(dlc_dict['pupil_top_r_x']))):
        print('x')
        continue
    x, y = get_xy_frame(dlc_dict, FRAME_NR)
    frame_image = get_video_frame(str(video_path), FRAME_NR)
    if frame_image is None:
        continue

    # Get pupil
    pupil_x, pupil_y, diameter = pupil_features(dlc_dict)
    diameter_filt = butter_filter(diameter, 2, 1/fs, 'lowpass', 1)

    # Get sniffing
    dis = np.sqrt(((dlc_dict['nostril_top_x'] - dlc_dict['nostril_bottom_x'])**2)
                  + ((dlc_dict['nostril_top_y'] - dlc_dict['nostril_bottom_y'])**2))
    dis_filt = butter_filter(dis, [7, 12], 1/fs, 'bandpass', 1)
    sniff_bouts = sniff_times(dlc_dict, threshold=0.5)

    # Get licking
    licks, _, dist_lick = lick_times(dlc_dict)

    # Plot tongue
    fig, ax = plt.subplots(1, 1)
    ax.imshow(frame_image, 'gray')
    ax.plot(x, y, 'og', markersize=10)
    ax.set(ylim=[np.int(dlc_dict['tongue_end_l_y'][FRAME_NR]-50),
                 np.int(dlc_dict['tongue_end_l_y'][FRAME_NR]+100)],
           xlim=[np.int(dlc_dict['tongue_end_l_x'][FRAME_NR]-100),
                 np.int(dlc_dict['tongue_end_l_x'][FRAME_NR]+100)])
    plt.gca().invert_yaxis()
    ax.axis('off')
    plt.savefig(join(FIG_PATH, '%s_%s_tongue.pdf' % (ses_info[i]['subject'], ses_info[i]['lab'])),
                dpi=300)
    plt.close(fig=fig)

    fig, ax = plt.subplots(1, 1)
    ax.plot(dlc_dict['times'], dist_lick)
    ax.plot(licks, np.zeros(np.size(licks)), 'xr')
    ax.set(ylabel='Distance tongue\nto spout (mm)', xlabel='Time (s)',
           xlim=[6, 12], ylim=[-0.2, 3])
    sns.set(style="ticks", context='talk', font_scale=1.2)
    plt.tight_layout(pad=2)
    sns.despine(trim=True)
    plt.savefig(join(FIG_PATH, '%s_%s_licking.pdf' % (ses_info[i]['subject'],
                                                      ses_info[i]['lab'])), dpi=300)
    plt.close(fig=fig)

    # Plot pupil
    fig, ax = plt.subplots(1, 1)
    ax.imshow(frame_image, 'gray')
    ax.plot(x, y, 'og', markersize=6)
    ax.plot(pupil_x[FRAME_NR], pupil_y[FRAME_NR], 'or', markersize=10)
    circle = plt.Circle((pupil_x[FRAME_NR], pupil_y[FRAME_NR]), diameter_filt[FRAME_NR]/2,
                        color='r', fill=False, lw=2)
    ax.add_artist(circle)
    ax.set(ylim=[np.int(dlc_dict['pupil_top_r_y'][FRAME_NR]-35),
                 np.int(dlc_dict['pupil_top_r_y'][FRAME_NR]+50)],
           xlim=[np.int(dlc_dict['pupil_top_r_x'][FRAME_NR]-50),
                 np.int(dlc_dict['pupil_top_r_x'][FRAME_NR]+50)])
    plt.gca().invert_yaxis()
    ax.axis('off')
    plt.savefig(join(FIG_PATH, '%s_%s_pupil.png' % (ses_info[i]['subject'], ses_info[i]['lab'])))
    plt.close(fig=fig)

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(0.01, 2000*fs, fs), diameter_filt[FRAME_NR:FRAME_NR+2000])
    ax.set(ylabel='Pupil diameter (mm)', xlabel='Time (s)', xlim=[0, 20])
    sns.set(style="ticks", context='talk', font_scale=1.2)
    sns.despine(trim=True)
    plt.tight_layout(pad=2)
    plt.savefig(join(FIG_PATH, '%s_%s_pupil_diameter.pdf' % (ses_info[i]['subject'],
                                                             ses_info[i]['lab'])), dpi=300)
    plt.close(fig=fig)

    # Plot nostril
    fig, ax = plt.subplots(1, 1)
    ax.imshow(frame_image, 'gray')
    ax.plot(x, y, 'og', markersize=10)
    ax.set(ylim=[np.int(dlc_dict['nostril_top_y'][FRAME_NR]-35),
                 np.int(dlc_dict['nostril_top_y'][FRAME_NR]+50)],
           xlim=[np.int(dlc_dict['nostril_top_x'][FRAME_NR]-50),
                 np.int(dlc_dict['nostril_top_x'][FRAME_NR]+50)])
    plt.gca().invert_yaxis()
    ax.axis('off')
    plt.savefig(join(FIG_PATH, '%s_%s_nostril.png' % (ses_info[i]['subject'],
                                                      ses_info[i]['lab'])))
    plt.close(fig=fig)

    fig, ax = plt.subplots(1, 1)
    ax.plot(dlc_dict['times'], dis_filt)
    ax.plot(sniff_bouts, np.ones(np.size(sniff_bouts))*-0.8, 'xr',
            markersize=8, markeredgewidth=2)
    ax.set(ylabel='Filtered nostril\ndistance (mm)', xlabel='Time (s)',
           xlim=[240, 280], ylim=[-1, 1])
    sns.set(style="ticks", context='talk', font_scale=1.2)
    plt.tight_layout(pad=2)
    sns.despine(trim=True)
    plt.savefig(join(FIG_PATH, '%s_%s_nostril_distance.png' % (ses_info[i]['subject'],
                                                               ses_info[i]['lab'])), dpi=300)
    plt.savefig(join(FIG_PATH, '%s_%s_nostril_distance.pdf' % (ses_info[i]['subject'],
                                                               ses_info[i]['lab'])), dpi=300)
    plt.close(fig=fig)




