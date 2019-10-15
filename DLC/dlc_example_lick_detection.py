#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:55:47 2019

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
from dlc_analysis_functions import pupil_features, blink_times, lick_times


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


mouse = 'DY_006'
lab = 'danlab'
eid = '147e66b0-fc99-4090-b515-2ed78ee9adb0'
FIG_PATH = '/home/guido/Figures/DLC/ExampleApplications/'

# Load in data
one = ONE()
dtypes = ['camera.dlc', 'camera.times']
d = one.load(eid, dataset_types=dtypes, download_only=True, dclass_output=True)
ses_path = Path(d.local_path[0]).parent
dlc_dict = alf.io.load_object(ses_path, '_ibl_leftCamera', short_keys=True)
video_path = Path(ses_path.parent, 'raw_video_data', '_iblrig_leftCamera.raw.mp4')
fs = np.mean(np.diff(dlc_dict['times']))

# blink
blinks, pupil_likelihood = blink_times(dlc_dict)
blink_frames = [i for i, x in enumerate(dlc_dict['times'] == blinks[0]) if x]
blink_image = get_video_frame(str(video_path), blink_frames[0])

# lick
licks, _, dist_lick = lick_times(dlc_dict)
lick_frames = [i for i, x in enumerate(dlc_dict['times'] == licks[100]) if x]
lick_image = get_video_frame(str(video_path), lick_frames[0])
x, y = get_xy_frame(dlc_dict, lick_frames[0])


dlc_dict = alf.io.load_object(ses_path, '_ibl_leftCamera', short_keys=True)

# Plot blink
fig, ax = plt.subplots(1, 1)
ax.imshow(blink_image, 'gray')
ax.set(ylim=[np.int(dlc_dict['pupil_top_r_y'][blink_frames[0]]-50),
             np.int(dlc_dict['pupil_top_r_y'][blink_frames[0]]+150)],
       xlim=[np.int(dlc_dict['pupil_top_r_x'][blink_frames[0]]-150),
             np.int(dlc_dict['pupil_top_r_x'][blink_frames[0]]+150)])
plt.gca().invert_yaxis()
ax.axis('off')
plt.savefig(join(FIG_PATH, '%s_%s_blink.png' % (mouse, lab)), dpi=300)
plt.savefig(join(FIG_PATH, '%s_%s_blink.pdf' % (mouse, lab)), dpi=300)

# Plot blink trace
fig, ax = plt.subplots(1, 1)
ax.plot(dlc_dict['times']-blinks[0], pupil_likelihood, lw=4)
ax.plot(0, 0, 'xr', markersize=15, markeredgewidth=4)
ax.set(ylabel='Likelihood pupil points', xlabel='Time (s)',
       xlim=[-1, 1], ylim=[-0.2, 1.2])
sns.set(style="ticks", context='talk', font_scale=1.2)
plt.tight_layout(pad=2)
sns.despine(trim=True)
plt.savefig(join(FIG_PATH, '%s_%s_blinking.png' % (mouse, lab)))

# Plot tongue
fig, ax = plt.subplots(1, 1)
ax.imshow(lick_image, 'gray')
ax.plot(x, y, 'og', markersize=10)
ax.set(ylim=[np.int(dlc_dict['tongue_end_l_y'][lick_frames[0]]-100),
             np.int(dlc_dict['tongue_end_l_y'][lick_frames[0]]+100)],
       xlim=[np.int(dlc_dict['tongue_end_l_x'][lick_frames[0]]-100),
             np.int(dlc_dict['tongue_end_l_x'][lick_frames[0]]+100)])
plt.gca().invert_yaxis()
ax.axis('off')
plt.savefig(join(FIG_PATH, '%s_%s_lick.png' % (mouse, lab)), dpi=300)
plt.savefig(join(FIG_PATH, '%s_%s_lick.pdf' % (mouse, lab)), dpi=300)

# Plot trace
fig, ax = plt.subplots(1, 1)
ax.plot(dlc_dict['times'], dist_lick)
ax.plot(licks, np.zeros(np.size(licks)), 'xr')
ax.set(ylabel='Distance tongue\nto spout (mm)', xlabel='Time (s)',
       xlim=[25, 65], ylim=[-0.2, 4])
sns.set(style="ticks", context='talk', font_scale=1.2)
plt.tight_layout(pad=2)
sns.despine(trim=True)
plt.savefig(join(FIG_PATH, '%s_%s_licking.png' % (mouse, lab)), dpi=300)
plt.savefig(join(FIG_PATH, '%s_%s_licking.pdf' % (mouse, lab)), dpi=300)


# Plot trace
fig, ax = plt.subplots(1, 1)
ax.plot(dlc_dict['times'], dist_lick, lw=5)
ax.plot(licks, np.zeros(np.size(licks)), 'xr', markersize=15, markeredgewidth=5)
ax.set(ylabel='Distance tongue\nto spout (mm)', xlabel='Time (s)',
       xlim=[29.5, 31.25], ylim=[-0.2, 2.5])
ax.axis('off')
plt.tight_layout(pad=2)
plt.savefig(join(FIG_PATH, '%s_%s_licking_zoom.png' % (mouse, lab)), dpi=300)
plt.savefig(join(FIG_PATH, '%s_%s_licking_zoom.pdf' % (mouse, lab)), dpi=300)




