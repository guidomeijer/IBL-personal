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
import numpy as np
import seaborn as sns
import cv2


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


# Query sessions with available DLC data using ONE
one = ONE()
dtypes = ['camera.dlc', 'camera.times']
eids, ses_info = one.search(dataset_types=dtypes, details=True)

# Loop over sessions
FRAME_NR = 5000
for i, eid in enumerate(eids):
    if np.mod(i+1, 5) == 0:
        print('Running session %d of %d' % (i+1, len(eids)))

    # Load in data
    d = one.load(eid, dataset_types=dtypes, download_only=True, dclass_output=True)
    ses_path = Path(d.local_path[0]).parent
    dlc_dict = alf.io.load_object(ses_path, '_ibl_leftCamera', short_keys=True)
    video_path = Path(ses_path.parent, 'raw_video_data', '_iblrig_leftCamera.raw.mp4')

    # Get DLC and video data
    if (np.isnan(dlc_dict['pupil_top_r_x'][0])) or (np.size(dlc_dict['pupil_top_r_x']) < 1000):
        continue
    x, y = get_xy_frame(dlc_dict, FRAME_NR)
    frame_image = get_video_frame(str(video_path), FRAME_NR)
    if frame_image is None:
        continue

    # Overlay in plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(frame_image)
    ax.plot(x, y, 'or', markersize=4)
    ax.axis('off')
    plt.savefig('/home/guido/Figures/DLC/Overlay/%s_%s_frame%d.png' % (ses_info[i]['subject'],
                                                                       ses_info[i]['lab'],
                                                                       FRAME_NR))
    plt.close(fig=fig)
