#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:29:26 2019

@author: guido
"""

from oneibl.one import ONE
from pathlib import Path
from glob import glob
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2
import sys
sys.path.insert(0, '/home/guido/Projects/ibllib/ibllib/dlc_analysis')
from dlc_basis_functions import load_dlc_training, load_event_times, load_events, px_to_mm
from dlc_plotting_functions import peri_plot
from dlc_analysis_functions import lick_times


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
    plt.imshow(frame)
    return frame_image


# Query sessions with available DLC data using ONE
one = ONE()
dtypes = ['_ibl_leftCamera.dlc', '_iblrig_taskData.raw', 'trials.feedback_times',
          'trials.feedbackType', 'trials.stimOn_times', 'trials.choice', '_iblrig_leftCamera.raw']
eids = one.search(dataset_types=dtypes)
eid = eids[0]

# Load data
d = one.load(eid, dataset_types=dtypes, download_only=True, dclass_output=True)
folder_path = str(Path(d.local_path[0]).parent.parent)
dlc_dict = load_dlc_training(folder_path)
dlc_dict = px_to_mm(dlc_dict)

# Detect licks
licks = lick_times(dlc_dict)

frame = get_video_frame(glob(join(
        folder_path, 'raw_video_data', '_iblrig_leftCamera.raw*.mp4'))[0],
                        np.argmin(np.abs(dlc_dict['timestamps'] - licks[0])))

plt.imshow(frame)

