# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:48:15 2019

List of basis functions for DLC

@author: Guido, Miles, Zoe, Kelly
"""

import numpy as np
from os.path import join
from glob import glob
import alf.io
from ibllib.io import raw_data_loaders


def load_dlc(folder_path, camera='left'):
    """
    Load in DLC traces and timestamps from FPGA and align them

    Parameters
    ----------
    folder_path: string of the path to the top-level folder of recording
    camera: which camera to use ('left', 'right', 'bottom')
    """

    # Load in DLC data
    dlc_dict = alf.io.load_object(join(folder_path, 'alf'), '_ibl_%sCamera' % camera)
    dlc_dict['camera'] = camera
    dlc_dict['units'] = 'px'

    # Hard-coded hack because extraction of timestamps was wrong
    if camera == 'left':
        camera = 'body'

    # Load in FPGA timestamps
    timestamps = np.load(join(folder_path, 'raw_video_data',
                              '_iblrig_%sCamera.times.npy' % camera))

    # Align FPGA and DLC timestamps
    if len(timestamps) > len(dlc_dict[list(dlc_dict.keys())[0]]):
        timestamps = timestamps[0:len(dlc_dict[list(dlc_dict.keys())[0]])]
    elif len(timestamps) < len(dlc_dict[list(dlc_dict.keys())[0]]):
        for key in list(dlc_dict.keys()):
            dlc_dict[key] = dlc_dict[key][0:len(timestamps)]
    dlc_dict['timestamps'] = timestamps
    dlc_dict['sampling_rate'] = 1 / np.mean(np.diff(timestamps))

    return dlc_dict


def load_event_times(folder_path):
    """
    Load in DLC traces and timestamps from FPGA and align them

    Parameters
    ----------
    folder_path: path to top-level folder of recording
    camera:      which camera to use

    """
    stim_on_times = np.load(glob(join(folder_path, 'alf', '_ibl_trials.stimOn_times*.npy'))[0])
    feedback_times = np.load(glob(join(folder_path, 'alf', '_ibl_trials.feedback_times*.npy'))[0])
    return stim_on_times, feedback_times


def load_events(folder_path):
    feedback_type = np.load(glob(join(folder_path, 'alf', '_ibl_trials.feedbackType*.npy'))[0])
    choice = np.load(glob(join(folder_path, 'alf', '_ibl_trials.choice*.npy'))[0])
    return choice, feedback_type


def px_to_mm(dlc_dict, width_mm=66, height_mm=54):
    """
    Transform pixel values to millimeter

    Parameters
    ----------
    width_mm:  the width of the video feed in mm
    height_mm: the height of the video feed in mm
    """

    # Set pixel dimensions for different cameras
    if dlc_dict['camera'] == 'left':
        px_dim = [1280, 1024]
    elif dlc_dict['camera'] == 'right' or dlc_dict['camera'] == 'body':
        px_dim = [640, 512]

    # Transform pixels into mm
    for key in list(dlc_dict.keys()):
        if key[-1] == 'x':
            dlc_dict[key] = dlc_dict[key] * (width_mm / px_dim[0])
        if key[-1] == 'y':
            dlc_dict[key] = dlc_dict[key] * (height_mm / px_dim[1])
    dlc_dict['units'] = 'mm'

    return dlc_dict


def load_dlc_training(folder_path):
    """
    Load in DLC output for a behavioral training session. Extract timestamps from raw BPod data
    """

    # Load in dlc dictionary
    dlc_dict = alf.io.load_object(glob(join(folder_path, 'alf', '_ibl_leftCamera.dlc.*.npy'))[0])

    # Load in BPod data
    bpod_data = raw_data_loaders.load_data(folder_path)

    # Check first couple of trials and determine in which trial camera timestamps begin
    for trial in range(len(bpod_data)):
        if 'Port1In' in bpod_data[trial]['behavior_data']['Events timestamps']:
            timestamps = np.array(
                    bpod_data[trial]['behavior_data']['Events timestamps']['Port1In'])
            first_trial = trial
            break
    if 'Port1In' not in bpod_data[trial]['behavior_data']['Events timestamps']:
        raise Exception('No camera timestamps found in BPod data')

    # Calculate frame rate
    frame_diff = np.mean(np.diff(timestamps))

    # Loop over trials and get camera timestamps
    for i in range(first_trial+1, len(bpod_data)):
        this_trial = np.array(bpod_data[i]['behavior_data']['Events timestamps']['Port1In'])

        # Interpolate the timestamps in the 'dead time' in between trials during which
        # Bpod does not log camera timestamps
        interp = np.arange(timestamps[-1] + frame_diff,
                           this_trial[0] - (frame_diff / 2),
                           frame_diff)
        timestamps = np.concatenate((timestamps, interp, this_trial))

    # Cut off video frames that don't have corresponding bpod timestamps at end of session
    for key in list(dlc_dict.keys()):
        dlc_dict[key] = dlc_dict[key][0:np.size(timestamps)]

    # Add to dictionary
    dlc_dict['timestamps'] = timestamps
    dlc_dict['camera'] = 'left'
    dlc_dict['units'] = 'px'
    return dlc_dict































