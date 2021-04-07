#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:29:25 2021

@author: guido
"""

from os.path import join
import cv2
import pandas as pd
import numpy as np
from my_functions import figure_style
import matplotlib.pyplot as plt

DATA_DIR = '/home/guido/Data/Camera_test/'
DURATION = 60

# Read in timestamps
ts_body = np.load(join(DATA_DIR, '_ibl_bodyCamera.times.10a29258-5701-4662-a7b3-368323649f79.npy'))
ts_left = np.load(join(DATA_DIR, '_ibl_leftCamera.times.20e57ee9-a9a3-4d25-9d21-98d98b6b2f7d.npy'))

# %% Body camera
cam_body = cv2.VideoCapture(join(DATA_DIR, '_iblrig_bodyCamera.raw.e67ff5de-3d56-4310-b8e7-3e3b62d80a07.mp4'))
mov_body = []
bool_first = True
while(cam_body.isOpened()):
    ret, frame = cam_body.read()
    if bool_first:
        previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bool_first = False
    else:
        try:
            this_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            break
        mov_body.append(np.sum(this_frame - previous_frame))
        previous_frame = this_frame
    if ts_body[len(mov_body)] > DURATION:
        break
cam_body.release()

# %% Left camera
cam_left = cv2.VideoCapture(join(DATA_DIR, '_iblrig_leftCamera.raw.ef4eb8b8-03b1-4c27-bdc7-475e31302aa8.mp4'))
mov_left = []
bool_first = True
while(cam_left.isOpened()):
    ret, frame = cam_left.read()
    if bool_first:
        previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bool_first = False
    else:
        try:
            this_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            break
        mov_left.append(np.sum(this_frame - previous_frame))
        previous_frame = this_frame
    if ts_left[len(mov_left)] > DURATION:
        break
cam_left.release()

# %% Plot

# smoothing
mov_body = np.convolve(mov_body, np.ones((5,))/5, mode='same')
mov_left = np.convolve(mov_left, np.ones((5,))/5, mode='same')

# normalize
mov_body = (mov_body - np.min(mov_body)) / np.max(mov_body)
mov_left = (mov_left - np.min(mov_left)) / np.max(mov_left)

figure_style(font_scale=4)
f, ax1 = plt.subplots(1, 1, figsize=(30, 10))
ax1.plot(ts_body[:len(mov_body)], mov_body, label='Body camera', lw=3)
ax1.plot(ts_left[:len(mov_left)], mov_left, label='Left camera', lw=3)
ax1.set(ylabel='Normalized movement energy', xlabel='Time (s)')
ax1.legend(frameon=False)
