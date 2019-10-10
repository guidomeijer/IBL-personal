#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:53:18 2019

@author: guido
"""


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