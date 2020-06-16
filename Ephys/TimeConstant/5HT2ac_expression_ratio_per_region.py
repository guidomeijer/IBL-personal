#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:34:08 2020

@author: guido
"""

import nibabel as nb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set paths
data_dir = '/home/guido/Repositories/intrinsic-time/allen_atlas/'
out_dir = '/home/guido/Repositories/intrinsic-time/results/gene_expression/'

# Load in parcellation and mask
structures = pd.read_csv(data_dir+'parcellation.csv', usecols=range(1,5))
parcellation = nb.load(data_dir+'parcellation_200um.nii.gz').get_fdata()
mask_img = nb.load(data_dir+'brain_mask_200um.nii.gz')
mask = mask_img.get_fdata()
aff = mask_img.affine
hdr = mask_img.header

# Load in gene expression data
energy = nb.load(data_dir + 'gene_expression/81671344_Htr2a_energy.nii.gz').get_fdata()


slice_idx = 30
f, ccf_axes = plt.subplots(1, 2, figsize=(10, 5))
ccf_axes[0].imshow(parcellation[slice_idx,:,:], cmap='gray_r', aspect='equal', vmin=parcellation.min(), vmax=parcellation.max())
ccf_axes[0].set_title("Parcellation")
ccf_axes[0].axis('off')
ccf_axes[1].imshow(energy[slice_idx,:,:]*1000, cmap='cubehelix_r', aspect='equal')
ccf_axes[1].set_title("Energy")
ccf_axes[1].axis('off')
plt.show()



"""
# make a mask of where this experiment has data and add to combined mask
data_mask = np.where(energy!=-1)
data_mask_comb[data_mask] += 1

# add data to average map of this entrez
energy_avg[data_mask] += energy[data_mask]
"""