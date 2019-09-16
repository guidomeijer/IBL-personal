#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:08:11 2019

@author: guido
"""

from ibllib.io import spikeglx
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp

#path ='/home/guido/IBLserver/Subjects/ZM_1150/2019-05-07/001/raw_ephys_data/probe_right/ephysData_g0_t0.imec.ap.bin'
path = '/home/guido/IBLserver/Ephys/20190710_sync_test/ephys/20190709_sync_right_g0_t0.imec.ap.bin'

time = [450.25, 451.5]
sf = 30000 # Sampling frequency
samples = [round(time[0]*sf), round(time[1]*sf)]

sr = spikeglx.Reader(path)
w, s = sr.read_samples(first_sample=samples[0], last_sample=samples[1])

w = sp.zscore(w, axis=0)

w_ref = w
for i in range(len(w[0])):
    ind = np.arange(i-4,i+4)
    ind = ind[ind != i]
    ind[ind>=len(w[0])] = len(w[0])-1
    w_ref[:,i] = (w[:,i] - (np.mean(w[:,ind], axis=1)*2))/3
    #w_ref[:,i] = (w[:,i] - (np.mean(w[:,ind], axis=1)))

vx = np.linspace(samples[0]/sf, samples[1]/sf, samples[1]-samples[0])
ch_show = [101,106]

fig = plt.figure()
plt.plot(vx, s[:,2], 'k', label='Camera Left')
plt.plot(vx, s[:,4]+2, 'k', label='Camera Right')
plt.plot(vx, s[:,3]+4, 'k', label='Camera Body')
plt.plot(vx, s[:,7]+6, 'k', label='BPod')
plt.plot(vx, s[:,12]+8, 'k', label='Frame2TTL')
plt.plot(vx, s[:,15]+10, 'k', label='Audio')
plt.plot(vx, s[:,13]+12, 'k', label='Rotary encoder 1')
plt.plot(vx, s[:,14]+14, 'k', label='Rotary encoder 2')
ch_inc = np.arange(17,17+(15*3),3)
count = 0
for i in ch_show:
    plt.plot(vx, w_ref[:,i]+ch_inc[count],'k')
    count = count + 1
#plt.legend()
plt.xlabel('Time (s)')
fig.set_size_inches((12, 8), forward=False)
plt.savefig('/home/guido/Figures/Ephys/traces_TTL.png',dpi=300)
plt.savefig('/home/guido/Figures/Ephys/traces_TTL.pdf',dpi=300)
plt.show()

#plt.figure()
#plt.plot(vx, w_ref[:,101])
#plt.plot(vx, np.mean(w[:,90:140], axis=1))
#plt.show()