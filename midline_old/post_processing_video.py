#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:50:54 2021

@author: baptistelafoux
"""
import pandas as pd 

import numpy as np 
import matplotlib.pyplot as plt
import glob 
import sys 
import cv2
from skimage.morphology import skeletonize
from matplotlib.colors import TwoSlopeNorm

sys.path.append('/Users/baptistelafoux/Google Drive/Pro/2020_PMMH_thèse_fish/5_PostProcessing/postProCode/python/tools')
import findBackground

from scipy.ndimage import gaussian_filter, median_filter
from scipy import interpolate

from tools import *
from matplotlib.animation import FFMpegWriter

import progressbar


plt.close('all')
directory = '/Volumes/baptiste/data_canal_nage/swimming/030521-channel-swimm-hemi-miror/030521-1hemi-side-10v-95fps-2/'
files = np.sort(glob.glob(directory + '*.tiff'))

fps = 95

bg = np.mean(findBackground.findBackground_imageseries(directory, 5), axis=2, dtype=np.uint8) 

roi = {'xmin': 150, 'xmax': 500,
       'ymin_top': 600, 'ymax_top': 800,
       'ymin_side': 110, 'ymax_side': 420}


data = pd.DataFrame(columns=['contour', 'midline', 'curvature', 's', 'rawmidline'])
    
for i, file in progressbar.progressbar(enumerate(files)): 
    
    frame = plt.imread(file)
    
    frame_nobg = cv2.absdiff(frame, bg)
    
    top  = frame_nobg[roi['ymin_top']:roi['ymax_top'],   roi['xmin']:roi['xmax']]
    side = frame_nobg[roi['ymin_side']:roi['ymax_side'], roi['xmin']:roi['xmax']]

    cnt_smooth, midline, curv, s, rawmidline= get_midline_info(top, threshold=80)
    
    #plt.imshow(top)
    #plt.plot(rawmidline[:,0], rawmidline[:,1], 'r')
    #plt.plot(curv)
    data = data.append({'contour': cnt_smooth,
                        'rawmidline': rawmidline,
                        'midline': midline, 
                        'curvature': curv,
                        's': s},
                       ignore_index=True)
    
    #plt.pause(0.1)
    #plt.cla()
    if i%100==0: print(i)
    
data.to_pickle(directory + 'data.pkl')
#%%
plt.close('all')
fig = plt.figure()

data = pd.read_pickle(directory + 'data.pkl')

n= 20
writer = FFMpegWriter(fps = 9.5) 

list_k = []
with writer.saving(fig, directory + 'video.mp4', 100):  
    for i in progressbar.progressbar(range(len(data))): 
            
        #for j in range(i, i+n):
         #   plt.plot(data.iloc[i+j].rawmidline[:,0], data.iloc[i+j].rawmidline[:,1], 'k-', linewidth=2*(j-i)/n)
        frame = plt.imread(files[i])
    
        frame_nobg = cv2.absdiff(frame, bg)
        
        top  = frame_nobg[roi['ymin_top']:roi['ymax_top'],   roi['xmin']:roi['xmax']]
        side = frame_nobg[roi['ymin_side']:roi['ymax_side'], roi['xmin']:roi['xmax']]
        
        plt.imshow(top, origin='lower', cmap='Greys_r')
        rawmidline = data.rawmidline[i]
        gaussmidline = gaussian_filter1d(rawmidline, sigma=8, axis=0 , mode='nearest')
        gaussmidline_p =  gaussian_filter1d(gaussmidline,   sigma=8, axis=0, order=1, mode='nearest')
        gaussmidline_pp = gaussian_filter1d(gaussmidline_p, sigma=8, axis=0, order=1, mode='nearest')
        
        k = (gaussmidline_pp[:,0]*gaussmidline_p[:,1] - gaussmidline_pp[:,1]*gaussmidline_p[:,0])
        k = median_filter(k, size=10)
        k = gaussian_filter1d(k, sigma=10)
        
        list_k.append(np.mean(k))
        line = plt.scatter(rawmidline[:,0], rawmidline[:,1], s=4, c=k, cmap='RdBu', vmin=-0.001, vmax=0.001)
        kmax = np.argmax(np.abs(k))
        #plt.plot(k, '-o')
        #line = plt.scatter(rawmidline[:,0], rawmidline[:,1], c=k, cmap='RdBu', vmin=-0.003, vmax=0.003)
        #plt.colorbar()
        plt.axis('off')
        #plt.pause(0.1)
        plt.tight_layout()
        #writer.grab_frame()
        
        plt.clf()
        #print(i)

#%%
from scipy.fft import fft, fftfreq, fftshift
from scipy import interpolate

# Number of sample points

list_k_mod = median_filter(list_k, size=3)
list_k_mod = list_k_mod - np.mean(list_k_mod)
t = np.arange(0, len(list_k_mod))/95

f = interpolate.interp1d(t, list_k_mod, kind='cubic')
t_new = np.linspace(t.min(), t.max(), 5000)

list_k_interp = f(t_new)

N = len(list_k_interp)

# sample spacing

T = 1.0 / 95
x = np.linspace(0.0, N*T, N, endpoint=False)


yf = fft(list_k_interp)

xf = fftfreq(N, T)[:N//2]



#fig, ax = plt.subplots(2, 1)

plt.figure(figsize=(6,2), dpi=100)

plt.axhline(0, linewidth=0.5, color='k')
plt.plot(t_new, list_k_interp/200, 'k-')
plt.xlabel('t [s]')
plt.ylabel(r'$\bar{\kappa$ [BL$^{-1}$]')

#ax[1].magnitude_spectrum(list_k_interp, Fs=95, pad_to=10**5, color ='k')
#plt.loglog(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.tight_layout()

plt.savefig('curvature_f(t)_example.pdf')


    #%%
        curv = gaussian_filter(curv, sigma=8)
maxcurv = np.argmax(curv)
ax1.scatter(midline[:,1], midline[:,0], c=np.abs(curv), cmap='Reds', vmin=0, vmax=0.02)
  ax1.plot(midline[maxcurv,1], midline[maxcurv,0], 'ko')
  ax1.invert_yaxis()
  ax1.axis('off')
  
  # ax2.plot(s, curv)
  # ax2.set_ylim([-0.02, 0.02])
  # ax2.axis('off')
  
  midlines = np.append(midlines, midline, axis=2)
  midlines = np.delete(midlines, 0, axis=2)
  
  for i in range(midlines.shape[2]):
  ax3.plot(midlines[:,1,i], midlines[:,0,i], 'k-', linewidth=0.25)
  #ax3.axis('off')
  #ax3.axis([30, 200, 30, 150])
  

  
 
  
  plt.pause(0.01)
  ax1.clear()

  #ax2.clear()
  print(file)
  
      
    
      #%%

        
        norm = TwoSlopeNorm(vmin=-0.015, vcenter=0, vmax=0.015)
ax1 = fig.add_subplot(211)
ax3 = fig.add_subplot(212)

    
