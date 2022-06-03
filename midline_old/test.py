#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:38:14 2021

@author: baptistelafoux
"""
plt.close('all')
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon



plt.figure()
plt.plot(midline[:,0], midline[:,1], 'k-o')

midgauss = gaussian_filter1d(midline, sigma=5, axis=0)
plt.plot(midgauss[:,0], midgauss[:,1], 'r-o')

plt.plot(rawmidline[:,0], rawmidline[:,1], 'b-o')
plt.plot(cnt_smooth[:,1], cnt_smooth[:,0], 'g-o')

vor = Voronoi(cnt_smooth)




vor_points = [Point(point) for point in vor.vertices]

polygon = Polygon(cnt_smooth)
is_inside = [polygon.contains(point) for point in vor_points]

vor_inside = vor.vertices[is_inside, :]

plt.plot(vor_inside[:,1], vor_inside[:,0], 'o')

d = np.mean(cdist(vor_inside, vor_inside), axis=0)
mini = np.argmax(d)

plt.plot(vor_inside[mini,1], vor_inside[mini,0], 'ko')


#%%
plt.close('all')
plt.figure()
plt.plot(rawmidline[:,0], rawmidline[:,1], 'k-o')


plt.plot(midline[:,0], midline[:,1], 'r-o')

medmid_x = median_filter(rawmidline[:,0], size=30)
medmid_y = median_filter(rawmidline[:,1], size=30)
medmid = np.c_[medmid_x, medmid_y]
gaussmid = gaussian_filter1d(rawmidline, sigma=2, axis=0, mode='mirror')
plt.plot(gaussmid[:,0], gaussmid[:,1], 'b-o')


dx, dy = np.gradient(gaussmid, axis=0).T
ddx, ddy = np.gradient(np.c_[dx, dy], axis=0).T
#dx = savgol_filter(dx, 101, 3)
#dy = savgol_filter(dy, 101, 3)
plt.figure()
plt.plot(dx, 'r.')
plt.plot(dy, 'g.')
plt.plot(ddx, 'b.')
plt.plot(ddy, 'k.')

n = 5
dx = np.convolve(dx, np.hanning(n)/np.sum(np.hanning(n)), mode='same')
dy = np.convolve(dy, np.hanning(n)/np.sum(np.hanning(n)), mode='same')


# ddx = savgol_filter(ddx, 101, 3)
# ddy = savgol_filter(ddy, 101, 3)
ddx = np.convolve(ddx, np.hanning(n)/np.sum(np.hanning(n)), mode='same')
ddy = np.convolve(ddy, np.hanning(n)/np.sum(np.hanning(n)), mode='same')


#plt.figure()
plt.plot(dx, 'r--')
plt.plot(dy, 'g--')
plt.plot(ddx, 'b--')
plt.plot(ddy, 'k--')

k = ( ddx*dy - ddy*dx ) / (dx**2 + dy**2)**(1.5)
k = gaussian_filter1d(k, sigma=30)
plt.plot(k, '-o')