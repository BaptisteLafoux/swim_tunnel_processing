#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 13:52:28 2022

@author: baptistelafoux
"""

#%% Import modules and global variables 

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from scipy.spatial import Voronoi
from scipy import interpolate

from utils.geom import ordered_path, fourier2space

import numpy as np 

def cnts2centerlines(cnts, fourier_space, n_points): 
        
    centerlines = np.zeros((len(cnts), n_points, 2))
    
    for i, cnt in enumerate(cnts) :
        
        if fourier_space: cnt = fourier2space(cnt)
            
        vor = Voronoi(cnt)
        vor_points = [Point(point) for point in vor.vertices]
    
        polygon = Polygon(cnt)
        is_inside = [polygon.contains(point) for point in vor_points]
    
        vor_inside = vor.vertices[is_inside, :]
        
        ordered_centerline = ordered_path(vor_inside, closed_path=False)
        centerlines[i] = resample_cl(ordered_centerline, n_points)

    return centerlines


def resample_cl(cline, n_points): 
    
    spline_function, u = interpolate.splprep([cline[..., 0], cline[..., 1]],
                               k=3, #spline degree
                               s=1, #smoothing factor (unclear)
                               )
    
    new_u = np.linspace(0, 1, n_points)
    new_x, new_y = interpolate.splev(new_u, spline_function)
    
    resampled_cline = np.c_[new_x, new_y]
        
    return resampled_cline
