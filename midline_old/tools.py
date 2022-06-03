#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 10:45:51 2021

@author: baptistelafoux
"""

import numpy as np 
import matplotlib.pyplot as plt
import glob 
import sys 
import cv2
from skimage.morphology import skeletonize
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

sys.path.append('/Users/baptistelafoux/Google Drive/Pro/2020_PMMH_thèse_fish/5_PostProcessing')
import findBackground
import astropy.units as u
from fil_finder import FilFinder2D

from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy import interpolate
from PIL import Image, ImageDraw

from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings("ignore")

from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon



kernel = np.ones((3,3),np.uint8)

def contour_largest_area(gray_im, threshold):
    
    gray_im = gaussian_filter(gray_im, sigma=1)
    
    binary = np.uint8(gray_im < threshold)
    
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    #opening = binary
    cnts, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    
    out_binary = np.zeros(gray_im.shape, np.uint8)
    out_binary = cv2.drawContours(out_binary, [cnt], -1, 255, cv2.FILLED)
    
    return np.squeeze(cnt), out_binary

def smooth_spline(raw_points_sorted, k, smoothing, closed_shape):

    f, u = interpolate.splprep([raw_points_sorted[:,0], raw_points_sorted[:,1]],
                               k=k, 
                               s=smoothing, 
                               per=closed_shape)
    
    x, y = interpolate.splev(u, f)
    points = np.c_[x, y]
    
    return (points, f, u)

def smooth_spline_savgol(raw_points_sorted, win_len, deg, closed):
    
    if closed: mode = 'wrap'
    else : mode = 'interp'
    
    x = savgol_filter(raw_points_sorted[:,0], win_len, deg, mode=mode)
    y = savgol_filter(raw_points_sorted[:,1], win_len, deg, mode=mode)
    
    xp = savgol_filter(raw_points_sorted[:,0], win_len, deg, deriv=1, mode=mode)
    yp = savgol_filter(raw_points_sorted[:,1], win_len, deg, deriv=1, mode=mode)
    
    xpp = savgol_filter(raw_points_sorted[:,0], win_len, deg, deriv=2, mode=mode)
    ypp = savgol_filter(raw_points_sorted[:,1], win_len, deg, deriv=2, mode=mode)
    
    points = np.c_[x, y]
    points_p = np.c_[xp, yp]
    points_pp = np.c_[xpp, ypp]
    
    return points, points_p, points_pp
    
def curvature(spline, npoints):
    
    s = np.linspace(0, 1, npoints)
    
    xp, yp =   interpolate.splev(s, spline, der=1)
    xpp, ypp = interpolate.splev(s, spline, der=2)
    
    curv = ( (xp**2 + yp**2) ** (3/2) / (xp*ypp - yp*xpp) )
    
    return (curv, s)

def curvature_savgol(p, p_p, p_pp):
    
    s = np.sqrt((p[:,0] - p[0,0])**2 + (p[:,1] - p[0,1])**2)
    s /= np.max(s)
    curv =  (p_p[:,0]**2 + p_p[:,1]**2) ** (3/2) / (p_p[:,1]*p_pp[:,0] - p_p[:,0]*p_pp[:,1]) 
    
    return (curv, s)

def find_skeleton(smooth_contour, initial_binary_shape, precision=1):
    
    img = Image.new('L', (initial_binary_shape[1]*precision, initial_binary_shape[0]*precision), 0)
    ImageDraw.Draw(img).polygon((smooth_contour*precision).flatten().tolist(), outline=1, fill=1)
    mask = np.array(img)

    skel_binary = skeletonize(mask.astype(bool), method='lee')
    
    x, y = np.where(skel_binary)
    points = np.c_[x,y]
    
    skel_binary[skel_binary!=0] = 1
    skel_binary = np.uint8(skel_binary)

    kernel_skel = np.uint8([[1,  1, 1],
                            [1, 10, 1],
                            [1,  1, 1]])
    
    nb_neighbours = cv2.filter2D(skel_binary, -1, kernel_skel)
    endpoint = np.array(np.where(nb_neighbours==11))[:,-1]
    

    endpoint_idx = np.squeeze(np.argwhere((points==endpoint).all(axis=1)))
    
    return (points, endpoint_idx)

def find_skeleton_voronoi(smooth_contour):
    vor = Voronoi(smooth_contour)
    vor_points = [Point(point) for point in vor.vertices]
    
    polygon = Polygon(smooth_contour)
    is_inside = [polygon.contains(point) for point in vor_points]
    
    vor_inside = vor.vertices[is_inside, :]
    
    mean_interdistances = np.mean(cdist(vor_inside, vor_inside), axis=0)
    endpoint_idx = np.argmax(mean_interdistances)
    
    return vor_inside, endpoint_idx

def ordered_path(points, closed_path, start=None):
    
    if closed_path: 
        start = np.random.randint(0, len(points)) 
        
    pass_by = points
    path = np.empty_like(points)
    path[0, :] = points[start, :]
    
    i = 1
    pass_by = np.delete(pass_by, start, 0)
    
    while len(pass_by[:, 0]) > 0:
        nearest = np.argmin(cdist([path[i-1, :]], pass_by))
        path[i, :] = pass_by[nearest, :]
        pass_by = np.delete(pass_by, nearest, 0)
        i+=1
        
    return path

def curvature_splines(points, error=0.0001):
    
    x, y = points.T 
    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)

    fx = UnivariateSpline(t, x, k=4, w=1 / np.sqrt(std))
    fy = UnivariateSpline(t, y, k=4, w=1 / np.sqrt(std))

    xˈ = fx.derivative(1)(t)
    xˈˈ = fx.derivative(2)(t)
    yˈ = fy.derivative(1)(t)
    yˈˈ = fy.derivative(2)(t)
    curvature = (xˈ* yˈˈ - yˈ* xˈˈ) / np.power(xˈ** 2 + yˈ** 2, 3 / 2)
    return curvature
    
def get_midline_info(im_gray, threshold):
    
    cnt, binary  = contour_largest_area(im_gray,  threshold)
    
    #cnt_smooth, _, _ = smooth_spline(cnt, k=3, smoothing=1000, closed_shape=True)
    cnt_smooth, _, _ = smooth_spline_savgol(cnt, win_len=101, deg=2, closed=True)
    
    skel, endpoint_idx = find_skeleton_voronoi(cnt_smooth)

    skel_sorted = ordered_path(skel, start=endpoint_idx, closed_path=False)
    
    #skel_sorted = gaussian_filter1d(skel_sorted, sigma=5, axis=0)
    
    #idx = np.arange(0, skel_sorted.shape[0], 2)
    #skel_sorted = skel_sorted[idx,:]
    
    #midline, spline, _ = smooth_spline(skel_sorted, k=5, smoothing=100, closed_shape=False)
    midline, p, pp = smooth_spline_savgol(skel_sorted, win_len=81, deg=5, closed=False)
    
    r_curv, s = curvature_savgol(midline, p, pp)
    #r_curv, s = curvature(spline, len(midline))
    
    return cnt_smooth, midline, 1/r_curv, s, skel_sorted
    
    
