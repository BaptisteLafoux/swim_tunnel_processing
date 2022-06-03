#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:34:20 2022

@author: baptistelafoux
"""

import numpy as np 
from scipy.spatial.distance import cdist

#%% Contour extraction and processing 

def ordered_path(points, closed_path):
    '''
    Re-orders a set of 2D points to create a closed loop contour 

    Parameters
    ----------
    points : ndarray (N, 2)
        DESCRIPTION.
    closed_path : bool
        If you want the path to be a closed contour or just a line .

    Returns
    -------
    path : ndarray (N, 2)
        Array with same points as in 'points', but ordered clockwise in space.

    '''
    
    start = np.argmin(points[..., 0])
        
    pass_by = points
    path = np.zeros_like(points)
    path[0, :] = points[start, :]
    
    i = 1
    pass_by = np.delete(pass_by, start, 0)
    
    while len(pass_by[:, 0]) > 0:
        nearest = np.argmin(cdist([path[i-1, :]], pass_by))
        path[i, :] = pass_by[nearest, :]
        pass_by = np.delete(pass_by, nearest, 0)
        i+=1
        
    return path

def space2fourier(cnt, n_coeffs=32):
    
    n_ini = cnt.shape[-2]
    
    # Fourier transform of the contour in the complex plane 
    fourier_trans_cnt = np.fft.fft(cnt[..., 0] + 1j * cnt[..., 1], axis=-1)
    # Shift the F-T so that the positive and negative frequencies are in the middle of the array 
    fourier_coeffs = np.fft.fftshift(fourier_trans_cnt)
    # Get the number of coefficients we want in the middle of the array (- and + freq) and normalize by the number of points (useful to reconstruct after)
    fourier_coeffs = fourier_coeffs[(n_ini - n_coeffs) // 2 : (n_ini + n_coeffs) // 2] / n_ini
    
    return fourier_coeffs


def fourier2space(fourier_coeffs, n_points=200):
    
    n_coeffs = fourier_coeffs.shape[-1]
    
    # Pad the coefficient with 0 to get a FT array the same size as the recontructed contour we want
    fourier_trans_cnt = pad_along_axis(fourier_coeffs * n_points, pad_size=n_points-n_coeffs, axis=-1)
    # Shift back and inverse F-T 
    cnt_complex = np.fft.ifft(np.fft.ifftshift(fourier_trans_cnt), axis=-1)
    # Back in (x,y)-plane
    cnt = np.concatenate([cnt_complex.real[..., None], cnt_complex.imag[..., None]], axis=-1) 
    
    return cnt

def pad_along_axis(array: np.ndarray, pad_size: int, axis: int = -1):

    npad = [(0, 0)] * array.ndim
    npad[axis] = (pad_size // 2, pad_size // 2)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)
