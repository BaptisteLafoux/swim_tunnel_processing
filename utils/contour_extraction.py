#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:26:00 2022

@author: baptistelafoux
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 13:52:28 2022

@author: baptistelafoux
"""

#%% Import modules and global variables 

import numpy as np 
import cv2 

from utils.geom import ordered_path, space2fourier

#%% Image processing functions 


def gray2BW(im_gray, method='gaussian', bitdepth=16): 
    '''
    Binarization of gray images in 8 or 16 bits from Basler Camera 

    Parameters
    ----------
    im_gray : 8 or 16 bit image (ndarray w * h) 
        DESCRIPTION.
    method : str, optional
        etiher 'gaussian' for adaptive gaussian binarization or 'otsu' for Otsu-based method. The default is 'gaussian'.

    Returns
    -------
    frame_BW : binary image with size w * h 
        (fish is white, bg is black).

    '''
    
    if method=='otsu': 
        _, frame_BW = cv2.threshold(cv2.GaussianBlur(im_gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if method=='gaussian': 
        if bitdepth == 16 : im_gray = (im_gray / 256).astype(np.uint8)
        frame_BW = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 40)
    
    return frame_BW

def BW2cnts(im_BW, n_blob, n_coeffs):
    '''
    Clean a binary image and keeps only the n_blobs largest areas in terms of surface

    Parameters
    ----------
    im_BW : np.uint8 array
        Binary image.
    n_blob : int
        Number of fish in the image.

    Returns
    -------
    list of n_blob ndarray
        Each ndarray is the contour of a fish in the Fourier space (n_blobs, n_coeffs) .

    '''
    # dilate the binary image to avoid 1 or 2 pxl channels that prevent the ordering of the path points (super important)
    im_BW = cv2.dilate(cv2.bitwise_not(im_BW), np.ones((3, 3), np.uint8), iterations=1)
    
    # Find the contours of the binary image 
    cnts, _ = cv2.findContours(im_BW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    # Sort the contours with respect to their area
    cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    # Recreate the BW image with only n_blob conours
    #im_BW_clean = cv2.drawContours(np.zeros_like(im_BW, dtype=np.uint8), cnts_sorted[:n_blob], -1, 255, cv2.FILLED)
    
    for blob in range(n_blob): 
        cnt = ordered_path(np.squeeze(cnts_sorted[blob]), closed_path=True)
        #change the 10 here if you want to change the typical number of "segments" in your shape (weird manipulation to make it odd)
        #cnt = cnt_filter(cnt, window_length=(len(cnt) // 10) // 2 * 2 + 1, polyorder=2, mode='wrap')
        
        cnt = space2fourier(cnt, n_coeffs=n_coeffs)
        
        cnts_sorted[blob] = cnt
        
    return cnts_sorted[:n_blob]


def frame2cnts(frame, n_fish, n_coeffs): 
    
    frame_BW = gray2BW(frame)
    cnts = BW2cnts(frame_BW, n_fish, n_coeffs)
    
    return cnts


    
    
    
    




