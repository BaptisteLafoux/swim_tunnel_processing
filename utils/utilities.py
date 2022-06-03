#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:39:52 2022

@author: baptistelafoux
"""
import cv2, os, shutil, tqdm, glob

import matplotlib.pyplot as plt 
import numpy as np 

global cache_path; cache_path = os.path.expanduser('~/cache')

def get_ROI(im):
    '''
    Asks user to prompt a Region of Interest in the image 

    Parameters
    ----------
    im : a ndarray
        Grayscale image.

    Returns
    -------
    ROI : tuple
        (a, b, c, d) (a, b) + -- +
                                 |   
                                 |
                                 + (c, d).
    '''
    
    window_name = 'Select the tunnel area'
    
    cv2.imshow(window_name, im)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    ROI = cv2.selectROI(window_name, im);
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    #ROI = np.array([ROI[1], ROI[1] + ROI[3], ROI[0], ROI[0] + ROI[2]])
    print('\n')
    
    return ROI

def copy_files_in_cache(path, files):
    
    if os.path.isdir(cache_path) : shutil.rmtree(cache_path)
    
    
    new_path = f'{cache_path}/{os.path.basename(path)}'
    os.makedirs(new_path)
    
    print(f'\nCopying {path} into {new_path} - processing {len(files)} images\n')
    for file in tqdm.tqdm(files, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
       shutil.copy(file, f'{new_path}/')
    #copy_tree(path, new_path)
    print('\nDone copying')
    
    return new_path

def extract_info_from_filename(path): 
    
    n_fish  = int(path[path.find('_F')+2:path.find('_V')])
    light   = int(path[path.find('_L')+2])
    voltage = float(path[path.find('_V')+2:path.find('_L')])
    
    path_movie = f'{os.path.dirname(path)}/movies/{os.path.basename(path)}.mp4'
    
    datestr = path[path.find('rk/')+3:path.find('_F')]
    #date    = datetime.datetime(int(datestr[0:4]), int(datestr[4:6]), int(datestr[6:])) #Invalid value for attr 'date': datetime.datetime(2022, 3, 24, 0, 0). For serialization to netCDF files, its value must be of one of the following types: str, Number, ndarray, number, list, tuple
    date = (int(datestr[0:4]), int(datestr[4:6]), int(datestr[6:]))
    
    return light, voltage, date, n_fish, date, path_movie

def delete_cache(): 
    
    shutil.rmtree(cache_path)
    
    print('\n')
    print(f'\nCache deleted ({cache_path})')
    
def get_all_ROIs(paths):
    
    ROIs = np.zeros((len(paths), 4))
    
    for i, path in enumerate(paths) : 
        
        files = glob.glob(f'{path}/*.tiff')
        ROIs[i] = get_ROI(plt.imread(files[0]))
    
    return ROIs
    
    
    
    
    
    
    
    
    
    
    