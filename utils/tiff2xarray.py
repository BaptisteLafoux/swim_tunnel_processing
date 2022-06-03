#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:51:20 2022

@author: baptistelafoux
"""
import matplotlib.pyplot as plt 
import numpy as np 
import xarray as xr 
import os 

import glob 

from natsort import natsorted

from utils.utilities import extract_info_from_filename, get_ROI, copy_files_in_cache, delete_cache, get_all_ROIs
from utils.contour_extraction import frame2cnts
from utils.centerline_extraction import cnts2centerlines

from tqdm import tqdm

def process_image_batch(path, n_image=None, ROI=False, n_coeffs_fourier=32, n_coeff_cl=32, local_processing=True, fps=50): 
    
    print('#####################################')
    print(f'Working on : {path}')
    light, voltage, date, n_fish, date, path_movie = extract_info_from_filename(path)
    
    files = natsorted(glob.glob(f'{path}/*.tiff')); original_n_image = len(files); path_original = path
    
    if local_processing: 
        
        path = copy_files_in_cache(path, files[:n_image]) #copy files locally and generate a new path
        files = natsorted(glob.glob(f'{path}/*.tiff'))
        
    n_image = len(files)
    
    print(f'\nProcessing {path} ({n_fish} fish, working on {n_image} images over {original_n_image})\n')
    
    CNT = np.full([n_image, n_fish, n_coeffs_fourier], np.nan, dtype=complex)
    CL  = np.full([n_image, n_fish, n_coeff_cl, 2], np.nan)
    
    if not ROI: ROI = get_ROI(plt.imread(files[0]))
    
    for i, file in enumerate(tqdm(files[:n_image])):
        
        frame = plt.imread(file)[int(ROI[1]):int(ROI[1] + ROI[3]), int(ROI[0]):int(ROI[0] + ROI[2])]
        cnts_fs = frame2cnts(frame, n_fish, n_coeffs_fourier)
        
        clines = cnts2centerlines(cnts_fs, fourier_space=True, n_points=n_coeff_cl)
            
        CNT[i] = cnts_fs; CL[i] = clines
    
    metadata = dict(fps = fps,
                    n_fish = n_fish,
                    date = date,
                    path = path_original,
                    path_movie = path_movie,
                    voltage = voltage,
                    light = light,
                    n_frames = len(CNT), 
                    ROI = np.array([ROI[1], ROI[1] + ROI[3], ROI[0], ROI[0] + ROI[2]]),
                    n_fourier_coeff_contour = n_coeffs_fourier,
                    n_pt_centerline = n_coeff_cl)
    
    if local_processing: delete_cache()
    
    print('#####################################')
    
    data = dict(c = CNT,
                l = CL)
    
    return data, metadata

def create_xarray(data, metadata):
    

    data_dict = dict(
        c=(["time", "fish", "fourier_coeff", "space"], np.concatenate([data['c'].real[..., None], data['c'].imag[..., None]], axis=-1)),
        l=(["time", "fish", "pt_centerline", "space"], data['l']),
        
    )

    coords = {
        "time": (["time"], np.arange(metadata['n_frames']) / metadata['fps']),
        "fish": (["fish"], np.arange(metadata['n_fish'])),
        "space": (["space"], ["x", "y"]),
        "pt_centerline": (["pt_centerline"], np.arange(metadata['n_pt_centerline'])),
        "fourier_coeff": (["fourier_coeff"], np.arange(metadata['n_fourier_coeff_contour']))
    }
    
    ds = xr.Dataset(data_vars=data_dict, coords=coords, attrs=metadata)
    
    return ds

def save_ds(ds):
    
    print('\nSaving the dataset')
    ds.to_netcdf(f'clean_data/{os.path.basename(ds.path)}.nc')
    
    print('\nSaving done')
    

if __name__ == "__main__": 
    plt.close('all')
    
    paths = glob.glob('/Volumes/baptiste/data_canal_nage/swimming/20220324_1hemi_light_and_dark/20220324*')
    
    #path = '/Volumes/baptiste/data_canal_nage/swimming/20220324_1hemi_light_and_dark/20220324_F1_V16_L1_2'
    
    #ROIs = get_all_ROIs(paths)
    
    for i, path in enumerate([paths[-1]]) : 
        
        data, metadata = process_image_batch(path, n_image=None, ROI=[41, 438, 1133, 446], local_processing=True)
        
        ds = create_xarray(data, metadata)
    
        save_ds(ds)

    
