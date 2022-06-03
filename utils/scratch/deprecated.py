#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:35:20 2022

@author: baptistelafoux
"""

def test_midline():
    plt.close('all')
    
    path = '/Volumes/baptiste/data_canal_nage/swimming/20220324_1hemi_light_and_dark/20220324_f1_16V_l1_2'
    path = '/Volumes/baptiste/data_canal_nage/swimming/20220324_1hemi_light_and_dark/20220324_f1_10V_l0_1'
    path = '/Volumes/baptiste/data_canal_nage/swimming/20220324_1hemi_light_and_dark/20220324_f1_18V_l0_1'
    
    files = natsorted(glob.glob(f'{path}/*.tiff'))
    
    n_fish = 1 
    
    #ROI = get_ROI(plt.imread(files[0]))
    ROI = (57, 428, 1119, 458)
    
    for file in progressbar([files[100]]): 
        
        frame_GRAY = plt.imread(file)[int(ROI[1]):int(ROI[1] + ROI[3]), 
                                      int(ROI[0]):int(ROI[0] + ROI[2])]
        
        frame_BW = gray2BW(frame_GRAY)
        
        frame_BW_cleaned, cnts = compute_cnts(frame_BW, n_fish)
        
        centerlines = centerline_voronoi(cnts)
        
        fig, ax = plt.subplots(5, 1, figsize=(8, 20))
        
        ax[0].imshow(frame_GRAY, cmap='Greys_r'); ax[0].set_title('Original')
        ax[1].imshow(frame_BW, cmap='Greys_r'); ax[1].set_title('BW')
        ax[2].imshow(frame_BW_cleaned, cmap='Greys_r'); ax[2].set_title('BW cleaned')
        
        for focal in range(n_fish) :
            ax[3].plot(cnts[focal][..., 0], cnts[focal][..., 1], 'r-o', label='Sav-Gol filt. contour')
            ax[3].plot(centerlines[focal][..., 0], centerlines[focal][..., 1], 'k-', label='Centerline Voronoi')
            
            ax[2].plot(centerlines[focal][..., 0], centerlines[focal][..., 1], 'g-', lw=3)

        ax[3].legend(); ax[3].axis('scaled'); ax[3].invert_yaxis()
    
    fig.tight_layout() 
    
def curvature_from_spline(f, x):
    
    xp, yp   = interpolate.splev(np.linspace(x, f, der=1)
    xpp, ypp = interpolate.splev(x, f, der=2)

    curv = ( (xp**2 + yp**2) ** (3/2) / (xp*ypp - yp*xpp) )
    
    return curv

def resample_contour(f, x, n_points):
    
    from scipy.signal import resample 
    
    R_curv = 1 / curvature_from_spline(f, x)
    new_x = np.cumsum(R_curv) / np.sum(R_curv)
    
    new_x = resample(nex_x, num=n_points)
    
    return new_x
    
def set_n_points(cnt, closed_path, n_points=100):
    
    f, x = interpolate.splprep([cnt[..., 0], cnt[..., 1]],
                               k=2, #spline degree
                               s=1, #smoothing factor (unclear)
                               per=closed_path)
    
    #new_x = np.linspace(0, 1, n_points) #a dense & homogeneous sampling of points for the contour
    
    nex_x = resample_contour(f, x, n_points)
    
    x, y = interpolate.splev(new_x, f)
    cnt_spline = np.c_[x, y]
    
    
    return (cnt_spline, f, u)