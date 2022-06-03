#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:22:08 2022

@author: baptistelafoux
"""

import cv2
import glob
import matplotlib.pyplot as plt
import natsort
import os
import progressbar

from pathlib import Path
import numpy as np 

#%%


def init_writers(path):

    print(f'\n#### Initializing {path}\n')

    frames = natsort.natsorted(glob.glob(f'{path}/*.tiff'))

    vid_basename = f'{os.path.dirname(os.path.dirname(path))}/movies/{os.path.basename(path[:-1])}'

    frame_ini = cv2.imread(frames[0])
    w, h = frame_ini.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(f'{vid_basename}.mp4', fourcc, FPS, (h, w), isColor=False)
    #writer =  cv2.VideoWriter('test.mp4', fourcc, FPS, (w, h))
    return (frames, writer)



def generate_movie(frames, writer):

    print('\n')
    #plt.pause(1)

    try:
        for file in progressbar.progressbar(frames):

            frame = cv2.imread(file, cv2.IMREAD_ANYDEPTH)
            #frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            frame = (frame >> 8).astype('uint8')
            
            writer.write(frame)            

    finally:
        writer.release()

if __name__ == "__main__":
    plt.close('all')
     
    global FPS
    FPS = 50

    paths = glob.glob('/Volumes/baptiste/data_canal_nage/swimming/20220324_1hemi_light_and_dark/2022*/')

    print(f'\nProcessing {len(paths)} movies')

    for path in paths:

        frames, writer = init_writers(path)
        generate_movie(frames, writer)
