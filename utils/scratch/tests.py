#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 17:17:00 2022

@author: baptistelafoux
"""

#%% 

import matplotlib.animation as animation
from utils.geom import fourier2space

plt.close('all')
n_fish = 1


fig, ax = plt.subplots()

line, = ax.plot([], [], 'k.', lw=3) 
ax.axis([0, 200, 0, 200])
ax.axis('scaled')

cnt = fourier2space(CNT, 500)

def anim(i): 
    
    
    #line.set_data(CL[i][0][..., 0], CL[i][0][..., 1])
    line.set_data(cnt[i, 0, :, 0], cnt[i, 0, :, 1])
    
    return line,

ani = animation.FuncAnimation(fig, anim, frames=len(CL), blit=True, interval=50, repeat=True)

#ani.save('output/movies/simple_fish_contour.mp4', writer=animation.FFMpegWriter(fps=20))

plt.show() 

fig, ax = plt.subplots()

for i in range(len(CL)):
    ax.plot(CL[i, 0, ..., 0] - CL[i, 0, 0, 0], CL[i, 0, ..., 1] - CL[i, 0, 0, 1], 'k-')

ax.axis('scaled')


#%% 
plt.close('all')
n_points=10000

def set_n_points(cnt, closed_path, n_points=n_points):
    
    f, u = interpolate.splprep([cnt[..., 0], cnt[..., 1]],
                                k=5, 
                                s=100, 
                                per=closed_path)
    
    x, y = interpolate.splev(np.linspace(0, 1, n_points), f)
    points = np.c_[x, y]
    
    return (points, f, u)

fig, ax = plt.subplots() 

ax.plot(cnt[..., 0], cnt[..., 1])

points, f, u = set_n_points(cnt, closed_path=True)
ax.plot(points[..., 0], points[..., 1], '.-')

ax.axis('scaled')

xp, yp =   interpolate.splev(np.linspace(0, 1, n_points), f, der=1)
xpp, ypp = interpolate.splev(np.linspace(0, 1, n_points), f, der=2)

curv = ( (xp**2 + yp**2) ** (3/2) / (xp*ypp - yp*xpp) )

fig, ax = plt.subplots()
ax.plot(1 / curv, '.-')
# ax.plot(points[..., 0], 'o')
# ax.plot(points[..., 1], 'o')

# ax.plot(points[..., 0], points[..., 1])
data = (1 / curv)
ax.plot(np.cumsum(data) / np.sum(data))


#%% 
plt.close('all')
complex_cnt = cnt[..., 0] + 1j * cnt[..., 1]

n_pts = 1000 
ft_cnt = np.fft.fft(complex_cnt)

n_coeff = 32
# ft_cnt[n_coeff // 2 : -n_coeff // 2] = 0
ft_shift_cnt = np.fft.fftshift(ft_cnt)[(len(ft_cnt) - n_coeff) // 2 : (len(ft_cnt) + n_coeff) // 2] / len(cnt) 

print(ft_shift_cnt)

ft_shift_cnt = np.pad(ft_shift_cnt * n_pts, ((n_pts-n_coeff)//2, (n_pts-n_coeff)//2), 'constant')

ft_cnt = np.fft.ifftshift(ft_shift_cnt)

cnt_back = np.fft.ifft(ft_cnt) 

fig, ax = plt.subplots() 
ax.plot(ft_cnt.real, 'o')
ax.plot(ft_cnt.imag, 'o')


fig, ax = plt.subplots() 

ax.plot(cnt[..., 0], cnt[..., 1], label='original')

points, f, u = set_n_points(cnt, closed_path=True)
ax.plot(points[..., 0], points[..., 1], label='Bspline')

ax.plot(cnt_back.real, cnt_back.imag, '.', label='Complex FFT')

ax.legend() 

ax.axis('scaled')

#%% 

def curvature_from_spline(f, u):
    
    xp, yp   = interpolate.splev(u, f, der=1)
    xpp, ypp = interpolate.splev(u, f, der=2)

    curv = (xp**2 + yp**2) ** (3/2) / (xp*ypp - yp*xpp) 
    
    return curv

def resample_contour(f, x, n_points):
    
    from scipy.signal import resample 
    
    curv = np.abs(curvature_from_spline(f, x))
    new_x = np.cumsum(curv) / np.sum(curv)
    
    new_x = new_x[::len(new_x) // n_points]
    
    return new_x
    
def set_n_points(cnt, closed_path, n_points=100):
    
    f, u = interpolate.splprep([cnt[..., 0], cnt[..., 1]],
                               k=2, #spline degree
                               s=100, #smoothing factor (unclear)
                               per=closed_path)
    
    #new_x = np.linspace(0, 1, n_points) #a dense & homogeneous sampling of points for the contour
    
    new_u = resample_contour(f, u, n_points)
    
    x, y = interpolate.splev(new_u, f)
    cnt_spline = np.c_[x, y]
    
    
    return cnt_spline, f, new_u

plt.close('all')
n_points=50

fig, ax = plt.subplots() 

ax.plot(cnt[..., 0], cnt[..., 1])

cnt_spline, f, x = set_n_points(cnt, closed_path=True, n_points=n_points)

ax.plot(cnt_spline[..., 0], cnt_spline[..., 1], 'o')

ax.axis('scaled')

fig, ax = plt.subplots() 
ax.plot(x, 'o')

#%%

from utils.geom import fourier2space, space2fourier

f_coef = space2fourier(cnt, 30)

cnt_new = fourier2space(f_coef, 100)
    
fig, ax = plt.subplots() 

ax.plot(cnt[..., 0], cnt[..., 1])
ax.plot(cnt_new[..., 0], cnt_new[..., 1], 'o')

ax.axis('scaled')

#%% 
plt.close("all")
print(cl.shape) 

f, u = interpolate.splprep([cl[..., 0], cl[..., 1]],
                          k=3, #spline degree
                          s=0, #smoothing factor (unclear)
                          )

x, y = interpolate.splev(u, f)

fig, ax = plt.subplots() 
ax.plot(cl[..., 0], cl[...,1])
ax.plot(x, y, 'o')

new_u = np.linspace(0, 1, 32)
x, y = interpolate.splev(new_u, f)
ax.plot(x, y, 'o')

f_new, u_3 = interpolate.splprep([x, y], k=3, s=0) 
x, y = interpolate.splev(u, f_new)

ax.plot(x, y, '-k')

#ax.axis('scaled')


