#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:11:25 2022

@author: hbahk
"""

import time
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from astropy.visualization import ZScaleInterval
from astropy.io import fits
from astropy.nddata import CCDData
import astroalign as aa
from astropy.modeling import models, fitting
from ccdproc import combine

#%%
# setting object name
OBJNAME = 'A2255'

# setting locations
HOME = Path.home()
WD = HOME/'class'/'ao22'/'ta'/'tutorial'
DATA_DIR = WD/'data'/OBJNAME

# fetching path list of the preprocessed science frames
sci_list_R  = list(DATA_DIR.glob('p' + OBJNAME + '*R.fits'))
sci_list_I  = list(DATA_DIR.glob('p' + OBJNAME + '*I.fits'))

#%% Initial alignment check
sci_list = sci_list_I

# for displaying in zscale
interval = ZScaleInterval()

fig, axs = plt.subplots(1,len(sci_list), figsize=(3*len(sci_list),3))

for i in range(len(sci_list)):
    img = fits.getdata(sci_list[i], ext=0)
    trim = img[760:860, 1063:1163]
    vmin, vmax = interval.get_limits(trim)
    ax = axs[i]
    ax.imshow(trim, origin='lower', vmin=vmin, vmax=vmax)
    if i == 0:
        # fit the data using astropy.modeling to obtain its center
        n, m = trim.shape
        x, y = np.mgrid[:n, :m]
        const_init = models.Const2D(amplitude=np.median(trim))
        g_init = models.Gaussian2D(amplitude=trim.max(), x_mean=45., y_mean=45.,
                                    x_stddev=10., y_stddev=10.)
        f_init = const_init + g_init
        fit_f = fitting.LevMarLSQFitter()
        f = fit_f(f_init, x, y, trim)
        xcenter, ycenter = f.x_mean_1.value, f.y_mean_1.value
        # if you want to find the center in simple way, find the peak coord 
        #xc, yc = np.unravel_index(np.argmax(trim), trim.shape)
    ax.axvline(ycenter, c='r', ls='--')
    ax.axhline(xcenter, c='r', ls='--')
    
#%% Solving registration with astroalign
# reference image: frame 0
id_ref = 0
dat_ref, hdr_ref = fits.getdata(sci_list[id_ref], header=True, ext=0)
ccd_ref = CCDData(dat_ref, unit='adu')

# Aligning other images with respect to the reference image
start_time = time.time()
aligned_list = []
for i in range(len(sci_list)):
    dat = fits.getdata(sci_list[i], ext=0)
    msk = fits.getdata(sci_list[i], ext=1)
    ccd = CCDData(dat.byteswap().newbyteorder(), unit='adu', mask=msk)
    if (i == id_ref):
        ccd_aligned = ccd
    else:
        dat_aligned, footprint = aa.register(ccd, ccd_ref,
                                             max_control_points=50,
                                             detection_sigma=5, min_area=16,
                                             propagate_mask=True)
        ccd_aligned = CCDData(dat_aligned, unit='adu')
    aligned_list.append(ccd_aligned)
    # fits.writeto(imglist[i].split(".fit")[0]+"_align_auto.fits", dat_aligned, overwrite=True)
end_time = time.time()
print(f"--- {end_time-start_time:.4f} sec were taken for aligning {len(sci_list):d} images ---") 

#%%
fig, axs = plt.subplots(1,len(sci_list), figsize=(3*len(sci_list),3))
for i in range(len(aligned_list)):
    img = aligned_list[i].data
    trim = img[760:860, 1063:1163]
    vmin, vmax = interval.get_limits(trim)
    ax = axs[i]
    ax.imshow(trim, origin='lower', vmin=vmin, vmax=vmax)
    if i == 0:
        # fit the data using astropy.modeling to obtain its center
        n, m = trim.shape
        x, y = np.mgrid[:n, :m]
        const_init = models.Const2D(amplitude=np.median(trim))
        g_init = models.Gaussian2D(amplitude=trim.max(), x_mean=45., y_mean=45.,
                                    x_stddev=10., y_stddev=10.)
        f_init = const_init + g_init
        fit_f = fitting.LevMarLSQFitter()
        f = fit_f(f_init, x, y, trim)
        xcenter, ycenter = f.x_mean_1.value, f.y_mean_1.value
        # if you want to find the center in simple way, find the peak coord 
        #xc, yc = np.unravel_index(np.argmax(trim), trim.shape)
    ax.axvline(ycenter, c='r', ls='--')
    ax.axhline(xcenter, c='r', ls='--')

#%%

combined = combine(aligned_list, sigma_clip=True,
                   sigma_clip_high_thresh=3, sigma_clip_low_thresh=3)

plt.figure(figsize=(9,6))
vmin, vmax = interval.get_limits(combined.data)
plt.imshow(combined, origin='lower', vmax=vmax, vmin=vmin)

band = hdr_ref['FILTER']
hdr_ref['CFRAMES'] = len(sci_list)
hdr_ref['history'] = f'Combined image of {len(sci_list)} frames'
now = time.strftime("%Y-%m-%d %H:%M:%S (%Z = GMT%z)")
hdr_ref['history'] = f'Combined at {now}'
scihdu = fits.PrimaryHDU(data=combined.data.astype('float32'), header=hdr_ref)
mskhdu = fits.ImageHDU(data=combined.mask.astype('int'), name='MASK')
hdul = fits.HDUList([scihdu, mskhdu])
hdul.writeto(DATA_DIR/('c'+OBJNAME+'_'+band+'.fits'), overwrite=True)

#%%
ccd = CCDData.read(DATA_DIR/'reduced'/(OBJNAME+'_I_master.fits'),
                   unit='adu')
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)#, projection=ccd.wcs)
vmin, vmax = interval.get_limits(ccd.data)
ax.imshow(combined, origin='lower', vmax=vmax, vmin=vmin)
ax.set_xlabel('RA')
ax.set_ylabel('DEC')
ax.grid('on')

#%% making a file for fits io tutorial

from astropy.nddata import Cutout2D

ccd_cut = Cutout2D(ccd, position=(2000, 1000), size=(250,250), wcs=ccd.wcs)
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection=ccd_cut.wcs)
vmin, vmax = interval.get_limits(ccd_cut.data)
ax.imshow(ccd_cut.data, origin='lower', vmax=vmax, vmin=vmin)
ax.set_xlabel('RA')
ax.set_ylabel('DEC')
ax.grid('on')

hdr_cut = hdr_ref + ccd_cut.wcs.to_header()
hdu = fits.PrimaryHDU(data=ccd_cut.data.astype('float32'), header=hdr_cut)
hdu.writeto(WD/'data'/'example1.fits', overwrite=True)
