#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is mostly based on following references.
References
https://github.com/joungh93/TAO_2022A/blob/main/Preprocess.ipynb
https://github.com/ysbach/SNU_AOclass/blob/master/Notebooks/07-Cosmic_Ray_Rejection.ipynb
"""
#TODO: include bad pixel masking
from pathlib import Path
from astropy.io import fits
from matplotlib import pyplot as plt
from astropy.visualization import ZScaleInterval
import numpy as np
from astropy.nddata import CCDData
import astroscrappy
import time
from ccdproc import combine

#%% Inspecting data

# setting object name
OBJNAME = 'A2255'

# setting locations
HOME = Path.home()
WD = HOME / 'class' / 'ao22' / 'ta' / 'tutorial'
DATA_DIR = WD / 'data' / OBJNAME

# fetching path lists for each type of frames
bias_list = list(DATA_DIR.glob('cal*bias.fit'))
dark_list_sci = list(DATA_DIR.glob('cal*dk300.fit'))
dark_list_flat = list(DATA_DIR.glob('cal*dk2.fit'))
flat_list_R = list(DATA_DIR.glob('skyflat*R.fit'))
flat_list_I = list(DATA_DIR.glob('skyflat*I.fit'))
sci_list_R  = list(DATA_DIR.glob(OBJNAME + '*R.fit'))
sci_list_I  = list(DATA_DIR.glob(OBJNAME + '*I.fit'))

# counting files
print(f"Bias: {len(bias_list):d} frames")
print(f"Dark: {len(dark_list_sci):d} frames")
print(f"Dark: {len(dark_list_flat):d} frames")
print(f"Flat (sky; R-band): {len(flat_list_R):d} frames")
print(f"Flat (sky; I-band): {len(flat_list_I):d} frames")
print(f"Science (object; R-band): {len(sci_list_R):d} frames")
print(f"Science (object; I-band): {len(sci_list_I):d} frames")

# test reading of science frame
fpath = sci_list_R[0]
data, hdr = fits.getdata(fpath, header=True)
# data - 2D image array of pixel values
# hdr  - image header information

gain = hdr['EGAIN'] # e-/ADU electronic gain

# key information of the image
for keys in ['DATE-OBS', 'EXPTIME', 'FILTER', 'INSTRUME']:
    print(keys+" = "+str(hdr[keys]))
    
# simple visualization using matplotlib pyplot
bias0 = fits.getdata(bias_list[0])
dark0 = fits.getdata(dark_list_sci[0])
flat0 = fits.getdata(flat_list_R[0])
sci0  = fits.getdata(sci_list_R[0])
label = ["Bias", "Dark", "Sky Flat", "Science Object (Raw)"]

# - for displaying in zscale
interval = ZScaleInterval()

# - plotting
fig, axs = plt.subplots(2, 2, figsize=(9,6))
for i, img0 in enumerate([bias0, dark0, flat0, sci0]):
    vmin, vmax = interval.get_limits(img0)
    ax = axs[i // 2][i % 2]
    ax.imshow(img0, cmap='gray_r', origin='lower', vmin=vmin, vmax=vmax)
    ax.tick_params(axis='both', length=0.0, labelleft=False, labelbottom=False)
    ax.text(0.05, 0.95, label[i], fontsize=15.0, fontweight='bold',
            transform=ax.transAxes, ha='left', va='top')
plt.tight_layout()

#%% Derive Readout Noise

# bring bias1 data
bias1, bias_hdr = fits.getdata(bias_list[0], header=True)            
bias1 = np.array(bias1).astype('float64')

# bring bias2 data
bias2 = fits.getdata(bias_list[0])            
bias2 = np.array(bias2).astype('float64')

# derive the differential image
dbias = bias2 - bias1

# bring gain 
gain = bias_hdr['EGAIN'] # e-/ADU electronic gain

#Calculate RN
RN = np.std(dbias)*gain / np.sqrt(2)
print('Readout Noise is {0:.2f}'.format(RN))

#Do it for all bias data
name = []
RN = []
for i in range(len(bias_list)-1):
    bias1 = fits.getdata(bias_list[i]).astype('float64')
    bias2 = fits.getdata(bias_list[i+1]).astype('float64')
    dbias = bias2 - bias1

    print(i,'st',np.std(dbias)*gain / np.sqrt(2))
    RN.append(np.std(dbias)*gain / np.sqrt(2))
print(np.mean(RN))    
rdnoise = np.mean(RN)

#%% Combining frames

# creating a master bias
# - printing observation date & exposure time of all the bias frames
for i in np.arange(len(bias_list)):
    bias_hdr = fits.getheader(bias_list[i])
    print(f"\nBias frame {i+1:d}")
    for keys in ['DATE-OBS', 'EXPTIME']:
        print("  "+keys+" = "+str(bias_hdr[keys]))
        
# an example of the median combining bias frames
# - make empty array to successively attach each bias frame
bias_array = np.empty((len(bias_list), bias0.shape[0], bias0.shape[1]))

# - stacking the empty array with bias frames
for i in range(len(bias_list)):
    bias_data = fits.getdata(bias_list[i])
    bias_array[i, :, :] = bias_data
# bias_array[0, :, :]    # data from bias_list[0]
# bias_array[1, :, :]    # data from bias_list[1]

# - median combine
bias_med = np.median(bias_array, axis=0)
bias_stack = []

# combining bias frames with sigma clipping
# - stacking the empty array with bias frames
for i in range(len(bias_list)):
    bias_data, bias_hdr = fits.getdata(bias_list[i], header=True)
    bias = CCDData(data=bias_data, header=bias_hdr, unit='adu')
    bias_stack.append(bias)

# - combining
mbias = combine(bias_stack, sigma_clip=True,
                sigma_clip_high_thresh=3, sigma_clip_low_thresh=3)

# - save the master bias as fits file
bias_hdr['NFRAMES'] = len(bias_list)    # recording # of frames combined
fits.writeto(DATA_DIR/"MBias.fits", mbias.data, bias_hdr, overwrite=True)

# - visualization of the combined master bias image
fig, ax = plt.subplots(1, 1, figsize=(5,3))
vmin, vmax = interval.get_limits(mbias)
ax.imshow(mbias, cmap='gray_r', origin='lower', vmin=vmin, vmax=vmax)
ax.tick_params(axis='both', length=0.0, labelleft=False, labelbottom=False)
ax.text(0.50, 0.96, "Master Bias", fontsize=12.0, fontweight='bold',
        transform=ax.transAxes, ha='center', va='top')
ax.text(0.50, 0.88, "(combined with sigma_clipping)", fontsize=11.0,
        transform=ax.transAxes, ha='center', va='top')
ax.text(0.04, 0.12, f"Mean bias level: {np.mean(mbias):.1f}", fontsize=10.0,
        transform=ax.transAxes, ha='left', va='bottom')
ax.text(0.04, 0.04, f"Bias fluctuation: {np.std(mbias):.2f}", fontsize=10.0,
        transform=ax.transAxes, ha='left', va='bottom')
plt.tight_layout()

print(f'bias_med fluctuation : {np.std(bias_med):.2f}')
print(f'bias_sc fluctuation : {np.std(mbias):.2f}')


#%% creating a master dark
# - Note that the dark frames might have different exposure times.
def make_mdark(dark_list):
    # - checking the basic info
    # - Please check the consistancy in observation dates and exposure times. 
    for i in np.arange(len(dark_list)):
        dark_hdr = fits.getheader(dark_list[i])
        print(f"\nDark frame {i+1:d}")
        for keys in ['DATE-OBS', 'EXPTIME']:
            print("  "+keys+" = "+str(dark_hdr[keys]))
    
    # - stacking dark frames
    dark_stack = []
    for i in range(len(dark_list)):
        dark_data, dark_hdr = fits.getdata(dark_list[i], header=True)
        dark_bn = (dark_data - mbias.data)# / dark_hdr['EXPTIME']
        dark = CCDData(data=dark_bn, header=dark_hdr, unit='adu')    
        dark_stack.append(dark)
        
    # - combine with sigma clipping
    mdark = combine(dark_stack, sigma_clip=True,
                    sigma_clip_high_thresh=3, sigma_clip_low_thresh=3)
    
    # - correcting the negative values (physically impossible)
    mdark.data[mdark.data < 0.] = 0.
    
    # - save the master dark as fits file
    dark_hdr['NFRAMES'] = len(dark_list)    # recording # of dark frames combined
    fits.writeto(DATA_DIR/f"MDark{dark_hdr['EXPTIME']:.0f}.fits",
                 mdark, dark_hdr, overwrite=True)
    
    # - visualization of the master dark
    fig, ax = plt.subplots(1, 1, figsize=(5,3))
    vmin, vmax = interval.get_limits(mdark)
    ax.imshow(mdark, cmap='gray_r', origin='lower', vmin=-1, vmax=1)
    ax.tick_params(axis='both', length=0.0, labelleft=False, labelbottom=False)
    ax.text(0.50, 0.96, "Master Dark", fontsize=12.0, fontweight='bold',
            transform=ax.transAxes, ha='center', va='top')
    ax.text(0.50, 0.88, "(sc combined, bias-subtracted)", fontsize=11.0,
            transform=ax.transAxes, ha='center', va='top')
    ax.text(0.04, 0.12, f"Mean dark level: {np.mean(mdark):.1f} (count)",
            fontsize=10.0, transform=ax.transAxes, ha='left', va='bottom')
    ax.text(0.04, 0.04, f"Dark fluctuation: {np.std(mdark):.2f}", fontsize=10.0,
            transform=ax.transAxes, ha='left', va='bottom')
    plt.tight_layout()
    
    return mdark

mdark_sci, mdark_flat = make_mdark(dark_list_sci), make_mdark(dark_list_flat)

#%% creating a master flat
# - combining flat frames for each filter
mflat_list = []      # to save median-combined master flat for each filter
for flat_list in [flat_list_R, flat_list_I]:
    # - checking the basic info : check dates, exposure times, and filters.
    for i in np.arange(len(flat_list)):
        flat_hdr = fits.getheader(flat_list[i])
        print(f"\nFlat frame {i+1:d}")
        for keys in ['DATE-OBS', 'EXPTIME', 'FILTER']:
            print("  "+keys+" = "+str(flat_hdr[keys]))
    
    # - stacking flat frames
    flat_stack = []
    for i in np.arange(len(flat_list)):
        flat_data, flat_hdr = fits.getdata(flat_list[i], header=True)  
        # - bias and dark subtraction
        flat_bd = (flat_data - mbias.data - mdark_flat.data)
        # - flat scaling (with relative sensitivity = 1 at the maximum data point)
        flat_bdn = flat_bd/flat_bd.max()
        flat_stack.append(CCDData(data=flat_bdn, unit='adu'))
    
    # - sigma clipping
    mflat = combine(flat_stack, sigma_clip=True,
                      sigma_clip_low_thresh=3,
                      sigma_clip_high_thresh=3)
    
    # - flat scaling (with relative sensitivity = 1 at the maximum data point)
    # flat_med /= flat_med.max()
    
    # - save the master flat as fits file
    filter_now = flat_hdr['FILTER']     # specifying current filter
    flat_hdr['NFRAMES'] = len(flat_list)
    fits.writeto(DATA_DIR/f"MFlat{filter_now}.fits", mflat.data,
                 header=flat_hdr, overwrite=True)
    mflat_list.append(mflat.data)
    
    # - visualization of the master flat
    fig, ax = plt.subplots(1, 1, figsize=(5,3))
    vmin, vmax = interval.get_limits(mflat)
    ax.imshow(mflat, cmap='gray_r', origin='lower', vmin=vmin, vmax=vmax)
    ax.tick_params(axis='both', length=0.0, labelleft=False, labelbottom=False)
    ax.text(0.50, 0.96, f"Master Flat ({filter_now})", fontsize=12.0,
            fontweight='bold', transform=ax.transAxes, ha='center', va='top')
    ax.text(0.50, 0.88, "(sc combined, bias/dark-subtracted)",
            fontsize=11.0, transform=ax.transAxes, ha='center', va='top')
    ax.text(0.04, 0.12, f"Flat sensitivity range: {100*mflat.data.min():.1f}"
            + f" - {100*mflat.data.max():.1f}%", ha='left', va='bottom',
            fontsize=10.0, color='w', transform=ax.transAxes)
    ax.text(0.04, 0.04, f"Flat fluctuation: {100*np.std(mflat.data):.2f}%",
            fontsize=10.0, color='w', transform=ax.transAxes,
            ha='left', va='bottom')
    plt.tight_layout()
    

#%% Image preprocessing

def crrej(data, header, readnoise, method='CR'):
    gain = header['EGAIN'] # e-/ADU electronic gain
    
    # setting kwargs for LACOSMIC
    # two params in LACOSMIC were skipped: gain=2.0, readnoise=6.
    LACOSMIC_KEYS = dict(sigclip=4.5, sigfrac=0.5, objlim=5.0,
                 satlevel=np.inf, niter=4, #pssl=0.0
                 cleantype='medmask', fsmode='median', psfmodel='gauss',
                 psffwhm=2.5, psfsize=7, psfk=None, psfbeta=4.765)
    
    # initialize two CCDData objects
    if method == 'LA':
        ccd_LA = CCDData(data=[0], header=header, unit='adu')
        
        # Following should give identical result to IRAF L.A.Cosmic,
        # "m_LA" is the mask image
        m_LA, ccd_LA.data = astroscrappy.detect_cosmics(
                                data,
                                sepmed=False,  # IRAF LACosmic is sepmed=False
                                gain=gain,
                                readnoise=readnoise,
                                **LACOSMIC_KEYS)
        return ccd_LA, m_LA
    elif method == 'CR':
        ccd_CR = CCDData(data=[0], header=header, unit='adu')
        
        # Following is the "fastest" astroscrappy version. 
        m_CR, ccd_CR.data = astroscrappy.detect_cosmics(
                                data,
                                sepmed=True,
                                gain=gain, 
                                readnoise=readnoise,
                                **LACOSMIC_KEYS)
        return ccd_CR, m_CR
    else:
        raise ValueError('kwarg "method" should be either "LA" or "CR".')
    
    

def preproc(sci_list, mbias, mdark, mflat, rdnoise, show=True, save=True):
    for i in range(len(sci_list)):
        # bias subtraction, dark subtraction, and flat fielding
        sci_path = sci_list[i]
        sci_data, sci_hdr  = fits.getdata(sci_path, header=True)
        sci_data0 = sci_data.astype('float')    # 'int' type may cause error when calculating
        
        sci_data1 = sci_data0 - mbias    # Bias subtraction
        sci_data1 -= mdark   # Dark subtraction
        sci_data1 /= mflat    # Flat fielding
        
        # cosmic ray rejection
        sci_crrej, sci_mask = crrej(sci_data1, sci_hdr, rdnoise, method='CR')
        
        # visual inspection
        if show:
            fig, axs = plt.subplots(1, 3, figsize=(14,3))
            title = ["Raw Data",
                      "Preprocessed Data",
                      "Cosmic Ray Rejeccted Data"]
            for i, sci_data in enumerate([sci_data0, sci_data1, sci_crrej]):
                ax = axs[i]
                vmin, vmax = interval.get_limits(sci_data)
                ax.imshow(sci_data, cmap='viridis',
                          origin='lower', vmin=vmin, vmax=vmax)
                ax.tick_params(axis='both', length=0.0,
                               labelleft=False, labelbottom=False)
                if i == 0:
                    ax.text(0.04, 0.04, sci_path.name, fontsize=12.0, 
                            transform=ax.transAxes, ha='left', va='bottom')
                ax.text(0.50, 0.96, title[i], fontsize=12.0, fontweight='bold',
                        transform=ax.transAxes, ha='center', va='top')
            plt.tight_layout()
        
        # recording preprocessing history
        now = time.strftime("%Y-%m-%d %H:%M:%S (%Z = GMT%z)")
        sci_hdr['history'] = 'Preprocessed at ' + now
            
        # saving preprocessed image to a fits file
        scihdu = fits.PrimaryHDU(data=sci_crrej, header=sci_hdr)
        mskhdu = fits.ImageHDU(data=sci_mask.astype('int'), name='MASK')
        hdul = fits.HDUList([scihdu, mskhdu])
        hdul.writeto(DATA_DIR/('p'+sci_path.name+'s'), overwrite=True)
        print(f'Done: {sci_path.name}')


for sci_list, mflat in zip([sci_list_R, sci_list_I], mflat_list):
    preproc(sci_list, mbias.data, mdark_sci.data, mflat, rdnoise, show=True)
