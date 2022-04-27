#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:21:27 2022

https://ps1images.stsci.edu/ps1_dr2_query.html

@author: hbahk
"""

import mastcasjobs

import os
from pathlib import Path
import sys
from astropy.io import fits
from astropy.table import Table, hstack

home = Path('/data1/hbahk')
wd = home / 'psz_candidate'
code_dir = wd / 'codes'
sys.path.append(code_dir.as_posix())

from PS1photoz.prepare_test_galaxies_tarrio import prepare_test_galaxies_tarrio
from PS1photoz.compute_photo_z_pan_tarrio import compute_photo_z_pan_tarrio

# get the WSID and password if not already defined
import getpass


def login_casjobs():
    if not os.environ.get('CASJOBS_USERID'):
        os.environ['CASJOBS_USERID'] = input('Enter Casjobs username:')
    if not os.environ.get('CASJOBS_PW'):
        os.environ['CASJOBS_PW'] = getpass.getpass('Enter Casjobs password:')

#%%


def get_zphot(cra, cdec, ang_limit, wd, path_result):
    query = f"""
    select o.objID, o.raMean, o.decMean,
    m.gMeanKronMag, m.rMeanKronMag, m.iMeanKronMag, m.zMeanKronMag, m.yMeanKronMag,
    psc.ps_score
    from fGetNearbyObjEq({cra},{cdec},{ang_limit}) nb
    inner join ObjectThin o on o.objid=nb.objid and o.nDetections>1
    inner join MeanObject m on o.objid=m.objid and o.uniquePspsOBid=m.uniquePspsOBid
    inner join HLSP_PS1_PSC.pointsource_scores psc on psc.objid=nb.objid and psc.ps_score < 0.83"""

    jobs = mastcasjobs.MastCasJobs(context="PanSTARRS_DR2")
    tab = jobs.quick(query, task_name="python cone search")

    gra, gdec = tab['raMean'], tab['decMean']
    g, r, i, z, y = tab['gMeanKronMag'], tab['rMeanKronMag'], tab['iMeanKronMag'], tab['zMeanKronMag'], tab['yMeanKronMag'],

    path_save = wd/'data'/'MC_overdensity'/'PS1'/'temp.fits'
    path_ebv = wd/'codes'/'PS1photoz'/'lambda_sfd_ebv.fits'
    path_train = wd/'codes'/'PS1photoz'/'ps1_training_tarrio.fits'
    path_out = wd/'data'/'MC_overdensity'/'PS1'/'temp_out.fits'

    prepare_test_galaxies_tarrio(ra=gra, dec=gdec, r_kron=r, g=g, r=r, i=i, z=z, y=y, feat_type='kron',
                                 dir_name_cat_test=path_save, dir_name_ebvmap=path_ebv)

    compute_photo_z_pan_tarrio(path_train, path_save, path_out, feat_type='kron')

    hdu = fits.open(path_out)
    ztbl = Table(hdu[1].data)
    zmask = ztbl['z_phot']!=-1
    result = hstack([tab[zmask], ztbl[zmask]])
    result.write(path_result, format='csv', overwrite=True)

    path_save.unlink()
    path_out.unlink()
    return result

#%%

if __name__ == '__main__':
    cra = 229.19051197148607
    cdec = -1.0172220494487687
    ang_limit = 15 # in arcmin

    path_result = wd/'data'/'MC_overdensity'/'PS1'/'test.csv'

    rtbl = get_zphot(cra, cdec, ang_limit, wd, path_result)




