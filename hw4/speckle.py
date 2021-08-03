#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 01:23:25 2021

@author: michail
"""
import numpy as np
import numpy.ma as ma
import json
from astropy.io import fits
from PIL import Image
from scipy import ndimage
from photutils.detection import DAOStarFinder
from astropy.stats import mad_std

#get data
data = fits.open('speckledata.fits')[2].data

# get mean image
data_mean = np.sum(data, axis=0)
mval = np.max(data_mean)
contrast = 1/200 
mean_img = Image.fromarray(data_mean)
Image.fromarray(data_mean*contrast).convert("L").resize((512, 512)).save('mean.png')

#fast fourier transform
fdata = np.abs(np.fft.fftshift(np.fft.fft2(data, s=None, axes=(-2, -1)))**2)
img = Image.fromarray(np.sum(fdata, axis=0))
contrast = 1.5 * 10**-9
Image.fromarray(np.sum(fdata, axis=0)*contrast).convert("L").resize((512, 512)).save('fourier.png')

# make a circle mask
h, w = np.shape(img)
center = (int(w/2), int(h/2))
radius = 50 
Y, X = np.ogrid[:h, :w]
mask = np.sqrt((X - center[0])**2 + (Y-center[1])**2) <= radius

# reduce noize out of central circle
mnoize = np.mean(ma.masked_array(img, mask=mask))
img -= mnoize

# averaging the value by the angles 
img_rot = np.mean(list(map(lambda ang : ndimage.rotate(img, ang, reshape=False), np.linspace(-180, 180, 100))), axis=0)
contrast = 3 * 10**-9
Image.fromarray(img_rot*contrast).resize((512, 512)).convert("L").save('rotaver.png')

# leave only the central part
crop_img = ma.masked_array(np.divide(img, img_rot), mask=np.logical_not(mask)).filled(0)

# finaly do inverse fast fourier transform and get image
final = (np.fft.ifftshift(np.fft.ifft2(crop_img)))
contrast = 5 * 10**3
n_mas = (np.abs(final)* contrast).astype('float64')
imgfinal = Image.fromarray(n_mas)
imgfinal.resize((512, 512)).convert("L").save('binary.png')

# find angular distance
bkg_sigma = mad_std(n_mas)  
daofind = DAOStarFinder(fwhm=4., threshold=3.*bkg_sigma)  
sources = daofind(n_mas)
sources = sources.group_by('peak')
x = np.array(sources['xcentroid'][-2:])
y = np.array(sources['ycentroid'][-2:])
cord1, cord2 = zip(x, y)
cord2, cord1 = np.array(cord1), np.array(cord2)
cord3 = np.array([sources['xcentroid'][-3], sources['ycentroid'][-3]])
ang_len = np.sqrt(np.sum(((cord1 - cord2)**2 + (cord1 - cord3)**2)/2)) * 0.0206

# save results
file = {}
file["distance"] = (float('{:.3f}'.format(ang_len)))
json_file = json.dumps(file) 
with open("binary.json", "w") as outfile: 
    outfile.write(json_file)
        