"""
Example for writing an aia rgb image from 3 raw fits files.
"""

import glob
import numpy as np
from calibration import aiaprep
from visualization import process_rgb_image


# Data test rgb files for the 3 wavelengths
data_files = [[glob.glob('../aia_data/*304*.fits')[0]],
              [glob.glob('../aia_data/*171*.fits')[0]],
              [glob.glob('../aia_data/*193*.fits')[0]]]

# High intensity percentiles for thresholding
percentiles_low = [25, 25, 25]
percentiles_high = [99.5, 99.99, 99.85]
# Intensity gamma scaling factors for tone-mapping the 3x12 bit high dynamic range into the 3x8 bit range
gamma_rgb=[2.8, 2.8, 2.4]
# Mix rgb channels. Mix red and green so the coronal loops don't look so green,
# and so the chromosphere look less red/magenta but more yellow-ish. Fine tuning can be done in CIELAB color space (see below)
rgbmix = np.array([[1.0, 0.4, -0.2],
                   [0.1, 1.0, 0.0],
                   [0.0, 0.0, 1.0]])

# [Optional] Crop half the image over x = [0, 2048[ y=[0, 4096[
# crop=(slice(0,4096), slice(0,2048))
crop=None
# [Optional processing] CIELab parameters. Make 10% brighter, skew the chromaticity toward red and yellow
lab = [1.1, 1.05, 1.10]
# Minimum value for contrast stretching on luminance (L) layer. L ranges in [0-255]
lmin = 25 # on a scale of [0-255]



# output directory and filename for the jpeg images. These filename will be appended with the image number:
outputdir = '../aia_data/'
filename_rgb = 'im_rgb_pmin_%.2f_%.2f_%.2f_pmax_%.2f_%.2f_%.2f_g_%0.1f_%0.1f_%0.1f'%(*percentiles_low, *percentiles_high, *gamma_rgb)
#filename_lab = 'im_lab_%.2f_%.2f_%.2f_lmin_%d'%(*lab, lmin)


if __name__ == '__main__':

    pdatargb = [aiaprep(data_files[j][0], cropsize=4096) for j in range(3)]
    rgblow = np.array([np.percentile(pdatargb[j], percentiles_low[j]) for j in range(3)])
    rgbhigh = np.array([np.percentile(pdatargb[j], percentiles_high[j]) for j in range(3)])
    bgr_stack, outputfile = process_rgb_image(0, data_files, rgblow, rgbhigh, gamma_rgb=gamma_rgb, rgbmix=rgbmix,
                                              lab=lab, lmin=lmin, crop=crop,
                                              outputdir=outputdir, filename=filename_rgb)



