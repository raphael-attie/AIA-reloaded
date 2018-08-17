"""
Example for writing an aia rgb image from 3 raw fits files.
"""

import glob
import numpy as np
from calibration import aiaprep
from visualization import compute_intensity_high, process_rgb_image

# Data test directories for the 3 wavelengths
data_files = [[glob.glob('../aia_data/*304*.fits')[0]],
              [glob.glob('../aia_data/*171*.fits')[0]],
              [glob.glob('../aia_data/*193*.fits')[0]]]

# High intensity percentiles for thresholding
percentiles = [99.5, 99.99, 99.85]
# Intensity gamma scaling factors for tone-mapping the 3x12 bit high dynamic range into the 3x8 bit range
gamma_rgb=[2.8, 2.8, 2.4]
# Blue tone factor: tune the "hot" vs "cold" look of the sun. The greater the value, the colder the sun will look
btf = 0.3

if __name__ == '__main__':

    pdatargb = [aiaprep(data_files[j][0], cropsize=4096) for j in range(3)]
    rgbhigh = np.array([compute_intensity_high(pdatargb[j], percentiles[j]) for j in range(3)])
    _, outputfile = process_rgb_image(0, data_files, rgbhigh, gamma_rgb=gamma_rgb, btf=btf, outputdir='../aia_data/')


