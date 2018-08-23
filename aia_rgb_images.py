"""
Script automating the creation of AIA rgb images.
This examples assumes that AIA raw fits files from the 3 different wavelengths are in three different directories under the
same parent directory.

The script creates first rgb images that are written to disk.
The intensity in each channel is scaled to exploit the high dynamic range of the images using default tone-mapping values.
Next, ffmpeg is called to create a movie from the jpeg files.

Personal note: Opencv 3.4, when compiled with multithreading, imposes to change the multiprocessing start method
with: multiprocessing.set_start_method('spawn'). Otherwise, the whole process exits silently.
see https://github.com/opencv/opencv/issues/5150
In addition, cv2.setNumThreads(0) will disable completely multithreading for opencv. Default is to use all available.

"""


import os
import glob
import numpy as np
import cv2
import visualization
import functools
import multiprocessing
#  disable multithreading in opencv. Default is to use all available, which is rather inefficient in this context
cv2.setNumThreads(0)

# Set parallelization. To disable, set ncores to 1 (or any smaller number)
ncores = 4
# Parent directory for the wavelength subdirectories. Adapt to personal case.
data_dir = os.path.expanduser('~/Data/SDO/AIA/event_2012_08_31/')
wvlt_dirs = [os.path.join(data_dir, wl) for wl in ['304', '171', '193']]
outputdir = os.path.join(data_dir, 'rgb')

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

filename_rgb = 'im_rgb_pmin_%.2f_%.2f_%.2f_pmax_%.2f_%.2f_%.2f_g_%0.1f_%0.1f_%0.1f'%(*percentiles_low, *percentiles_high, *gamma_rgb)
#filename_lab = 'im_lab_%.2f_%.2f_%.2f_lmin_%d'%(*lab, lmin)

# Data initialial rgb files for the 3 wavelengths
init_rgb_files = [[glob.glob('../aia_data/*304*.fits')[0]],
                  [glob.glob('../aia_data/*171*.fits')[0]],
                  [glob.glob('../aia_data/*193*.fits')[0]]]


if __name__ == '__main__':

    data_files = [glob.glob(os.path.join(data_dir, '*.fits')) for data_dir in wvlt_dirs]
    # Get treshold for all images on 1st sample. Must be shared across workers in case of parallel processing
    rgblow, rgbhigh = visualization.rgb_high_low(init_rgb_files, percentiles_low, percentiles_high)
    # partial_process = functools.partial(visualization.process_rgb_image, data_files=data_files,
    #                                     gamma_rgb=gamma_rgb,
    #                                     rgbmix=rgbmix,
    #                                     lab=lab,
    #                                     lmin=lmin,
    #                                     rgblow=rgblow, rgbhigh=rgbhigh,
    #                                     outputdir=outputdir)

    partial_process = functools.partial(visualization.process_rgb_image, data_files=data_files,
                                        gamma_rgb=gamma_rgb,
                                        rgbmix=rgbmix,
                                        rgblow=rgblow, rgbhigh=rgbhigh,
                                        outputdir=outputdir)

    if ncores >1:
        multiprocessing.set_start_method('spawn')
        p = multiprocessing.Pool(ncores)
        p.map(partial_process, range(len(data_files[0])))
        #p.map(partial_process, range(4))
        p.close()
    else:
        for i in range(len(data_files[0])):
            _ = partial_process(i)


