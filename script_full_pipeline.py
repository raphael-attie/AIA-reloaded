"""
Script automating the creation of AIA rgb images and movies
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
ncores = 1
# Parent directory for the wavelength subdirectories. Adapt to personal case.
data_dir = os.path.expanduser('~/Data/SDO/AIA/event_2012_08_31/')
wvlt_dirs = [os.path.join(data_dir, wl) for wl in ['304', '171', '193']]
outputdir = os.path.join(data_dir, 'rgb')
# sequence of image indices to process
img_seq= range(169,170)

# High intensity percentiles for thresholding
percentiles_low = [25, 25, 25]
percentiles_high = [99.5, 99.99, 99.85]
# Intensity gamma scaling factors for tone-mapping the 3x12 bit high dynamic range into the 3x8 bit range
gamma_rgb=[2.8, 2.8, 2.4]
# Mix rgb channels. Mix red and green so the coronal loops don't look so green,
# and so the chromosphere look less red/magenta but more yellow-ish. Fine tuning can be done in CIELAB color space (see below)

# V1
# rgbmix = np.array([[1.0, 0.6, -0.3],
#                    [0.1, 1.0, 0.1],
#                    [0.0, 0.1, 1.0]])


#V4
rgbmix = np.array([[1.0, 0.6, -0.3],
                   [0.0, 1.0, 0.1],
                   [0.0, 0.1, 1.0]])

scalemin = 20 #25 for V1

# [Optional] Crop half the image over x = [0, 2048[ y=[0, 4096[
# crop=(slice(0,4096), slice(0,2048))
crop=None
# [Optional processing] CIELab parameters. Make 10% brighter, skew the chromaticity toward red and yellow
lab = [1, 0.95, 1.02]
# For contrast stretching on luminance (L) layer. L ranges in [0-255]
lmin = 0 # on a scale of [0-255]

filename_rgb = 'im_rgb_pmin_%.1f_%.1f_%.1f_pmax_%.2f_%.2f_%.2f_smin%d_g_%0.1f_%0.1f_%0.1f_V4'%(*percentiles_low, *percentiles_high, scalemin, *gamma_rgb)
filename_lab = 'im_min_%d_lab_%.2f_%.2f_%.2f_lmin%d_V4'%(scalemin, *lab, lmin)

# Data initialial rgb files for the 3 wavelengths
init_rgb_files = [[glob.glob('../aia_data/*304*.fits')[0]],
                  [glob.glob('../aia_data/*171*.fits')[0]],
                  [glob.glob('../aia_data/*193*.fits')[0]]]


if __name__ == '__main__':

    data_files = [glob.glob(os.path.join(data_dir, '*.fits')) for data_dir in wvlt_dirs]
    # Get treshold for all images on 1st sample. Must be shared across workers in case of parallel processing
    rgblow, rgbhigh = visualization.rgb_high_low(init_rgb_files, percentiles_low, percentiles_high)
    partial_process = functools.partial(visualization.process_rgb_image, data_files=data_files,
                                        rgblow=rgblow, rgbhigh=rgbhigh, scalemin=scalemin,
                                        gamma_rgb=gamma_rgb,
                                        rgbmix=rgbmix,
                                        lab=lab,
                                        lmin=lmin,
                                        outputdir=outputdir, filename_rgb=filename_rgb, filename_lab=filename_lab)


    # partial_process = functools.partial(visualization.process_rgb_image, data_files=data_files,
    #                                     gamma_rgb=gamma_rgb,
    #                                     rgbmix=rgbmix,
    #                                     rgblow=rgblow, rgbhigh=rgbhigh,
    #                                     outputdir=outputdir)

    if ncores >1:
        multiprocessing.set_start_method('spawn')
        p = multiprocessing.Pool(ncores)
        p.map(partial_process, img_seq)
        #p.map(partial_process, range(4))
        p.close()
    else:
        for i in img_seq:
            _ = partial_process(i)


    # Directory of the rgb images
    images_dir = outputdir

    # # With a 16:9 aspect ratio, crop over 3840 x 2160 around bottom half and output at full HD resolution (1920 x 1080)
    # crop = [3840, 2160, 128, 1935]
    # frame_size = (1920, 1080)
    # movie_filename = 'rgb_movie_3840x2160_1920x1080'
    # fps = 30
    # # Encode movie
    # command = visualization.encode_video(images_dir, movie_filename, crop=crop, frame_size=frame_size)


    # # full sun rescaled to 1080x1080 and padded at 1920 x 1080 for optimized youtube videos
    # frame_size = (1080, 1080)
    # padded_size = (1920, 1080)
    # filename = 'rgb_movie_full_padded_1920_1080'
    # # Number of frames per second
    # fps = 30
    # # Encode movie
    # visualization.encode_video(images_dir, filename, fps=fps, frame_size=frame_size, padded_size=padded_size)