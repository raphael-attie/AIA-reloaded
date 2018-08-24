"""
Script automating the creation of AIA rgb images and movies
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
ncores = 6
# Parent directory for the wavelength subdirectories. Adapt to personal case.
data_dir = os.path.expanduser('~/Data/SDO/AIA/event_2012_08_31/')
wvlt_dirs = [os.path.join(data_dir, wl) for wl in ['304', '171', '193']]
outputdir = os.path.join(data_dir, 'rgb')
data_files = [glob.glob(os.path.join(data_dir, '*.fits')) for data_dir in wvlt_dirs]
# sequence of image indices to process
#img_seq= range(0, len(data_files[0]))
img_seq= range(169, 170)


# High intensity percentiles for thresholding
percentiles_low = [25, 25, 25]
percentiles_high = [99.5, 99.99, 99.85]
# Intensity gamma scaling factors for tone-mapping the 3x12 bit high dynamic range into the 3x8 bit range
gamma_rgb=[2.8, 2.8, 2.4]

## Mix rgb channels. Mix red and green so the coronal loops don't look so green,
# Example of assuming fine tuning in CIELab color space:
# 60% more of green-171 and 30% less of blue-193 in red. 10% more of blue-193 in green, 10% more of 171-green in blue
rgbmix = np.array([[1.0, 0.6, -0.3],
                   [0.0, 1.0, 0.1],
                   [0.0, 0.1, 1.0]])

# Similar results without CIELab requires addition of 10% of the red-304 to the green channel.
# rgbmix = np.array([[1.0, 0.6, -0.3],
#                    [0.1, 1.0, 0.1],
#                    [0.0, 0.1, 1.0]])


scalemin = 20

# [Optional] Crop half the image over x = [0, 2048[ y=[0, 4096[
# crop=(slice(0,4096), slice(0,2048))
crop=None
# [Optional processing] CIELab parameters. Make 10% brighter, skew the chromaticity toward red and yellow
lab = [1, 0.96, 1.04]
# For contrast stretching on luminance (L) layer. L ranges in [0-255]
lmin = 0 # on a scale of [0-255]
# Path and file pattern of the output images
filename_lab = os.path.join(outputdir, 'im_min_%d_lab_%.2f_%.2f_%.2f_lmin%d'%(scalemin, *lab, lmin))

# Reference image used to compute the scaling values.
ref_rgb_files = [files[0] for files in data_files]

if __name__ == '__main__':


    # Get treshold for all images on 1st sample. Must be shared across workers in case of parallel processing
    rgblow, rgbhigh = visualization.rgb_high_low(ref_rgb_files, percentiles_low, percentiles_high)
    partial_process = functools.partial(visualization.process_rgb_image, data_files=data_files,
                                        rgblow=rgblow, rgbhigh=rgbhigh, scalemin=scalemin,
                                        gamma_rgb=gamma_rgb,
                                        rgbmix=rgbmix,
                                        lab=lab,
                                        lmin=lmin,
                                        filename_lab=filename_lab)


    if ncores >1:
        multiprocessing.set_start_method('spawn')
        p = multiprocessing.Pool(ncores)
        p.map(partial_process, img_seq)
        #p.map(partial_process, range(4))
        p.close()
    else:
        for i in img_seq:
            _ = partial_process(i)



    # With a 16:9 aspect ratio, crop over 3840 x 2160 around bottom half and output at full HD resolution (1920 x 1080)
    image_pattern_search = "im_min_%d_lab_*.jpeg"%scalemin
    crop = [3840, 2160, 128, 1935]
    frame_size = (1920, 1080)
    movie_filename = 'rgb_movie_3840x2160_1920x1080'
    fps = 30
    # Encode movie
    command = visualization.encode_video(outputdir, movie_filename, crop=crop, frame_size=frame_size, image_pattern_search=image_pattern_search)


    # full sun rescaled to 1080x1080 and padded at 1920 x 1080 for optimized youtube videos
    frame_size = (1080, 1080)
    padded_size = (1920, 1080)
    filename = 'rgb_movie_full_padded_1920_1080'
    # Number of frames per second
    fps = 30
    # Encode movie
    visualization.encode_video(outputdir, filename, fps=fps, frame_size=frame_size, padded_size=padded_size)