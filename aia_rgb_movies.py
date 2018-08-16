import os
import glob
import numpy as np
import calibration as calibration
import visualization as visualization
from multiprocessing import Pool
import functools

data_dir = '/Users/rattie/Data/SDO/AIA/event_2012_08_31/'
outputdir = os.path.join(data_dir, 'rgb')
wavelengths = ['304',
               '171',
               '193']
data_dirs = [os.path.join(data_dir, wl) for wl in wavelengths]


data_files = [glob.glob(os.path.join(data_dir, '*.fits')) for data_dir in data_dirs]

# Get the high percentiles for rescaling the intensity for all images
pdatargb0 = [calibration.aiaprep(data_files[j][0], cropsize=4096) for j in range(3)]
# Get max percentile values for each channel.
percentiles = [99.5, 99.99, 99.85]
rgbhigh = np.array([visualization.compute_intensity_high(pdatargb0[i], percentiles[i]) for i in range(3)])
partial_process = functools.partial(visualization.process_rgb_image, data_files=data_files, rgbhigh=rgbhigh, outputdir=outputdir)

# For parallel processing
p = Pool(4)
p.map(visualization.process_rgb_image, range(len(data_files[0])))

# For non-parallel processing
# for i in range(len(data_files[0])):
#     process_rgb_image(i)

## For creating the movie, use ffmpeg. E.g from terminal:
# ffmpeg -framerate 30 -i 'im_rgb_gamma_2.6_2.8_2.4_%3d.jpeg' -c:v libx264 -preset slow -crf 10 -r 30 -vf "scale=2048:2048" -pix_fmt yuv420p rgb_movie_2048.mp4
