"""
Script automating the creation of AIA rgb movies.
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
import visualization
import functools
import multiprocessing

# Set parallelization. To disable, set ncores to 1 (or any smaller number)
ncores = 4
# Parent directory for the wavelength subdirectories
data_dir = '/Users/rattie/Data/SDO/AIA/event_2012_08_31/'
# Adapt to personal case.
wvlt_dirs = [os.path.join(data_dir, wl) for wl in ['304', '171', '193']]
outputdir = os.path.join(data_dir, 'rgb')
# Default intensity percentiles for the rescaling of each channel
percentiles = [99.5, 99.99, 99.85]


if __name__ == '__main__':

    data_files = [glob.glob(os.path.join(data_dir, '*.fits')) for data_dir in wvlt_dirs]
    rgbhigh = visualization.get_rgb_high(data_files, percentiles=percentiles)

    partial_process = functools.partial(visualization.process_rgb_image, data_files=data_files, rgbhigh=rgbhigh,
                                        outputdir=outputdir)
    if ncores >1:
        multiprocessing.set_start_method('spawn')
        p = multiprocessing.Pool(4)
        p.map(partial_process, range(len(data_files[0])))
        #p.map(partial_process, range(4))
        p.close()
    else:
        for i in range(len(data_files[0])):
            _ = partial_process(i)


