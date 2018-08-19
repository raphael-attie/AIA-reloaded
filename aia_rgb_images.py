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
import visualization
import functools
import multiprocessing

# Set parallelization. To disable, set ncores to 1 (or any smaller number)
ncores = 1
# Parent directory for the wavelength subdirectories
data_dir = os.path.expanduser('~/Data/SDO/AIA/event_2012_08_31/')
# Adapt to personal case.
wvlt_dirs = [os.path.join(data_dir, wl) for wl in ['304', '171', '193']]
# Default intensity percentiles for the rescaling of each channel
percentiles = [99.5, 99.99, 99.85]
# Intensity gamma scaling factors for tone-mapping the 3x12 bit high dynamic range into the 3x8 bit range
gamma_rgb=[2.8, 2.8, 2.4]
# Blue tone factor: tune the "hot" vs "cold" look of the sun. The greater the value, the colder the sun will look
btf = 0.2
# output directory and filename for the jpeg images. These filename will be appended with the image number:
outputdir = os.path.join(data_dir, 'rgb')
filename = 'im_rgb_gamma_%0.1f_%0.1f_%0.1f_btf_%0.1f'%(*gamma_rgb, btf)

if __name__ == '__main__':

    data_files = [glob.glob(os.path.join(data_dir, '*.fits')) for data_dir in wvlt_dirs]
    rgbhigh = visualization.get_rgb_high(data_files, percentiles=percentiles)

    partial_process = functools.partial(visualization.process_rgb_image, data_files=data_files, rgbhigh=rgbhigh,
                                        btf=btf,
                                        gamma_rgb=gamma_rgb,
                                        outputdir=outputdir,
                                        filename = filename)
    if ncores >1:
        multiprocessing.set_start_method('spawn')
        p = multiprocessing.Pool(ncores)
        p.map(partial_process, range(len(data_files[0])))
        #p.map(partial_process, range(4))
        p.close()
    else:
        for i in range(len(data_files[0])):
            _ = partial_process(i)


