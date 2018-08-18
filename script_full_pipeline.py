"""
Script automating the creation of AIA rgb images and movies
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
outputdir = os.path.join(data_dir, 'rgb')
# Default intensity percentiles for the rescaling of each channel
percentiles = [99.5, 99.99, 99.85]
# Intensity gamma scaling factors for tone-mapping the 3x12 bit high dynamic range into the 3x8 bit range
gamma_rgb=[2.8, 2.8, 2.4]
# Blue tone factor: tune the "hot" vs "cold" look of the sun. The greater the value, the colder the sun will look
btf = 0.2


if __name__ == '__main__':

    data_files = [glob.glob(os.path.join(data_dir, '*.fits')) for data_dir in wvlt_dirs]
    rgbhigh = visualization.get_rgb_high(data_files, percentiles=percentiles)

    partial_process = functools.partial(visualization.process_rgb_image, data_files=data_files, rgbhigh=rgbhigh,
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


    # Directory of the rgb images
    images_dir = outputdir

    # With a 16:9 aspect ratio, crop over 3840 x 2160 around bottom half and output at full HD resolution (1920 x 1080)
    crop = [3840, 2160, 128, 1935]
    frame_size = (1920, 1080)
    filename = 'rgb_movie_3840x2160_1920x1080'
    fps = 30
    # Encode movie
    visualization.encode_video(images_dir, filename, crop=crop, frame_size=frame_size)

    # full sun rescaled to 1080x1080 and padded at 1920 x 1080 for optimized youtube videos
    frame_size = (1080, 1080)
    padded_size = (1920, 1080)
    filename = 'rgb_movie_full_padded_1920_1080'
    # Number of frames per second
    fps = 30
    # Encode movie
    visualization.encode_video(images_dir, filename, fps=fps, frame_size=frame_size, padded_size=padded_size)