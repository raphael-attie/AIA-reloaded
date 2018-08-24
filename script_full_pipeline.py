"""
Script automating the creation of AIA rgb images and movies.
This examples assumes that AIA raw fits files from the 3 different wavelengths are in three different directories under the
same parent directory.

The script creates first rgb images that are written to disk.
The intensity in each channel is scaled to exploit the high dynamic range of the images using both linear and non-linear
rescaling method (e.g: linear contrast stretching and "gamma" curves).
Color mixing happens first in the RGB color space, and fine tuned in CIELab color space.
Next, ffmpeg is called to create a movie from the jpeg files.


Personal note: Opencv 3 compiled with multithreading imposes to change the multiprocessing start method
with multiprocessing.set_start_method('spawn'). Otherwise, the whole process exits silently.
see https://github.com/opencv/opencv/issues/5150
In addition, cv2.setNumThreads(0) will disable completely multithreading for opencv. Default is to use all available.
"""

import os
import visualization

# Set parallelization. To disable, set ncores to 1. Try to leave 1 or 2 cores available:
# E.g., if you have 8 cores in your computer (whether virtual cores or physical cores). Use maximum ncores = 6
ncores = 4
# List of file numbers to process.
file_range = range(8)


if __name__ == '__main__':

    aia_mixer = visualization.RGBMixer(
        data_dir=os.path.expanduser('~/Data/SDO/AIA/event_2012_08_31/'),
        wavel_dirs=['304', '171', '193'],
        outputdir=os.path.abspath('../aia_data/rgb/'))
    aia_mixer.set_aia_default()

    aia_mixer.process_rgb_list(ncores, file_range)


    ## Create .mp4 videos

    # Set the image pattern search. Default here uses the default file names of the class above.
    # Image numbers must all be padded equally.
    image_pattern_search = "im_lab_*.jpeg"

    # Create video With a 16:9 aspect ratio, crop over 3840 x 2160 around bottom half
    # and output at full HD resolution (1920 x 1080) at 30 frames per second.
    crop = [3840, 2160, 128, 1935]
    frame_size = (1920, 1080)
    movie_filename = 'rgb_movie_3840x2160_1920x1080'
    fps = 30
    # Encode movie
    command = visualization.encode_video(aia_mixer.outputdir, movie_filename, crop=crop, frame_size=frame_size, image_pattern_search=image_pattern_search)


    # full sun rescaled to 1080x1080 and padded to 1920 x 1080 for optimized youtube videos
    frame_size = (1080, 1080)
    padded_size = (1920, 1080)
    filename = 'rgb_movie_full_padded_1920_1080'
    # Number of frames per second
    fps = 30
    # Encode movie
    visualization.encode_video(aia_mixer.outputdir, filename, fps=fps, frame_size=frame_size, padded_size=padded_size)