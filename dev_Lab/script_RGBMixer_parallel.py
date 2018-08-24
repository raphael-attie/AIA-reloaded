import os
from visualization import RGBMixer

# Set parallelization. To disable, set ncores to 1. Try to leave 1 or 2 cores available:
# E.g., if you have 8 cores in your computer (whether virtual cores or physical cores). Use maximum ncores = 6
ncores = 4
# List of file numbers to process.
file_range = range(8)


if __name__ == '__main__':

    aia_mixer = RGBMixer(
        data_dir=os.path.expanduser('~/Data/SDO/AIA/event_2012_08_31/'),
        wavel_dirs=['304', '171', '193'],
        outputdir=os.path.abspath('../aia_data/rgb/'))
    aia_mixer.set_aia_default()

    aia_mixer.process_rgb_list(ncores, file_range)
