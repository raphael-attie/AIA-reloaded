import os, glob
from visualization import RGBMixer

## Create rgb image using the default aia rgb mixer.
# This example provides the data file paths and output directory

aia_mixer = RGBMixer(
    data_files = [glob.glob('../aia_data/*.%s*.fits'%wavel) for wavel in ['304', '171', '193']],
    outputdir = os.path.abspath('../aia_data/'))
aia_mixer.set_aia_default()

_ = aia_mixer.process_rgb(0)


# This example provides paths to input directories instead of file paths.
# Here, '304', '171', '193' are the names of sub-directories the parent directory data_dir
# aia_mixer = RGBMixer(
#     data_dir = os.path.expanduser('~/Data/SDO/AIA/event_2012_08_31/'),
#     wavel_dirs= ['304', '171', '193'],
#     outputdir = os.path.abspath('../aia_data/rgb/'))
#
# aia_mixer.set_aia_default()
# _ = aia_mixer.process_rgb(0)

