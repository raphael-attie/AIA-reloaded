import os, glob
from visualization import RGBMixer

## Create rgb image using the default aia rgb mixer.
# This example provides the data file paths and output directory

aia_mixer = RGBMixer(
    data_files = [glob.glob('../aia_data/*.%s*.fits'%wavel) for wavel in ['304', '171', '193']],
    outputdir = os.path.abspath('../aia_data/'))
aia_mixer.set_aia_default()

# Example of cropping. Combined with the extra 2x downsampled version of the RGBMixer, this will create also images in
# full HD (1920x1080) that do not need require ffmpeg to apply a lossy, blurring rescaling.
aia_mixer.crop = (slice(1935,1935+2160), slice(128,128+3840))

aia_mixer.filename_lab = 'im_crop_lab'

_ = aia_mixer.process_rgb(0)


#### This example provides paths to input directories instead of file paths, e.g useful to build a terminal command

# Here, '304', '171', '193' are the names of sub-directories the parent directory data_dir
# aia_mixer = RGBMixer(
#     data_dir = os.path.expanduser('~/Data/SDO/AIA/event_2012_08_31/'),
#     wavel_dirs= ['304', '171', '193'],
#     outputdir = os.path.abspath('../aia_data/rgb/'))
#
# aia_mixer.set_aia_default()
# _ = aia_mixer.process_rgb(0)

