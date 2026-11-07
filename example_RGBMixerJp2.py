import os, glob
from visualization import RGBMixer

## Create rgb image using the default aia jp2 rgb mixer.
# Pre-calibrated JPEG2000 (.jp2) files do not require aiaprep or gamma/CIELab tone-mapping,
# only an equivalent RGB mixing matrix.

aia_mixer = RGBMixer(
    data_files=[glob.glob('samples/*.%s*.jp2' % wavel) for wavel in ['304', '171', '193']],
    jp2=True,
    outputdir=os.path.abspath('samples/'))
aia_mixer.set_jp2_default()

# Output the full-size (4096x4096) image (uses default filename_lab = 'im_lab_jp2')
_ = aia_mixer.process_rgb(0)

# Example of cropping in (x, y) = (cols, rows) coordinates (3840x2160 16:9 crop)
aia_mixer.crop = (slice(128, 128 + 3840), slice(1935, 1935 + 2160))

aia_mixer.filename_lab = 'im_crop_lab_jp2'

_ = aia_mixer.process_rgb(0)
