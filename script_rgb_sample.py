"""
Example for writing an aia rgb image from 3 raw fits files with the default color mixer for AIA
See class definition in visualization.RGBMixer
"""
import os
import glob
import numpy as np
import visualization
import calibration


# Parent directory for the wavelength subdirectories. Adapt to personal case.
data_dir = os.path.expanduser('~/Data/SDO/AIA/event_2012_08_31/')
wvlt_dirs = [os.path.join(data_dir, wl) for wl in ['304', '171', '193']]
outputdir = os.path.join(data_dir, 'rgb')
data_files = [glob.glob(os.path.join(data_dir, '*.fits')) for data_dir in wvlt_dirs]

# image index to process
i = 0

# Intensity percentiles for linear scaling
percentiles_low = [25, 25, 25]
percentiles_high = [99.5, 99.99, 99.85]
# Intensity inverse gamma scaling factors for tone-mapping the 3x12 bit high dynamic range into the 3x8 bit range
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

# [Optional] Minimum scaling value for contrast stretching after gamma-scaling
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

# initiali rgb files for the 3 wavelengths
init_rgb_files = [[glob.glob('../aia_data/*304*.fits')[0]],
                  [glob.glob('../aia_data/*171*.fits')[0]],
                  [glob.glob('../aia_data/*193*.fits')[0]]]


if __name__ == '__main__':


    # Get treshold for all images on 1st sample. Must be shared across workers in case of parallel processing
    pdatargb = [calibration.aiaprep(data_files[j][0]) for j in range(3)]
    rgblow = np.array([np.percentile(pdatargb[j], percentiles_low[j]) for j in range(3)])
    rgbhigh = np.array([np.percentile(pdatargb[j], percentiles_high[j]) for j in range(3)])

    bgr_stack1, bgr_stack2 = visualization.process_rgb_image(i, data_files=data_files,
                                        rgblow=rgblow, rgbhigh=rgbhigh, scalemin=scalemin,
                                        gamma_rgb=gamma_rgb,
                                        rgbmix=rgbmix,
                                        lab=lab,
                                        lmin=lmin,
                                        filename_lab=filename_lab)

