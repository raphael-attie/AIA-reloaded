import os, glob
import numpy as np
import calibration
import visualization

# Parent directory for the wavelength subdirectories. Adapt to personal case.
data_dir = os.path.expanduser('~/Data/SDO/AIA/event_2012_08_31/')
wvlt_dirs = [os.path.join(data_dir, wl) for wl in ['304', '171', '193']]
outputdir = os.path.join(data_dir, 'rgb')

# High intensity percentiles for thresholding
percentiles_low = [25, 25, 25]
percentiles_high = [99.5, 99.99, 99.85]
# Intensity gamma scaling factors for tone-mapping the 3x12 bit high dynamic range into the 3x8 bit range
gamma_rgb=[2.8, 2.8, 2.4]
# Mix rgb channels. Mix red and green so the coronal loops don't look so green,
# and so the chromosphere look less red/magenta but more yellow-ish. Fine tuning can be done in CIELAB color space (see below)
rgbmix = np.array([[1.0, 0.4, -0.2],
                   [0.1, 1.0, 0.0],
                   [0.0, 0.0, 1.0]])
# [Optional processing] CIELab parameters. Make 10% brighter, skew the chromaticity toward red and yellow
lab = [1.1, 1.05, 1.10]
# Minimum value for contrast stretching on luminance (L) layer. L ranges in [0-255]
lmin = 25 # on a scale of [0-255]

data_files = [glob.glob(os.path.join(data_dir, '*.fits')) for data_dir in wvlt_dirs]

rgblist = []
for i in range(10):
    rgb = [calibration.aiaprep(data_files[j][i], cropsize=4096) for j in range(3)]
    rgblist.append(rgb)

rgblow = np.array([np.percentile(rgblist[0][j], percentiles_low[j]) for j in range(3)])
rgbhigh = np.array([np.percentile(rgblist[0][j], percentiles_high[j]) for j in range(3)])

scaled_rgblist = []
for i in range(10):
    im_rgb255, newmins, newmaxs = visualization.scale_rgb(rgblist[i], rgblow, rgbhigh, gamma_rgb=gamma_rgb, rgbmix=rgbmix)
    scaled_rgblist.append(im_rgb255)
    newdiffs = newmins - newmaxs
    print(i)
    print('%.2f , %.2f, %.2f' % (newmins[0], newmins[1], newmins[2]))
    print('%.2f , %.2f, %.2f' % (newmaxs[0], newmaxs[1], newmaxs[2]))
    print('%.2f , %.2f, %.2f' % (newdiffs[0], newdiffs[1], newdiffs[2]))


print(rgblow)
for i in range(10):
    print(rgblist[i][0].min(), rgblist[i][1].min(), rgblist[i][2].min())


print(rgbhigh)
for i in range(10):
    print(rgblist[i][0].max(), rgblist[i][1].max(), rgblist[i][2].max())


