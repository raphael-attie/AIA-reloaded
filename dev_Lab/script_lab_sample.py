""""
Example for writing an aia rgb image from 3 raw fits files.
"""

import os, glob
import numpy as np
from calibration import aiaprep
import cv2


def scale_rgb(rgb, rgbmin, rgbhigh, gamma_rgb=[2.8, 2.8, 2.4], btf=0):

    rgb_gamma = 1 / np.array(gamma_rgb)
    # Get the rgb stack and normalize to 255.

    # Normalize to [0-1] using the 2 percentiles
    for i in range(3):
        rgb[i] = (rgb[i] - rgbmin[i]) * 1 / (rgbhigh[i] - rgbmin[i])
        rgb[i].clip(0, 1, out=rgb[i])

    # Gamma scale
    red, green, blue = [(channel ** gamma) for (channel, gamma) in zip(rgb, rgb_gamma)]

    # V1
    # nred = red
    # ngreen = green
    # nblue = blue

    # # V2
    # nred = red +0.4*green - btf*blue
    # ngreen = green
    # nblue = blue

    # V3
    nred = red +0.4*green - btf*blue
    ngreen = green + 0.1*red
    nblue = blue


    # Stack channels
    rgb_stack = np.stack((nred, ngreen, nblue), axis=-1).astype(np.float32)
    newrgbmin = rgb_stack.min(axis=0).min(axis=0)
    newrgbmax = rgb_stack.max(axis=0).max(axis=0)
    rgb_stack = (rgb_stack - newrgbmin) * 255 / (newrgbmax - newrgbmin)

    return rgb_stack


def scale_rgb_mod(rgb, rgbmin, rgbhigh, gamma_rgb=[2.8, 2.8, 2.4], btf=0):

    rgb_gamma = 1 / np.array(gamma_rgb)
    # Get the rgb stack and normalize to 255.

    # Normalize to [0-1] using the 2 percentiles
    for i in range(3):
        rgb[i] = (rgb[i] - rgbmin[i]) * 1 / (rgbhigh[i] - rgbmin[i])
        rgb[i].clip(0, 1, out=rgb[i])

    # Gamma scale
    red, green, blue = [(channel ** gamma) * 255 for (channel, gamma) in zip(rgb, rgb_gamma)]

    # V5 30% more yellow
    nred = 1.1*(red +0.4*green - btf*blue)
    ngreen = 1.1*(0.9*green + 0.1*red)
    nblue = blue + btf*green


    # Stack channels
    rgb_stack = np.stack((nred, ngreen, nblue), axis=-1)
    rgb_stack.clip(0, 255, out=rgb_stack)

    return rgb_stack



def process_lab_32bit(bgr, lf=1, af=1, bf=1, Lmin=0):

    # 32 bits needs to be scaled withi [0-1]
    bgr2 = (bgr - bgr.min()) * 1 / (bgr.max() - bgr.min())
    lab = cv2.cvtColor(bgr2, cv2.COLOR_BGR2Lab)
    L, a, b = [lab[:, :, i] for i in range(3)]
    # In 32 bits, L ranges within [0 - 100]. In 8 bit: [0 255]
    # Convert to 8 bit range
    L *= 255/100
    a += 128
    b += 128

    L = (L - Lmin) * lf * 255 / (255 - Lmin)

    L.clip(0, 255, out=L)
    a *= af
    a.clip(0, 255, out=a)
    b *= bf
    b.clip(0, 255, out=b)

    lab = np.stack([L, a, b], axis=-1)

    return lab



def process_lab_8bit(bgr, lf=1, af=1, bf=1, Lmin=0):

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L, a, b = [lab[:, :, i] for i in range(3)]

    L = (L - Lmin) * lf * 255 / (255 - Lmin)

    L.clip(0, 255, out=L)
    a *= af
    a.clip(0, 255, out=a)
    b *= bf
    b.clip(0, 255, out=b)

    lab = np.stack([L, a, b], axis=-1)

    return lab



# Data test directories for the 3 wavelengths
data_files = [[glob.glob('../aia_data/*304*.fits')[0]],
              [glob.glob('../aia_data/*171*.fits')[0]],
              [glob.glob('../aia_data/*193*.fits')[0]]]

# Ref
# High intensity percentiles for thresholding
# Ref
#percentile_high = [99.85, 99.85, 99.85]
percentile_high = [99.5, 99.99, 99.85]
# Intensity gamma scaling factors for tone-mapping the 3x12 bit high dynamic range into the 3x8 bit range
# gamma_rgb=[2.8, 2.8, 2.4]

#percentile_high = [99.8, 99.8, 99.8]
percentile_low = [25, 25, 25]

gamma_rgb=[2.8, 2.8, 2.4]
# Blue tone factor: tune the "hot" vs "cold" look of the sun. The greater the value, the colder the sun will look
btf = 0.2

# output directory and filename for the jpeg images. These filename will be appended with the image number:
outputdir = '../aia_data/rgb_lab/'
filename_rgb = 'im_rgb_pmin_%.1f_%.1f_%.1f_pmax_%.2f_%.2f_%.2f_g_%.1f_%.1f_%.1f_btf%.1f'\
               %(*percentile_low, *percentile_high, *gamma_rgb, btf)

file_rgb = os.path.join(outputdir, filename_rgb + '.jpeg')
file_rgb_mod = os.path.join(outputdir, filename_rgb + '_mod.jpeg')

pdatargb = [aiaprep(data_files[j][0], cropsize=4096) for j in range(3)]

#pdatargb = [pdatargb[j][:, 0:2048] for j in range(3)]

# Get min and max normalization values at the 2 input percentiles
rgblow = np.array([np.percentile(pdatargb[j], percentile_low[j]) for j in range(3)])
rgbhigh = np.array([np.percentile(pdatargb[j], percentile_high[j]) for j in range(3)])
scaled_rgb = scale_rgb(pdatargb.copy(), rgblow, rgbhigh, gamma_rgb=gamma_rgb, btf=btf)
# OpenCV orders channels as B,G,R instead of R,G,B, and flip upside down.
bgr = np.flipud(np.flip(scaled_rgb, axis=2))
bgrc = np.clip(bgr, 0, 255)
cv2.imwrite(file_rgb, bgrc.astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 85])


# Make it 30% brighter
# dmax = 255
# dmin = 0
# bgr2 = bgr.copy().astype(np.float)*1.3
# bgr2 = (bgr2 - dmin) * (255 - 0) / (dmax - dmin)
# bgr2.clip(0, 255, out=bgr2)
# file_rgb = os.path.join(outputdir, filename_rgb + '_C.jpeg')
# cv2.imwrite(file_rgb, bgr2.astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 85])

# 20% yellow. Scale rgb skewed toward yellow (for comparison with Lab method)
# scaled_rgb_mod = scale_rgb_mod(pdatargb.copy(), rgblow, rgbhigh, gamma_rgb=gamma_rgb, btf=btf)
# bgr_mod = np.flipud(np.flip(scaled_rgb_mod, axis=2)).astype(np.uint8)
# cv2.imwrite(file_rgb_mod, bgr_mod, [int(cv2.IMWRITE_JPEG_QUALITY), 85])


# # # Convert rgb to CIELab
lf = 1.1
af = 1.05
bf = 1.10
Lmin = 20

labparams = [lf, af, bf]
#filename_lab = 'im_lab_pmin_%.1f_%.1f_%.1f_pmax_%.2f_%.2f_%.2f_g_%.1f_%.1f_%.1f_lab_%.2f_%.2f_%.2f'%(*percentile_low, *percentile_high, *gamma_rgb, *labparams)
filename_lab = 'im_lab_%.2f_%.2f_%.2f_Lmin%d'%(lf, af, bf, Lmin)

lab32 = process_lab_32bit(bgr, lf=lf, af=af, bf=bf, Lmin=Lmin)
bgr2 = cv2.cvtColor(lab32.astype(np.uint8), cv2.COLOR_Lab2BGR)
file_lab = os.path.join(outputdir, filename_lab + '_32bit.jpeg')
cv2.imwrite(file_lab, bgr2.astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 85])

lab8 = process_lab_8bit(bgrc.astype(np.uint8), lf=lf, af=af, bf=bf, Lmin=Lmin)
bgr2 = cv2.cvtColor(lab8.astype(np.uint8), cv2.COLOR_Lab2BGR)
file_lab = os.path.join(outputdir, filename_lab + '_8bit.jpeg')
cv2.imwrite(file_lab, bgr2.astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), 85])

