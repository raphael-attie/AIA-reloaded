import os
import numpy as np
import cv2
import calibration as calibration


def compute_intensity_high(image, cutoff_high):
    """
    Compute the image percentile-based high intensity threshold up to which we stretch (rescale) the intensity.

    :param image: input image
    :param cutoff_high: percentile. Typical values within [99.97 - 99.99]
    :return: maximum thresholding intensity
    """
    nbins = 256
    im_hist, bin_edges = np.histogram(image, nbins)
    hist_width = bin_edges[1] - bin_edges[0]
    intensity_high = calc_threshold(cutoff_high, im_hist, hist_width)

    return intensity_high



def calc_threshold(cutoff, im_hist, hist_width):
    """
    Calculate the intensity at the input cutoff-percentage of total pixels given an image histogram and bin width.
    This can be used to calculate, for example, the min or max threshold for rescaling the image intensity

    :param cutoff: cutoff percentage of total pixels in the histogram
    :param im_hist: image histogram
    :param hist_width: bin width of the image histogram
    :return: intensity value
    """
    npixels = im_hist.sum()
    nbins = im_hist.shape[0]

    cdf = 0.0
    i = 0
    count = 0
    while i < nbins:
        cdf += im_hist[i]
        if 100.0 * cdf / npixels > cutoff:
            count = i
            break
        i += 1

    if i == nbins:
        count = nbins - 1

    intensity = hist_width * count

    return intensity


def process_rgb_image(i, data_files, rgbhigh, outputdir = None):
    """
    Create an rgb image out of three fits files at different wavelengths, conveniently scaled for visualization.

    :param i: image index in the list of file
    :param data_files: list of RGB files. The list is 2D: [rgb channels][image index]
    :param rgbhigh: maximum intensity value(s) for rescaling. scalar or 3-element numpy array.
    :param outputdir: path to output directory for printing the rgb jpeg image
    :return: rgb image as a 3-channel numpy array: [image rows, image cols, 3]
    """

    # Prep aia data and export the r,g,b arrays into a list of numpy arrays.
    #pdatargb = [aiaprep(data_files[j][i]) for j in range(3)]
    pdatargb = [calibration.aiaprep(data_files[j][i], cropsize=4096) for j in range(3)]

    # Get the rgb stack and normalize to 255.
    im_rgb = np.stack(pdatargb, axis=-1) / rgbhigh
    im_rgb.clip(0, 1, out=im_rgb)

    g_r = 2.6
    g_g = 2.8
    g_b = 2.4
    rgb_gamma = 1 / np.array([g_r, g_g, g_b])
    im_rgb255 = (im_rgb ** rgb_gamma) * 255

    #im_rgb255[:,:,0] = im_rgb255[:,:,0] + 0.5 * im_rgb255[:,:,1]  #V1
    im_rgb255[:, :, 0] = im_rgb255[:, :, 0] + 0.7 * im_rgb255[:, :, 1]- 0.4 * im_rgb255[:,:,2] #V2
    im_rgb255[:, :, 1] = im_rgb255[:, :, 1] + 0.15 * im_rgb255[:, :, 0]
    im_rgb255[:, :, 2] = im_rgb255[:, :, 2] + 0.1 * im_rgb255[:, :, 1]

    newMin = np.array([35, 35, 35])
    im_rgb255 = (im_rgb255 - newMin) * 255 / (255 - newMin)
    im_rgb255.clip(0, 255, out=im_rgb255)

    im_rgb255 = np.flipud(im_rgb255.astype(np.uint8))
    rgb_stack = np.stack([im_rgb255[:, :, 2], im_rgb255[:, :, 1], im_rgb255[:, :, 0]], axis=-1)

    if outputdir is not None:
        outputfile = os.path.join(outputdir,
                                 'im_rgb_gamma_%0.1f_%0.1f_%0.1f_%03d.jpeg' % (g_r, g_g, g_b, i))
        cv2.imwrite(outputfile, rgb_stack, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        return rgb_stack, outputfile

    return rgb_stack
