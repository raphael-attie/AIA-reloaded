import os, glob
import numpy as np
import cv2
import calibration
import subprocess
import visualization
import functools
from multiprocessing import Pool

cv2.setNumThreads(2)

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


def get_rgb_high(data_files, percentiles=[99.5, 99.99, 99.85]):
    """
    Calculate the maximum intensity values for each channel to use for rescaling.
    :param data_files: images files listed in 3 image directories, 1 per wavelength.
    :param percentiles: define the high intensity threshold for each wavelength
    :return: high intensity values for each wavelength
    """
    # Get the high percentiles for rescaling the intensity for all images
    pdatargb0 = [calibration.aiaprep(data_files[j][0], cropsize=4096) for j in range(3)]
    # Get max percentile values for each channel.
    rgbhigh = np.array([visualization.compute_intensity_high(pdatargb0[i], percentiles[i]) for i in range(3)])

    return rgbhigh


def process_rgb_image(i, data_files, rgbhigh, gamma_rgb=[2.8, 2.8, 2.4], btf=0.3, outputdir = None):
    """
    Create an rgb image out of three fits files at different wavelengths, conveniently scaled for visualization.

    :param i: image index in the list of file
    :param data_files: list of RGB files. The list is 2D: [rgb channels][image index]
    :param rgbhigh: maximum intensity value(s) for rescaling. scalar or 3-element numpy array.
    :param gamma_rgb: gamma scaling factor for tone-mapping each channel from 3x12 bit hdr intensity to 3x8 bit
    :param btf: a "blue tone factor" to tune the balance between a "hot" and a "cold" looking star (the greater btf, the colder)
    :param outputdir: path to output directory for printing the rgb jpeg image
    :return: rgb image as a 3-channel numpy array: [image rows, image cols, 3]
    """

    # Prep aia data and export the r,g,b arrays into a list of numpy arrays.
    #pdatargb = [aiaprep(data_files[j][i]) for j in range(3)]
    pdatargb = [calibration.aiaprep(data_files[j][i], cropsize=4096) for j in range(3)]

    # Get the rgb stack and normalize to 255.
    im_rgb = np.stack(pdatargb, axis=-1) / rgbhigh
    im_rgb.clip(0, 1, out=im_rgb)

    g_r = gamma_rgb[0]
    g_g = gamma_rgb[1]
    g_b = gamma_rgb[2]
    rgb_gamma = 1 / np.array([g_r, g_g, g_b])
    im_rgb255 = (im_rgb ** rgb_gamma) * 255

    im_rgb255[:, :, 0] = im_rgb255[:, :, 0] + 0.7 * im_rgb255[:, :, 1] - btf * im_rgb255[:, :, 2]
    im_rgb255[:, :, 1] = im_rgb255[:, :, 1] + 0.15 * im_rgb255[:, :, 0]
    im_rgb255[:, :, 2] = im_rgb255[:, :, 2] + 0.1 * im_rgb255[:, :, 1]

    newMin = np.array([35, 35, 35])
    im_rgb255 = (im_rgb255 - newMin) * 255 / (255 - newMin)
    im_rgb255.clip(0, 255, out=im_rgb255)

    im_rgb255 = np.flipud(im_rgb255.astype(np.uint8))
    # OpenCV orders channels as B,G,R instead of R,G,B
    bgr_stack = np.stack([im_rgb255[:, :, 2], im_rgb255[:, :, 1], im_rgb255[:, :, 0]], axis=-1)

    if outputdir is not None:
        outputfile = os.path.join(outputdir,
                                 'im_rgb_gamma_%0.1f_%0.1f_%0.1f_btf_%0.1f_%03d.jpeg' % (g_r, g_g, g_b, btf, i))
        cv2.imwrite(outputfile, bgr_stack, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        return bgr_stack, outputfile

    return bgr_stack


def encode_video(images_dir, movie_filename, image_format='jpeg', fps=30, file_ext='.mp4', crop=None, video_size=None, image_pattern_search=None):
    """
    Create a movie from jpeg images. Input images will be found based on the image directory and a pattern search.

    :param images_dir: path to directory where images will be searched based on a pattern search. (default is *.jpeg).
    :param movie_filename: output file without extension (e.g: /path/to/my_movie and not /path/to/my_movie.mp4).
    It can be an absolute path or a path relative to the images_dir.
    :param fps: number of frames per second to display
    :param file_ext: suffix defining the file format and appended to movie_filename. Default is '.mp4'.
    :param crop: (width, height, x, y) crop the input images frop top left (x, y) coordinates over width and height pixels
    :param video_size: video dimensions (width, height) in pixels. Default to image size.
    :param image_format: format of the input images. Default is jpeg.
    :param image_pattern_search: pattern to look for input images: e.g. "my_images_aia_blah_*.jpeg".
    Default is to use "*.image_format". E.g if image_format ='jpeg', will look for images in images_dir/*.jpeg
    :return: None.
    """

    # Check valid file suffix
    if not file_ext.startswith('.') and not movie_filename.endswith('.'):
        file_ext = '.'+file_ext
    filename = movie_filename + file_ext

    if image_pattern_search is None:
        image_pattern_search = "*.%s"%image_format

    # Get default output video size
    if video_size is None:
        video_size = cv2.imread(glob.glob(os.path.join(images_dir, '*.%s')%image_format)[0]).shape[0:2][::-1]
    if crop is None:
        scale_crop_command = "scale=%d:%d"%video_size
    else:
        if video_size is None:
            video_size = (crop[0], crop[1])
        scale_crop_command = "crop=%d:%d:%d:%d,scale=%d:%d" % (*crop, *video_size)


    command = ["ffmpeg",
               "-framerate", "%d" % fps,
               "-pattern_type", "glob",
               "-i", "%s"%image_pattern_search,
               "-c:v", "libx264",
               "-preset", "veryslow",
               "-crf", "10",
               "-r", "30",
               "-vf", scale_crop_command,
               "-pix_fmt", "yuv420p",
               filename,
               "-y"]

    try:
        _ = subprocess.check_call(command, cwd=images_dir)
        print('Movie file written at: %s'%filename)
    except subprocess.CalledProcessError:
        print('Movie creation failed')

    return None


