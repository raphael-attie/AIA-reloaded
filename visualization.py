import os, glob
import numpy as np
import cv2
import calibration
import subprocess
import visualization

#cv2.setNumThreads(0)

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


def scale_rgb(prepped_rgb, rgbhigh, gamma_rgb=[2.8, 2.8, 2.4], btf=0.2):

    rgb_gamma = 1 / np.array(gamma_rgb)
    # Get the rgb stack and normalize to 255.

    # im_rgb = np.stack(prepped_rgb, axis=-1) / rgbhigh  # ~700 ms
    # im_rgb.clip(0, 1, out=im_rgb) # ~50 ms
    # im_rgb255 = (im_rgb ** rgb_gamma) * 255  # ~1.4 s

    for i in range(3):
        np.divide(prepped_rgb[i], rgbhigh[i], out=prepped_rgb[i])
        prepped_rgb[i].clip(0, 1, out=prepped_rgb[i])

    red, green, blue = [(channel ** gamma) * 255 for (channel, gamma) in zip(prepped_rgb, rgb_gamma)]

    nred = red + 0.6 * green - btf * blue  # ~ 320 ms
    ngreen = green + 0.1*red + 0.1 * blue  # ~ 180 ms
    nblue = blue + 0.1 * green  # ~ 180 ms

    # Reverse channel order for opencv
    rgb_stack = np.stack((nred, ngreen, nblue), axis=-1)

    newmin = np.array([35, 35, 35])
    rgb_stack = (rgb_stack - newmin) * 255 / (255 - newmin)
    rgb_stack.clip(0, 255, out=rgb_stack)

    return rgb_stack


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
    # Apply hdr tone-mapping
    im_rgb255 = scale_rgb(pdatargb, rgbhigh, gamma_rgb=gamma_rgb, btf=btf)
    # OpenCV orders channels as B,G,R instead of R,G,B, and flip upside down.
    bgr_stack = np.flipud(np.flip(im_rgb255, axis=2))

    if outputdir is not None:
        outputfile = os.path.join(outputdir,
                                 'im_rgb_gamma_%0.1f_%0.1f_%0.1f_btf_%0.1f_%03d.jpeg' % (*gamma_rgb, btf, i))
        cv2.imwrite(outputfile, bgr_stack, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        return bgr_stack, outputfile

    return bgr_stack


def encode_video(images_dir, movie_filename, image_format='jpeg', fps=30, file_ext='.mp4', crop=None, frame_size=None, padded_size=None, image_pattern_search=None):
    """
    Create a movie from jpeg images. Input images will be found based on the image directory and a pattern search.

    :param images_dir: path to directory where images will be searched based on a pattern search. (default is *.jpeg).
    :param movie_filename: output file without extension (e.g: /path/to/my_movie and not /path/to/my_movie.mp4).
    It can be an absolute path or a path relative to the images_dir.
    :param fps: number of frames per second to display
    :param file_ext: suffix defining the file format and appended to movie_filename. Default is '.mp4'.
    :param crop: (width, height, x, y) crop the input images frop top left (x, y) coordinates over width and height pixels
    :param frame_size: (width, height) without padding, actual video size. Default to image size.
    With padding (see padded_size bellow), this will be the output size of the image within the padded frame.
    :param padded_size: (width, height) actual output video size with horizontal and vertical padding.
    If used, the frame_size must be able to fit in the padded_size.
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
    if frame_size is None:
        frame_size = cv2.imread(glob.glob(os.path.join(images_dir, '*.%s')%image_format)[0]).shape[0:2][::-1]

    scale_crop_command = "scale=%d:%d" % frame_size

    if padded_size is not None: # frame_size must be smaller than the padding size
        x = int((padded_size[0] - frame_size[0]) / 2)
        y = int((padded_size[1] - frame_size[1]) / 2)
        scale_crop_command = "scale=%d:%d,pad=%d:%d:%d:%d" % (*frame_size, *padded_size, x, y)

    if crop is not None:
        if frame_size is None:
            frame_size = (crop[0], crop[1])
        if padded_size is None:
            scale_crop_command = "crop=%d:%d:%d:%d,scale=%d:%d" % (*crop, *frame_size)
        else:
            x = int((padded_size[0] - frame_size[0])/2)
            y = int((padded_size[1] - frame_size[1])/2)
            scale_crop_command = "crop=%d:%d:%d:%d,scale=%d:%d,pad=%d:%d:%d:%d" % (*crop, *frame_size, *padded_size, x, y)


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


