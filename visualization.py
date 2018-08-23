import os, glob
import numpy as np
import cv2
import calibration
import subprocess
from calibration import aiaprep


def rgb_high_low(data_files, percentiles_low, percentiles_high):

    pdatargb = [aiaprep(data_files[j][0]) for j in range(3)]
    rgblow = np.array([np.percentile(pdatargb[j], percentiles_low[j]) for j in range(3)])
    rgbhigh = np.array([np.percentile(pdatargb[j], percentiles_high[j]) for j in range(3)])
    return rgblow, rgbhigh


def scale_rgb(rgb, rgblow, rgbhigh, gamma_rgb=[2.8, 2.8, 2.4], scalemin=0, rgbmix=None):


    rgb_gamma = 1 / np.array(gamma_rgb)

    # Copy is needed due to in-place operations
    rgb2 = rgb.copy()
    for i in range(3):
        rgb2[i] = (rgb[i] - rgblow[i]) * 1 / (rgbhigh[i] - rgblow[i])
        rgb2[i].clip(0, 1, out=rgb2[i])

    red, green, blue = [(channel ** gamma) for (channel, gamma) in zip(rgb2, rgb_gamma)]

    # nred = red + 0.4 * green - btf * blue  # ~ 320 ms
    # ngreen = green + 0.1*red  # ~ 180 ms
    # nblue = blue
    if rgbmix is not None:
        [[rr, rg, rb], [gr, gg, gb], [br, bg, bb]] = rgbmix
        nred = rr*red + rg*green + rb*blue  # ~ 320 ms
        ngreen = gr*red + gg*green + gb*blue  # ~ 180 ms
        nblue = br*red + bg*green + bb*blue
        rgb_stack = np.stack((nred, ngreen, nblue), axis=-1).astype(np.float32)
    else:
        rgb_stack = np.stack((red, green, blue), axis=-1).astype(np.float32)

    # stack and rescale channels for 8-bit range, convert from 64 to 32 bit float for now (needed for CIELab)
    rgb_stack.clip(0, 1, out=rgb_stack)
    rgb_stack *= 255
    # Contrast stretch in RGB space
    rgb_stack = (rgb_stack- scalemin) * 255 / (255 - scalemin)
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
    a *= af
    b *= bf
    lab = np.stack([L, a, b], axis=-1)
    lab.clip(0, 255, out=lab)

    return lab


def process_rgb_image(i, data_files, rgblow, rgbhigh, gamma_rgb=(2.8, 2.8, 2.4), scalemin=0, rgbmix=None, lab=None, lmin=0, crop=None, outputdir = None, filename_rgb='image_rgb', filename_lab='image_lab'):
    """
    Create an rgb image out of three fits files at different wavelengths, conveniently scaled for visualization.

    :param i: image index in the list of file
    :param data_files: list of RGB files. The list is 2D: [rgb channels][image index]
    :param rgblow: minimum intensity value(s) for rescaling. scalar or 3-element numpy array.
    :param rgbhigh: maximum intensity value(s) for rescaling. scalar or 3-element numpy array.
    :param gamma_rgb: gamma scaling factor for tone-mapping each channel from 3x12 bit hdr intensity to 3x8 bit
    :param scalemin: minimum intensity for contrast stretching
    :param rgbmix: 3x3 array of mixing parameters of the rgb channels.
    :param outputdir: path to rgb jpeg image
    :param lab: CIELab parameters
    :param lmin: optionnaly used with lab. Minimum luminance for rescaling into the 8 bit range.
    :param crop: tuple of slices of (y,x)=(rows, cols) coordinates for cropping. E.g (slice(0,1024), slice(100,3200))
    :param filename_rgb: common basename for the jpeg images if lab space unused. Appended with the image number
    :param filename_lab: used for lab-space-modified image. Appended with the image number.
    :return: rgb image as a 3-channel numpy array: [image rows, image cols, 3]
    """

    bgr_stack2 = None
    # Prep aia data and export the r,g,b arrays into a list of numpy arrays.
    #pdatargb = [aiaprep(data_files[j][i]) for j in range(3)]
    pdatargb = [calibration.aiaprep(data_files[j][i], cropsize=4096) for j in range(3)]
    # Apply hdr tone-mapping
    im_rgb255 = scale_rgb(pdatargb, rgblow, rgbhigh, gamma_rgb=gamma_rgb, scalemin=scalemin, rgbmix=rgbmix)

    # OpenCV orders channels as B,G,R instead of R,G,B, and flip upside down.
    bgr_stack = np.flipud(np.flip(im_rgb255, axis=2))

    bgr_stack1 = np.clip(bgr_stack, 0, 255)
    bgr_stack1 = bgr_stack1.astype(np.uint8)
    if crop is not None:
        bgr_stack1 = bgr_stack1[crop]

    if outputdir is not None and lab is None:
        outputfile_rgb = os.path.join(outputdir, filename_rgb + '_%03d.jpeg' %i)
        cv2.imwrite(outputfile_rgb, bgr_stack1, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    if lab is not None:
        lab32 = process_lab_32bit(bgr_stack, lf=lab[0], af=lab[1], bf=lab[2], Lmin=lmin)
        bgr_stack2 = cv2.cvtColor(lab32.astype(np.uint8), cv2.COLOR_Lab2BGR)
        if crop is not None:
            bgr_stack2 = bgr_stack2[crop]
        if outputdir is not None:
            outputfile_lab = os.path.join(outputdir, filename_lab + '_%03d.jpeg' % i)
            cv2.imwrite(outputfile_lab, bgr_stack2, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    return bgr_stack1, bgr_stack2


def encode_video(images_dir, movie_filename, image_format='jpeg', fps=30, file_ext='.mp4', crop=None, frame_size=None, padded_size=None, image_pattern_search=None, command_only=False):
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
               "-i", image_pattern_search,
               "-c:v", "libx264",
               "-preset", "slow",
               "-crf", "18",
               "-r", "30",
               "-vf", scale_crop_command,
               "-pix_fmt", "yuv420p",
               filename,
               "-y"]
    # Working example:
    # ffmpeg -framerate 30 -pattern_type glob -i 'im_rgb_*.jpeg' -c:v libx264 -preset slow -crf 18 -r 30 -vf crop=3840:2160:128:1935,scale=1920:1080 -pix_fmt yuv420p rgb_movie_3840x2160_1920x1080.mp4 -y
    if command_only:
        return subprocess.list2cmdline(command)

    try:
        _ = subprocess.check_call(command, cwd=images_dir)
        print('Movie file written at: %s'%filename)
    except subprocess.CalledProcessError:
        print('Movie creation failed')

    return subprocess.list2cmdline(command)


