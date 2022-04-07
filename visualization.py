import os, glob
import numpy as np
from astropy.io import fits
import cv2
import calibration
import subprocess
from calibration import aiaprep
import multiprocessing

#  disable multithreading in opencv. Default is to use all available, which is rather inefficient in this context
cv2.setNumThreads(0)


class RGBMixer:
    """ Automate the creation of AIA rgb images.
    The intensity in each channel is scaled to exploit the high dynamic range of the images using default tone-mapping values.
    Next, ffmpeg is called to create a movie from the jpeg files.
    """

    def __init__(self, data_dir=None, wavel_dirs=None, data_files=None, calibrate=True, outputdir=None, ref=0, crop=None,
                 filename_lab=None, filename_rgb=None):
        """

        :param data_dir: Parent directory of the 3 subdirectories.
        Either data_dir AND wavel_dirs, or data_files must be present.
        :param wavel_dirs: sequence of 3 sub-directory relative path in the order of (r,g,b) components (e.g. 3 aia wavelengths)
        the paths must be relative to data_dir.
        :param data_files: list of RGB files. data_files[rgb channel][image index]. data_dir or data_files must be present.
        :param calibrate: set to True if the aia fits files need to be calitrated.
        :param ref: index of the reference image used to get the intensity scaling values
        :param outputdir: directory for the output rgb images
        :param crop: (x,y)=(cols, rows) coordinates for cropping. E.g (slice(0,1024), slice(100,3200))
        :param filename_rgb: basename for the jpeg images if lab space unused, appended with the image number
        :param filename_lab: basename for lab-space-modified images, appended with the image number.
        """

        self.data_dir = data_dir

        if data_files is None:
            if data_dir is not None and wavel_dirs is None:
                raise ValueError('Missing input directory or files. Need either data_dir AND wavel_dirs, OR data_files')

        if wavel_dirs is not None:
            if data_dir is None:
                self.wavel_dirs = wavel_dirs
            else:
                self.wavel_dirs =  [os.path.join(data_dir, wavel) for wavel in wavel_dirs]

        if os.path.isdir(outputdir):
            self.outputdir = outputdir
        else:
            raise ValueError('output directory does not exist')

        if data_files is not None:
            self.data_files=data_files
        else:
            if os.path.isdir(self.wavel_dirs[0]) and os.path.isdir(self.wavel_dirs[1]) and os.path.isdir(self.wavel_dirs[2]):
                self.data_files = [glob.glob(os.path.join(ddir, '*.fits')) for ddir in self.wavel_dirs]
            else:
                raise ValueError('wavelength directories do not exist')

        self.calibrate = calibrate
        self.ref_rgb_files = [files[ref] for files in self.data_files]
        self.crop = crop
        # Reference image index to extract scaling values
        self.ref = ref
        # minimum and maximum rescaling values of each channel before gamma scaling
        self.rgblow = None
        self.rgbhigh = None
        self.scalemin = 0
        # Path and file naming scheme of the output images. This will be appended with the image number
        # for images processed in rgb space
        self.filename_rgb = filename_rgb
        self.filepath_rgb = None
        # for images processed  in lab space
        self.filename_lab = filename_lab
        self.filepath_lab = None
        # Intensity percentiles for linear scaling
        self.percentiles_low = (0, 0, 0)
        self.percentiles_high = (99.99, 99.99, 99.99)

        # Intensity gamma scaling factors for tone-mapping the 3x12 bit high dynamic range into the 3x8 bit range
        self.gamma_rgb = (1, 1, 1)
        self.rgbmix = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])
        # [Optional processing] CIELab parameters. Make 10% brighter, skew the chromaticity toward red and yellow
        self.lab = None
        # For contrast stretching on luminance (L) layer. L ranges in [0-255]
        self.lmin = 0
        # Reference rgb image used for the intensity scaling values
        self.ref_rgb = None

    def set_aia_default(self):

        # Load and prep reference image
        self.ref_rgb = [aiaprep(fitsfile) for fitsfile in self.ref_rgb_files]

        self.percentiles_low = (25, 25, 25)
        self.percentiles_high = (99.5, 99.99, 99.85)

        self.rgbmix= np.array([[1.0, 0.6, -0.3],
                               [0.0, 1.0, 0.1],
                               [0.0, 0.1, 1.0]])

        self.gamma_rgb = (2.8, 2.8, 2.4)
        self.scalemin = 20
        self.lab = (1, 0.96, 1.04)
        self.lmin = 0
        self.filename_lab = 'im_lab'
        self.set_ref_low_high()

    def set_ref_low_high(self, plow=None, phigh=None):

        if plow is None:
            plow = self.percentiles_low
        if phigh is None:
            phigh = self.percentiles_high

        self.rgblow = np.array([np.percentile(self.ref_rgb[j], plow[j]) for j in range(3)])
        self.rgbhigh = np.array([np.percentile(self.ref_rgb[j], phigh[j]) for j in range(3)])

    def process_rgb(self, image_index):
        """Setup which image version to output. Can be either just rgb, just lab, or both"""

        bgr_stack1, bgr_stack2 = process_rgb_image(image_index, data_files=self.data_files,
                                                   rgblow=self.rgblow, rgbhigh=self.rgbhigh, scalemin=self.scalemin,
                                                   gamma_rgb=self.gamma_rgb,
                                                   rgbmix=self.rgbmix,
                                                   lab=self.lab,
                                                   lmin=self.lmin,
                                                   crop=self.crop,
                                                   filename_rgb=self.filepath_rgb, filename_lab=self.filepath_lab)
        return bgr_stack1, bgr_stack2


    def process_rgb_list(self, ncores, file_range):

        if ncores > 1:
            multiprocessing.set_start_method('spawn')
            p = multiprocessing.Pool(ncores)
            p.map(self.process_rgb, file_range)
            p.close()
        else:
            for i in file_range:
                _ = self.process_rgb(i)



def rgb_high_low(rgb_files, percentiles_low, percentiles_high):
    """ Convenience function to get the minimum and maximum rescaling values of each channel before gamma scaling.

    :param rgb_files: list of 3 files. 1 per channel
    :param percentiles_low: list of percentiles for the minimum scaling value, in order of [red, green, blue]
    :param percentiles_high: list of percentiles for the maximum scaling value, in order of [red, green, blue]
    :return: 2 lists of 3 minimum and maximum intensity. one per channel in each list.
    """
    pdatargb = [aiaprep(rgb_files[j]) for j in range(3)]
    rgblow = np.array([np.percentile(pdatargb[j], percentiles_low[j]) for j in range(3)])
    rgbhigh = np.array([np.percentile(pdatargb[j], percentiles_high[j]) for j in range(3)])
    return rgblow, rgbhigh


def scale_rgb(rgb, rgblow, rgbhigh, gamma_rgb=(2.8, 2.8, 2.4), rgbmix=None, scalemin=0):
    """ Rescale the rgb image series.
    First linearly rescales between minimum and maximum values independently on each channel.
    Apply gamma-scaling on the [0-1]-normalized channels.
    Optionnally restretch contrast linearly. Often necessary e.g. if you do not tune this in CIELab-space.

    :param rgb: list of 3 series of aia prepped images for the red, green, blue channel.
    :param rgblow: minimum rescaling values before gamma scaling. 1-per channel
    :param rgbhigh: maximum rescaling values before gamma scaling
    :param gamma_rgb: gamma scaling factors for the normalized image. The inverse is the actual exponent.
    This is meant to be a single value and not a channel-dependent parameter.
    :param rgbmix: rgb mixing matrix in [red, green, blue] order in both dimensions.
    :param scalemin: minimum value for optional contrast stretching after gamma-scaling.
    :return: rescaled rgb image as numpy 3D array: [height, width, rgb channels]
    """

    rgb_gamma = 1 / np.array(gamma_rgb)

    # Copy is needed due to in-place operations
    rgb2 = rgb.copy()
    for i in range(3):
        rgb2[i] = (rgb[i] - rgblow[i]) * 1 / (rgbhigh[i] - rgblow[i])
        rgb2[i].clip(0, 1, out=rgb2[i])

    red, green, blue = [(channel ** gamma) for (channel, gamma) in zip(rgb2, rgb_gamma)]

    # Apply color mixing
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


def process_lab_32bit(bgr, lf=1, af=1, bf=1, lmin=0):
    """
    Process the color balancing in CIELab space. Due to the format needed by the library used (openCV), the order of the
    channels must be ordered as blue, green, red (red and blue swapped).
    Multipliers are applied to the l, a, and b axes.

    :param bgr: rescaled images as a numpy array [height, width, channels] with channels in order blue, green, red
    :param lf: luminance modifier within. Must be >0. E.g 1.5 will make the image 50% brighter in the CIELab sense.
    :param af: green-red axis modifier (>0).
    :param bf: blue-yellow axis modifier (>0).
    :param lmin: Minimum value for the contrast stretching in the luminance dimension. Luminance range = [0-255]
    :return: Numpy array of the rescaled "bgr" image. For visualization in Matplotlib, must swap again blue <-> red
    """
    # 32 bits needs to be scaled withi [0-1]
    bgr2 = (bgr - bgr.min()) * 1 / (bgr.max() - bgr.min())
    lab = cv2.cvtColor(bgr2, cv2.COLOR_BGR2Lab)
    L, a, b = [lab[:, :, i] for i in range(3)]
    # In 32 bits, L ranges within [0 - 100]. In 8 bit: [0 255]
    # Convert to 8 bit range
    L *= 255/100
    a += 128
    b += 128

    L = (L - lmin) * lf * 255 / (255 - lmin)
    a *= af
    b *= bf
    lab = np.stack([L, a, b], axis=-1)
    lab.clip(0, 255, out=lab)

    return lab


def process_rgb_image(i, data_files, rgblow, rgbhigh, calibrate=True, gamma_rgb=(2.8, 2.8, 2.4), scalemin=0, rgbmix=None, lab=None, lmin=0, crop=None, filename_rgb=None, filename_lab=None):
    """
    Create an rgb image out of three fits files at different wavelengths, conveniently scaled for visualization.

    :param i: image index in the list of file
    :param data_files: list of RGB files. The list is 2D: [rgb channels][image index]
    :param rgblow: minimum intensity value(s) for rescaling. scalar or 3-element numpy array.
    :param rgbhigh: maximum intensity value(s) for rescaling. scalar or 3-element numpy array.
    :param calibrate: True if you need to calibrate from raw fits files.
    :param gamma_rgb: gamma scaling factor for tone-mapping each channel from 3x12 bit hdr intensity to 3x8 bit
    :param scalemin: minimum intensity for contrast stretching
    :param rgbmix: 3x3 array of mixing parameters of the rgb channels.
    :param lab: CIELab parameters
    :param lmin: optionnaly used with lab. Minimum luminance for rescaling into the 8 bit range.
    :param crop: tuple of slices of (x,y) zero-based coordinates for cropping. E.g (slice(100,1300), slice(0,1000))
    :param filename_rgb: basename for the jpeg images if lab space unused, appended with the image number
    :param filename_lab: basename for lab-space-modified images, appended with the image number.
    :return: rgb image as a 3-channel numpy array: [image rows, image cols, 3]
    """

    bgr_stack2 = None

    # Prep aia data and export the r,g,b arrays into a list of numpy arrays.
    if calibrate:
        pdatargb = [calibration.aiaprep(data_files[j][i]) for j in range(3)]
    else:
        pdatargb = [load_fits(data_files[j][i]) for j in range(3)]

    # Apply hdr tone-mapping
    im_rgb255 = scale_rgb(pdatargb, rgblow, rgbhigh, gamma_rgb=gamma_rgb, scalemin=scalemin, rgbmix=rgbmix)

    # OpenCV orders channels as B,G,R instead of R,G,B, and flip upside down.
    bgr_stack = np.flipud(np.flip(im_rgb255, axis=2))

    bgr_stack1 = np.clip(bgr_stack, 0, 255)
    bgr_stack1 = bgr_stack1.astype(np.uint8)
    if crop is not None:
        bgr_stack1 = bgr_stack1[crop[::-1]]

    if filename_rgb is not None:
        outputfile_rgb = filename_rgb + '_%04d.jpeg'%i
        cv2.imwrite(outputfile_rgb, bgr_stack1, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    if lab is not None:
        lab32 = process_lab_32bit(bgr_stack, lf=lab[0], af=lab[1], bf=lab[2], lmin=lmin)
        bgr_stack2 = cv2.cvtColor(lab32.astype(np.uint8), cv2.COLOR_Lab2BGR)
        if crop is not None:
            bgr_stack2 = bgr_stack2[crop[::-1]]
        if filename_lab is not None:
            outputfile_lab = filename_lab + '_%04d.jpeg'%i
            cv2.imwrite(outputfile_lab, bgr_stack2, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return bgr_stack1, bgr_stack2


def encode_video(images_dir, movie_filename, image_format='jpeg', fps=30, file_ext='.mp4', crop=None, frame_size=None, padded_size=None, image_pattern_search=None, command_only=False):
    """
    Run ffmpeg to create a movie from jpeg images. Input images will be found based on the image directory and a pattern search.
    If you're writing images with your own methods, use padded numbering: 001, 002, ..., 010 instead of 1,2,...10
    or else you'll need to write your own parser for inputing the list of input image the right order for ffmpeg.


    :param images_dir: path to directory where images will be searched based on a pattern search. (default is *.jpeg).
    :param movie_filename: output file without extension (e.g: /path/to/my_movie and not /path/to/my_movie.mp4).
    It can be an absolute path or a path relative to the images_dir.
    :param fps: number of frames per second to display
    :param file_ext: suffix defining the file format and appended to movie_filename. Default is '.mp4'.
    :param crop: (width, height, x, y) crop the input images frop top left (x, y) coordinates over width and height pixels
    :param frame_size: (width, height) without padding, actual video size. Default to image size.
    With padding (see padded_size bellow), this will be the size of the image within the padded frame.
    :param padded_size: (width, height) actual output video size with horizontal and vertical padding.
    If used, the frame_size must be able to fit in the padded_size.
    :param image_format: format of the input images. Default is jpeg.
    :param image_pattern_search: pattern string to look for input images:
    e.g. "my_images_aia_blah_*.jpeg" if using padded numbers. or "my_images_aia_blah_%d.jpeg" if not
    Default is to use "*.image_format". E.g if image_format ='jpeg', will look for images in images_dir/*.jpeg
    :param command_only: set to True if you only want to get the command line string that gets executed.
    If you use this in the terminal, you need to add single or double quotes around the image name pattern
    :return: Command-line string called by subprocess.
    """

    video_filter = None

    # Check valid file suffix
    if not file_ext.startswith('.') and not movie_filename.endswith('.'):
        file_ext = '.'+file_ext
    filename = movie_filename + file_ext

    if image_pattern_search is None:
        image_pattern_search = "*.%s"%image_format

    # cropping must be given in input coordinate frame.
    if crop is not None:
        video_filter = "crop=%d:%d:%d:%d" %(crop[0], crop[1], crop[2], crop[3])

    # Only instruct ffmpeg to rescale if frame_size is explicitly given.
    if frame_size is not None:
        if video_filter is not None: # append comma first.
            video_filter += ",scale=%d:%d" % frame_size
        else:
            video_filter = "scale=%d:%d" % frame_size
    else:
        frame_size = cv2.imread(glob.glob(os.path.join(images_dir, '*.%s') % image_format)[0]).shape[0:2][::-1]

    # Padding happens last before color adjustments
    if padded_size is not None:
        x = int((padded_size[0] - frame_size[0])/2)
        y = int((padded_size[1] - frame_size[1])/2)
        if video_filter is not None:
            video_filter += ",pad=%d:%d:%d:%d" % (*padded_size, x, y)
        else:
            video_filter = "pad=%d:%d:%d:%d" % (*padded_size, x, y)


    # No matter what, we need at least to boost contrast to 1.1 because of codecs effect of washing out colors
    if video_filter is None:
        video_filter = 'eq=contrast=1.1'
    else:
        video_filter += ',eq=contrast=1.1'

    command = ["ffmpeg",
               "-framerate", "%d" % fps,
               "-pattern_type", "glob",
               "-i", image_pattern_search,
               "-c:v", "libx264",
               "-preset", "slow",
               "-crf", "18",
               "-r", "30",
               "-vf", video_filter,
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


def load_fits(fitsfile):
    """
    This is only used if working with aia fits files already calibrated. Because the headers aren't needed in this case,
    this just loads the data from the HDU. This tests first if the fits file at hand is single-hdu (primary-only) or
    primary hdu with an image extension.

    :param fitsfile: path to fits file
    :return:
    """
    try:
        hdul = fits.open(fitsfile)
    except FileNotFoundError:
        print("Could not open fits file")
    else:
        if len(hdul) == 1:
            data = hdul[0].data.astype(np.float64)
        else:
            hdul[1].verify('silentfix')
            data = hdul[1].data.astype(np.float64)

        hdul.close()
        return data