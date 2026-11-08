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

    def __init__(self, data_dir=None, wavel_dirs=None, data_files=None, calibrate=True, jp2=False, outputdir=None, ref=0, crop=None,
                 filename_lab=None, filename_rgb=None):
        """

        :param data_dir: Parent directory of the 3 subdirectories.
        Either data_dir AND wavel_dirs, or data_files must be present.
        :param wavel_dirs: sequence of 3 sub-directory relative path in the order of (r,g,b) components (e.g. 3 aia wavelengths)
        the paths must be relative to data_dir.
        :param data_files: list of RGB files. data_files[rgb channel][image index]. data_dir or data_files must be present.
        :param calibrate: set to True if the aia fits files need to be calitrated.
        :param jp2: set to True if input files are pre-calibrated JPEG2000 (.jp2) images.
        :param ref: index of the reference image used to get the intensity scaling values
        :param outputdir: directory for the output rgb images
        :param crop: (x,y)=(cols, rows) coordinates for cropping. E.g (slice(0,1024), slice(100,3200))
        :param filename_rgb: basename for the jpeg images if lab space unused, appended with the image number
        :param filename_lab: basename for lab-space-modified images, appended with the image number.
        """

        self.data_dir = data_dir
        self.jp2 = jp2

        if data_files is None and wavel_dirs is None:
            raise ValueError('Missing input directory or files. Need either data_dir AND wavel_dirs, OR data_files')

        if wavel_dirs is not None:
            if data_dir is None:
                self.wavel_dirs = wavel_dirs
            else:
                self.wavel_dirs = [os.path.join(data_dir, wavel) for wavel in wavel_dirs]

        if outputdir is not None and os.path.isdir(outputdir):
            self.outputdir = outputdir
        else:
            raise ValueError('output directory does not exist')

        if data_files is not None:
            self.data_files = data_files
        else:
            if os.path.isdir(self.wavel_dirs[0]) and os.path.isdir(self.wavel_dirs[1]) and os.path.isdir(self.wavel_dirs[2]):
                ext = '*.jp2' if self.jp2 else '*.fits'
                self.data_files = [sorted(glob.glob(os.path.join(ddir, ext))) for ddir in self.wavel_dirs]
            else:
                raise ValueError('wavelength directories do not exist')

        self.calibrate = False if self.jp2 else calibrate
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
        # for images processed  in lab space
        self.filename_lab = filename_lab
        # Intensity percentiles for linear scaling
        self.percentiles_low = (0, 0, 0)
        self.percentiles_high = (100, 100, 100)

        # Intensity gamma scaling factors for tone-mapping the 3x12 bit high dynamic range into the 3x8 bit range
        self.gamma_rgb = (1.0, 1.0, 1.0)
        # Sigmoidal contrast applied to each channel after gamma (0 = off), pivoting at contrast_mid
        # (a scalar or one pivot per channel). Restores the mid-tone contrast that gamma > 1 flattens,
        # without clipping black or white. Use mean_preserving_midpoint() for per-channel pivots that
        # keep each channel's mean level, so contrast does not also change brightness or color balance.
        self.contrast = 0.0
        self.contrast_mid = 0.5
        # Spatial enhancement applied to each channel after the tone curve (all off by default):
        # MGN "structure" weight (multi-scale Gaussian normalization, Morgan & Druckmueller 2014) and
        # Lightroom-style unsharp mask (amount, radius in full-res pixels, detail = halo allowance %,
        # masking = edge-mask percentile %). See apply_structure() and unsharp_mask().
        self.structure = 0.0
        self.sharpen = (0.0, 1.0, 100.0, 0.0)  # (amount, radius, detail, masking)
        self.sharpen_mask_thr = None  # optional absolute masking threshold per channel (masking_threshold)
        self.rgbmix = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])
        # [Optional processing] CIELab parameters. Make 10% brighter, skew the chromaticity toward red and yellow
        self.lab = (1.0, 1.0, 1.0)
        # For contrast stretching on luminance (L) layer. L ranges in [0-255]
        self.lmin = 0
        # Reference rgb image used for the intensity scaling values
        self.ref_rgb = None

    @property
    def filepath_rgb(self):
        if self.filename_rgb is not None and self.outputdir is not None:
            return os.path.join(self.outputdir, self.filename_rgb)
        return None

    @property
    def filepath_lab(self):
        if self.filename_lab is not None and self.outputdir is not None:
            return os.path.join(self.outputdir, self.filename_lab)
        return None

    def set_aia_default(self):
        if self.jp2:
            return self.set_jp2_default()

        # Load and prep reference image
        if self.calibrate:
            self.ref_rgb = [aiaprep(fitsfile) for fitsfile in self.ref_rgb_files]
        else:
            self.ref_rgb = [load_fits(fitsfile) for fitsfile in self.ref_rgb_files]

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

    def set_jp2_default(self):
        # Load pre-calibrated JPEG2000 reference images
        self.ref_rgb = [load_jp2(jp2file) for jp2file in self.ref_rgb_files]

        self.percentiles_low = (0, 0, 0)
        self.percentiles_high = (100, 100, 100)

        self.rgbmix = np.array([[1.045, 0.635, -0.295],
                                [0.0,   0.955,  0.095],
                                [0.0,   0.195,  0.565]])

        self.gamma_rgb = (1.0, 1.0, 1.0)
        self.scalemin = 0
        self.lab = (1.0, 1.0, 1.0)
        self.lmin = 0
        self.filename_lab = 'im_lab_jp2'
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
                                                   rgblow=self.rgblow, rgbhigh=self.rgbhigh,
                                                   calibrate=self.calibrate, jp2=self.jp2,
                                                   scalemin=self.scalemin,
                                                   gamma_rgb=self.gamma_rgb,
                                                   contrast=self.contrast,
                                                   contrast_mid=self.contrast_mid,
                                                   structure=self.structure,
                                                   sharpen=self.sharpen,
                                                   sharpen_mask_thr=self.sharpen_mask_thr,
                                                   rgbmix=self.rgbmix,
                                                   lab=self.lab,
                                                   lmin=self.lmin,
                                                   crop=self.crop,
                                                   filename_rgb=self.filepath_rgb, filename_lab=self.filepath_lab)
        return bgr_stack1, bgr_stack2


    def process_rgb_list(self, ncores, file_range):

        if ncores > 1:
            if multiprocessing.get_start_method(allow_none=True) != 'spawn':
                multiprocessing.set_start_method('spawn', force=True)
            with multiprocessing.Pool(ncores) as p:
                p.map(self.process_rgb, file_range)
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


def sigmoid_contrast(x, strength, midpoint=0.5):
    """
    Normalized sigmoidal contrast curve on [0, 1] data (as ImageMagick's -sigmoidal-contrast).
    Steepens the mid-tones around `midpoint` while keeping 0 -> 0 and 1 -> 1, so nothing clips.

    :param x: array with values in [0, 1]
    :param strength: slope of the sigmoid; 0 (or less) returns x unchanged, ~3 is mild, ~10 is strong.
    :param midpoint: pivot value in [0, 1] whose neighborhood gains the most contrast.
    :return: array of the same shape and dtype as x
    """
    if strength <= 1e-6:
        return x
    x = np.asarray(x)
    dtype = x.dtype.type if np.issubdtype(x.dtype, np.floating) else np.float64
    k, m = dtype(strength), dtype(midpoint)  # keep float32 inputs in float32 (speed)
    lo = 1.0 / (1.0 + np.exp(-strength * (0.0 - midpoint)))
    hi = 1.0 / (1.0 + np.exp(-strength * (1.0 - midpoint)))
    y = 1.0 / (1.0 + np.exp(-k * (x - m)))
    return np.clip((y - dtype(lo)) / dtype(hi - lo), 0.0, 1.0).astype(dtype, copy=False)


def mean_preserving_midpoint(x, strength, tol=1e-4):
    """
    Pivot m for sigmoid_contrast(x, strength, m) such that the mean of the output equals the mean of x.

    With one pivot per channel chosen this way, the contrast curve leaves every channel's mean level
    unchanged; since RGB mixing is linear, the mean of any mixed output is then unchanged too
    (before clipping), whatever the mixing matrix. A shared pivot instead brightens channels whose
    values sit mostly above it and darkens the others, so the overall brightness and color would
    depend on the mixing weights.

    :param x: sample of the channel's values in [0, 1] (e.g. on-disk pixels after gamma)
    :param strength: sigmoid strength as in sigmoid_contrast
    :return: pivot in [0, 1]
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if strength <= 1e-6 or x.size == 0:
        return 0.5
    target = float(x.mean())
    lo, hi = 0.0, 1.0  # the output mean decreases monotonically as the pivot moves right
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(sigmoid_contrast(x, strength, mid).mean()) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


MGN_SIGMAS = (1.25, 2.5, 5.0, 10.0, 20.0, 40.0)  # Gaussian scales in full-resolution pixels (Morgan & Druckmueller 2014)
MGN_K = 0.7


MGN_BLUR_MIN_SIGMA = 4.0  # wide blurs run on an image downsampled so that sigma stays >= this many pixels


def _gaussian_local_mean(img, sigma):
    """Gaussian blur of a float32 image. Wide kernels (sigma >= 2 * MGN_BLUR_MIN_SIGMA) are computed on an
    area-downsampled copy (factor 2^n, at most 8) and bilinearly upsampled: the blur suppresses the detail the
    resampling loses, so the result matches the full-resolution blur closely at a fraction of the cost
    (direct sigma = 40 px blurs on 4096^2 images dominate MGN otherwise)."""
    f = 1
    while f < 8 and sigma / (2 * f) >= MGN_BLUR_MIN_SIGMA:
        f *= 2
    if f == 1:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    h, w = img.shape
    small = cv2.resize(img, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def mgn_detail(x, scale=1.0, sigmas=MGN_SIGMAS, k=MGN_K):
    """
    Local component of Multi-scale Gaussian Normalization (Morgan & Druckmueller 2014, Solar Phys. 289, 2945).

    For each Gaussian scale sigma: C = arctan(k * (x - <x>) / sqrt(<(x - <x>)^2>)), with <.> a Gaussian-weighted
    local mean; the result is the mean over scales, divided by pi/2 so it lies in [-1, 1] (zero-mean).
    Flat regions (e.g. zero-valued corners) give 0.

    :param x: 2D float32/float64 image in [0, 1]
    :param scale: image pixels per full-resolution pixel (e.g. 0.25 for a 1024 px view of a 4096 px image), so
        the scales cover the same physical sizes in previews and in full-resolution output. Scales smaller than
        half an image pixel are skipped (they carry no information at that sampling).
    """
    x = np.asarray(x, dtype=np.float32)
    acc = np.zeros_like(x)
    n = 0
    for s in sigmas:
        sig = s * scale
        if sig < 0.5:
            continue
        mean = _gaussian_local_mean(x, sig)
        d = x - mean
        sd = np.sqrt(_gaussian_local_mean(d * d, sig))
        acc += np.arctan(np.float32(k) * d / np.maximum(sd, np.float32(1e-3)))
        n += 1
    return acc / np.float32(max(n, 1) * np.pi / 2)


def apply_structure(x, weight, scale=1.0):
    """Add the MGN local component (mgn_detail) with the given weight (0 = off) and clip to [0, 1]."""
    if weight <= 1e-6:
        return x
    return np.clip(x + np.float32(weight) * mgn_detail(x, scale), 0.0, 1.0).astype(np.asarray(x).dtype, copy=False)


def edge_strength(x, radius=1.0, scale=1.0):
    """Edge strength used by unsharp_mask's masking: gradient magnitude of the image blurred at `radius`
    (full-res px), expressed per full-resolution pixel so values compare across preview scales."""
    xf = np.asarray(x, dtype=np.float32)
    sig = radius * scale
    blur = cv2.GaussianBlur(xf, (0, 0), sig) if sig >= 0.3 else xf
    g = np.hypot(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    return g * np.float32(scale)


def masking_threshold(x, masking, radius=1.0, scale=1.0):
    """Absolute edge-strength threshold for unsharp_mask: the `masking` percentile of edge_strength(x)."""
    g = edge_strength(x, radius, scale)
    return float(np.percentile(g[:: max(1, g.shape[0] // 1024), :: max(1, g.shape[1] // 1024)], min(masking, 99.9)))


def unsharp_mask(x, amount, radius=1.0, detail=100.0, masking=0.0, scale=1.0, mask_threshold=None):
    """
    Lightroom-style unsharp mask on a 2D [0, 1] image.

    :param amount: strength (1.0 adds the full high-pass detail x - blur once); 0 = off.
    :param radius: Gaussian radius (sigma) in full-resolution pixels; multiplied by `scale` for previews.
    :param detail: halo allowance in % (100 = plain unsharp mask; 0 = no overshoot beyond the local min/max
        of the neighborhood, which suppresses bright/dark halos around the limb and flare kernels).
    :param masking: edge mask in % (0 = sharpen everywhere; e.g. 60 = only where the edge strength is above
        the 60th percentile of the image), keeping smooth / noisy areas such as the off-limb sky unsharpened.
    :param scale: image pixels per full-resolution pixel. A radius below ~0.3 image pixel has no effect.
    :param mask_threshold: absolute edge-strength threshold (see masking_threshold). Pass the value computed
        on the full image so crops, previews and the full-resolution output mask the same structures; when
        None it is the `masking` percentile of this image.
    """
    if amount <= 1e-6:
        return x
    x = np.asarray(x)
    sig = radius * scale
    if sig < 0.3:
        return x
    xf = x.astype(np.float32, copy=False)
    blur = cv2.GaussianBlur(xf, (0, 0), sig)
    out = xf + np.float32(amount) * (xf - blur)
    if detail < 100.0:
        ksize = 2 * int(np.ceil(sig)) + 1
        kernel = np.ones((max(ksize, 3), max(ksize, 3)), np.uint8)
        clamped = np.clip(out, cv2.erode(xf, kernel), cv2.dilate(xf, kernel))
        out = clamped + np.float32(detail / 100.0) * (out - clamped)
    if masking > 0.0:
        g = edge_strength(xf, radius, scale)
        thr = masking_threshold(xf, masking, radius, scale) if mask_threshold is None else float(mask_threshold)
        t = np.clip((g - 0.5 * thr) / max(thr, 1e-12), 0.0, 1.0)  # smoothstep from 0.5*thr to 1.5*thr
        out = xf + (t * t * (3.0 - 2.0 * t)).astype(np.float32) * (out - xf)
    return np.clip(out, 0.0, 1.0).astype(x.dtype, copy=False)


def enhance_channel(x, structure=0.0, sharpen=(0.0, 1.0, 100.0, 0.0), scale=1.0, mask_threshold=None):
    """Spatial enhancement after the tone curve: MGN structure, then unsharp-mask sharpening."""
    x = apply_structure(x, structure, scale)
    amount, radius, detail, masking = sharpen
    return unsharp_mask(x, amount, radius, detail, masking, scale, mask_threshold)


def scale_rgb(rgb, rgblow, rgbhigh, gamma_rgb=(2.8, 2.8, 2.4), rgbmix=None, scalemin=0, contrast=0.0, contrast_mid=0.5,
              structure=0.0, sharpen=(0.0, 1.0, 100.0, 0.0), sharpen_mask_thr=None):
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
    :param contrast: strength of the sigmoidal contrast applied to each channel after gamma (0 = off).
    :param contrast_mid: pivot of the sigmoidal contrast in the [0, 1] gamma-scaled range: a scalar,
    or one value per channel (see mean_preserving_midpoint).
    :param structure: weight of the MGN local component added to each channel after the tone curve (0 = off).
    :param sharpen: (amount, radius, detail, masking) of the unsharp mask applied last (amount 0 = off).
    :param sharpen_mask_thr: optional absolute masking thresholds, one per channel (see masking_threshold).
    :return: rescaled rgb image as numpy 3D array: [height, width, rgb channels]
    """

    rgb_gamma = 1 / np.array(gamma_rgb)

    # Copy is needed due to in-place operations
    rgb2 = rgb.copy()
    for i in range(3):
        rgb2[i] = (rgb[i] - rgblow[i]) * 1 / (rgbhigh[i] - rgblow[i])
        rgb2[i].clip(0, 1, out=rgb2[i])

    mids = np.broadcast_to(np.asarray(contrast_mid, dtype=np.float64), (3,))
    thrs = [None] * 3 if sharpen_mask_thr is None else list(sharpen_mask_thr)
    red, green, blue = [
        enhance_channel(sigmoid_contrast(channel ** gamma, contrast, mid), structure, sharpen, 1.0, thr)
        for (channel, gamma, mid, thr) in zip(rgb2, rgb_gamma, mids, thrs)
    ]

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
    bgr2 = ((bgr - bgr.min()) * 1 / (bgr.max() - bgr.min())).astype(np.float32)
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


def process_rgb_image(i, data_files, rgblow, rgbhigh, calibrate=True, jp2=False, gamma_rgb=(2.8, 2.8, 2.4), scalemin=0, rgbmix=None, lab=None, lmin=0, crop=None, filename_rgb=None, filename_lab=None, contrast=0.0, contrast_mid=0.5,
                      structure=0.0, sharpen=(0.0, 1.0, 100.0, 0.0), sharpen_mask_thr=None):
    """
    Create an rgb image out of three fits or jp2 files at different wavelengths, conveniently scaled for visualization.

    :param i: image index in the list of file
    :param data_files: list of RGB files. The list is 2D: [rgb channels][image index]
    :param rgblow: minimum intensity value(s) for rescaling. scalar or 3-element numpy array.
    :param rgbhigh: maximum intensity value(s) for rescaling. scalar or 3-element numpy array.
    :param calibrate: True if you need to calibrate from raw fits files.
    :param jp2: True if input files are pre-calibrated JPEG2000 (.jp2) images.
    :param gamma_rgb: gamma scaling factor for tone-mapping each channel from 3x12 bit hdr intensity to 3x8 bit
    :param scalemin: minimum intensity for contrast stretching
    :param contrast: strength of the sigmoidal contrast applied to each channel after gamma (0 = off)
    :param contrast_mid: pivot of the sigmoidal contrast in the [0, 1] gamma-scaled range (scalar or per channel)
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
    if jp2:
        pdatargb = [load_jp2(data_files[j][i]) for j in range(3)]
    elif calibrate:
        pdatargb = [calibration.aiaprep(data_files[j][i]) for j in range(3)]
    else:
        pdatargb = [load_fits(data_files[j][i]) for j in range(3)]

    # Apply hdr tone-mapping
    im_rgb255 = scale_rgb(pdatargb, rgblow, rgbhigh, gamma_rgb=gamma_rgb, scalemin=scalemin, rgbmix=rgbmix,
                          contrast=contrast, contrast_mid=contrast_mid, structure=structure, sharpen=sharpen,
                          sharpen_mask_thr=sharpen_mask_thr)

    # OpenCV orders channels as B,G,R instead of R,G,B. FITS files also need to be flipped upside down.
    if jp2:
        bgr_stack = np.flip(im_rgb255, axis=2)
    else:
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


def encode_video(images_dir, movie_filename, image_format='jpeg', fps=30, file_ext='.mp4',
                 crop=None, frame_size=None, padded_size=None, image_pattern_search=None, command_only=False):
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
               "-crf", "26",
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
        with fits.open(fitsfile) as hdul:
            if len(hdul) == 1:
                data = hdul[0].data.astype(np.float64)
            else:
                hdul[1].verify('silentfix')
                data = hdul[1].data.astype(np.float64)
            return data
    except FileNotFoundError:
        print("Could not open fits file")


def load_jp2(jp2file):
    """
    Load a pre-calibrated JPEG2000 (.jp2) image as a 2D float64 numpy array.

    :param jp2file: path to jp2 file
    :return: 2D numpy array of image intensities
    """
    data = cv2.imread(jp2file, cv2.IMREAD_UNCHANGED)
    if data is None:
        raise FileNotFoundError(f"Could not open jp2 file: {jp2file}")
    return data.astype(np.float64)
