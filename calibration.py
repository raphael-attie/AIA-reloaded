import numpy as np
import cv2
from astropy.io import fits

def scale_rotate(image, angle=0, scale_factor=1, reference_pixel=None):
    """
    Perform scaled rotation with opencv. About 20 times faster than with Sunpy & scikit/skimage warp methods.
    The output is a padded image that holds the entire rescaled,rotated image, recentered around the reference pixel.
    Positive-angle rotation will go counterclockwise if the array is displayed with the origin on top (default),
    and clockwise with the origin at bottom.

    :param image: Numpy 2D array
    :param angle: rotation angle in degrees. Positive angle  will rotate counterclocwise if array origin on top-left
    :param scale_factor: ratio of the wavelength-dependent pixel scale over the target scale of 0.6 arcsec
    :param reference_pixel: tuple of (x, y) coordinate. Given as (x, y) = (col, row) and not (row, col).
    :return: padded scaled and rotated image
    """
    array_center = (np.array(image.shape)[::-1] - 1) / 2.0

    if reference_pixel is None:
        reference_pixel = array_center

    # convert angle to radian
    angler = angle * np.pi / 180
    # Get basic rotation matrix to calculate initial padding extent
    rmatrix = np.matrix([[np.cos(angler), -np.sin(angler)],
                         [np.sin(angler), np.cos(angler)]])

    extent = np.max(np.abs(np.vstack((image.shape * rmatrix,
                                      image.shape * rmatrix.T))), axis=0)

    # Calculate the needed padding or unpadding
    diff = np.asarray(np.ceil((extent - image.shape) / 2), dtype=int).ravel()
    diff2 = np.max(np.abs(reference_pixel - array_center)) + 1
    # Pad the image array
    pad_x = int(np.ceil(np.max((diff[1], 0)) + diff2))
    pad_y = int(np.ceil(np.max((diff[0], 0)) + diff2))

    padded_reference_pixel = reference_pixel + np.array([pad_x, pad_y])
    padded_image = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=(0, 0))
    padded_array_center = (np.array(padded_image.shape)[::-1] - 1) / 2.0

    # Get scaled rotation matrix accounting for padding
    rmatrix_cv = cv2.getRotationMatrix2D((padded_reference_pixel[0], padded_reference_pixel[1]), angle, scale_factor)
    # Adding extra shift to recenter:
    # move image so the reference pixel aligns with the center of the padded array
    shift = padded_array_center - padded_reference_pixel
    rmatrix_cv[0, 2] += shift[0]
    rmatrix_cv[1, 2] += shift[1]
    # Do the scaled rotation
    rotated_image = cv2.warpAffine(padded_image, rmatrix_cv, padded_image.shape, cv2.INTER_CUBIC)
    rotated_image[rotated_image<0] = 0

    return rotated_image


def aiaprep(fitsfile, cropsize=None):

    hdul = fits.open(fitsfile)
    hdul[1].verify('silentfix')
    header = hdul[1].header
    data = hdul[1].data.astype(np.float64)
    data /= header['EXPTIME']
    # Target scale is 0.6 arcsec/px
    target_scale = 0.6
    scale_factor = header['CDELT1'] / target_scale
    # Center of rotation at reference pixel converted to a coordinate origin at 0
    reference_pixel = [header['CRPIX1'] - 1, header['CRPIX2'] - 1]
    # Rotation angle with openCV uses coordinate origin at top-left corner. For solar images in numpy we need to invert the angle.
    angle = -header['CROTA2']
    # Run scaled rotation. The output will be a rotated, rescaled, padded array.
    prepdata = scale_rotate(data, angle=angle, scale_factor=scale_factor, reference_pixel=reference_pixel)
    prepdata[prepdata<0] = 0

    if cropsize is not None:
        center = ((np.array(prepdata.shape) - 1) / 2.0).astype(int)
        half_size = int(cropsize / 2)
        prepdata = prepdata[center[1] - half_size:center[1] + half_size, center[0] - half_size:center[0] + half_size]

    return prepdata
