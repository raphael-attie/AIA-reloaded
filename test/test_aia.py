import os, glob
import numpy as np
from calibration import scale_rotate, aiaprep
from visualization import RGBMixer


# Testing for any non-zero values at borders
def test_scale_rotate_600x600_random_data():
    image = (np.random.rand(600, 600) * 10).astype(np.float64)
    image[100:500, 300:400] = 0
    image[299:302, 349:352] = 10
    # reference pixel defined as (x, y) = (cols, rows) and not (rows, cols)
    reference_pixel = np.array([350, 300])
    angle = 30  # degrees
    rotated_image = scale_rotate(image, angle=angle, scale_factor=1, reference_pixel=reference_pixel)
    edge_sum = int(rotated_image[:,0].sum() + rotated_image[:,-1].sum() + rotated_image[0,:].sum() + rotated_image[-1,:].sum())
    assert edge_sum == 0


# Testing for any non-zero values at borders
def test_scale_rotate_4096x4096_random_data():
    image = (np.random.rand(4096, 4096) * 2**12).astype(np.float64)
    # reference pixel defined as (x, y) = (cols, rows) and not (rows, cols)
    reference_pixel = np.array([2051, 2054])
    angle = 30  # degrees
    rotated_image = scale_rotate(image, angle=angle, scale_factor=1, reference_pixel=reference_pixel)
    edge_sum = int(rotated_image[:,0].sum() + rotated_image[:,-1].sum() + rotated_image[0,:].sum() + rotated_image[-1,:].sum())
    assert edge_sum == 0


def test_file_exist():
    assert len(glob.glob('../aia_data/*.fits')) > 0, "the list is empty"


def test_rgb_files_exist():
    assert len(glob.glob('../aia_data/*304*.fits')) > 0 and \
           len(glob.glob('../aia_data/*193*.fits')) > 0 and \
           len(glob.glob('../aia_data/*171*.fits')) > 0


def test_aiaprep():
    fitsfile = glob.glob('../aia_data/*.fits')[0]
    _ = aiaprep(fitsfile)


def test_aiaprep_crop_4096():
    fitsfile = glob.glob('../aia_data/*.fits')[0]
    _ = aiaprep(fitsfile, cropsize=4096)



def setup_function(test_aia_rgb_sample):

    for file in glob.glob('../aia_data/*.jpeg'):
        if os.path.exists(file):
            os.remove(file)


def test_aia_rgb_sample():

    aia_mixer = RGBMixer(
        data_files=[glob.glob('../aia_data/*.%s*.fits' % wavel) for wavel in ['304', '171', '193']],
        outputdir=os.path.abspath('../aia_data/rgb/'))
    aia_mixer.set_aia_default()

    _ = aia_mixer.process_rgb(0)

    outputfile_lab = aia_mixer.filepath_lab + '_%04d.jpeg' % 0

    assert os.path.isfile(outputfile_lab)











