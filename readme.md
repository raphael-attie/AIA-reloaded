# AIA-reloaded
We are a group of solar physicist working with the Solar Dynamics Observatory (SDO), a NASA mission. This project is a singular attempt at delivering a new view of our nearest star through the combination of advanced image processing methods with more awareness on how we, as humans and scientists, associate color with information on the Sun. 

This repository provides python scripts and functions to create calibrated and enhanced RGB image series and movies from the full resolution images of SDO/AIA at 3 different wavebands. 

[![RGB image from SDO/AIA](images/im_rgb_gamma_2.8_2.8_2.4_btf_0.3_169.jpeg)](https://youtu.be/TG8th-Skhx0)
See youtube video at https://youtu.be/TG8th-Skhx0


## Installation

Download and unpack files into an empty parent directory.

Main python dependencies:

* Python 3 (tested on 3.5 and above)
* numpy 
* astropy
* opencv (version 3.x.x)
* multiprocessing (optional, for parallel processing)
* [optional] pytest if you wish to run the unit tests. 

External dependencies:
* To fully automate the video creation from the rgb image series, you will need to install **ffmpeg**(https://www.ffmpeg.org)
 
If you wish to run the test functions with **pytest**: 

* create an **aia_data** directory in the parent directory

* Download the 3 test raw AIA fits files and place them into the **aia_data** directory:
  * [AIA 171](https://drive.google.com/open?id=1-qT9RFb8NXFWlhbvNVVXm52JJ0cjLJyV)
  * [AIA 193](https://drive.google.com/open?id=1NCVU91LQoFfmZMdg9eFec6nzdDf6q7Qx)
  * [AIA 304](https://drive.google.com/open?id=1lCoKH_BghuCuFwsrTVbtBz76ynVG_NQP)
  
* From the terminal, go into the project directory and run **pytest test/test_aia.py -v**

### How does it work? 

This framework assumes you know how to download the raw fits files from SDO/AIA. 
An example of the rgb image processing is given in **example_RGBMixer.py_**. After "prepping" the raw fits files (remapping to equal plate scales and rotations if needed), the intensity in each channel is rescaled non-linearly. Our examples use the wavebands centered at 304 (red channel), 171 (green channel) and 193 Anstrom (blue channel). Default rescaling parameters are provided but the method is meant to accept your own color mixing and rescaling parameters. 
To create movies, see examples in **aia_rgb_movies.py**. A full pipeline example is given in **script_full_pipeline.py**
The assignment of these wavebands to these colors is chosen in accordance to general human perception of colors. To that end, we are studying how to best make use of the CIE-based color spaces (e.g: CIELab, CIELuv, ...) which intend to implement as the frontend of the colorization instead of directly acting on the mixing between the RGB components. 

### Getting started

Example using the 3 samples from above that writes a .jpeg file of a colored image with default scaling and color mixing parameters. 
The .jpeg file is written in the **aia_data** directory where you put the 3 samples.

```python
import os, glob
from visualization import RGBMixer

## Create rgb image using the default aia rgb mixer.
# This example provides the data file paths and output directory

aia_mixer = RGBMixer(
    data_files = [glob.glob('../aia_data/*.%s*.fits'%wavel) for wavel in ['304', '171', '193']],
    outputdir = os.path.abspath('../aia_data/'))
aia_mixer.set_aia_default()

aia_mixer.process_rgb(0)
```

The above example will produce one colored image at full resolution in .jpeg that is visually identical to a lossless format (e.g png). 
typically, beamers in conferences are at best at "full HD", i.e 1920x1080. That means you do not use the whole resolution. What you display with a powerpoint or keynote presentation is actually downsampled to fit the screen. So our pipeline uses FFMPEG to enable you to do this downsampling and get properly sized video, with much managable file sizes to share with various media (powerpoint, youtube, ...). 

We also provide options to create videos and crop within the full resolution images for close-ups. Nonetheless, even a full resolution jpeg image size is 4.1 MB. if you do not want or do not need to retain the full resolution images and prefer to directly crop them before creating the image series, you can ask for cropped image directly when instantiating the RGBMixer class. For example, to only retain the bottom half of the sun one would add the following optional crop parameter:

```python
aia_mixer.crop = (slice(0,2048), slice(2048,4095)) # the first slice is over

```

To create movies, you'll process multiple rgb images from a list of raw fits files. Instead of ```aia_mixer.process_rgb(0)``` you would use ```aia_mixer.process_rgb_list(ncores, file_range)``` where e.g: ```file_range = range(200)``` to process images, and ```ncores = 4``` to parralelize over 4 workers. 

Here is an example of full pipeline for processing, say the first 225 images present in your directory and create movie of of the full sun with a square resolution of 1080x1080:

```python
import os
import visualization

# Set parallelization. To disable, set ncores to 1. Try to leave 1 or 2 cores available in your machine if you want to keep working normally:
# E.g., if you have 8 cores in your computer (whether virtual cores or physical cores). Use maximum ncores = 6
ncores = 4
# Range of images indices to process.
file_range = range(225)

aia_mixer = visualization.RGBMixer(
    data_dir=os.path.expanduser('~/Data/SDO/AIA/event_2012_08_31/'),
    wavel_dirs=['304', '171', '193'],
    outputdir=os.path.abspath('../aia_data/'))
aia_mixer.set_aia_default()
aia_mixer.filename_lab = 'im_lab'

aia_mixer.process_rgb_list(ncores, file_range)

'''
This will be followed by the encoding of the movie with FFMPEG:

'''python
##### Create .mp4 videos

## full sun rescaled to 1080x1080 px
frame_size = (1080, 1080)
filename = 'rgb_movie_full_sun_1080x1080'
fps = 30  # Number of frames per second
visualization.encode_video(aia_mixer.outputdir, filename, fps=fps, frame_size=frame_size)

```
Video at: https://youtu.be/LivB3rEmXJQ (make sure you set the maximum resolution on the player)


Sometimes some media players work better if you provide videos with a 16:9 or 4:3 geometry. For the video above, we can instead get a padded version, where black stripes will be added on either side of our initial 1080x1080 picture frame to make it a 1920x1080 video. We would just add the extra ```padded_size``` parameter of (1920,1080):

```python
    ## Rectangularly padded to fit 16:9 1920x1080p for optimized youtube streaming
    padded_size = (1920, 1080)
    visualization.encode_video(aia_mixer.outputdir, filename, fps=fps, frame_size=frame_size, padded_size=padded_size)

```
Video at: https://youtu.be/XyqYKlLQZ8o (make sure you set the maximum resolution on the player)

Having such 16:9 ratio usually enables your computer to decode and play the video more efficiently than a less "standard" geometry. 4:3 is also a standard geometry that plays nicely in modest computers and older softwares. However, Youtube players work best at 16:9 and uploading those online will make maximize quality, as they would otherwise be re-encoded by Youtube to fit their 16:9 players. For such complex images, this can lead to unexpected results due to the use of (sometimes) lossy compression schemes. It is best to do that conversion from the source images like we showed above. In addition, the file size should be the same, as the padding only add zeros. 
The file size of the above videos is ~11 MB. 

Another example is if you want a close-up on the sun, so you can view a sub-field of the sun at higher resolution. Here, we crop a 3840x2160 section (which is a 16:9 geometry) off the 4096x4096 frame, and encode this again at 1920x1080, although we do not need to pad anything and exploit thus the full resolution of a typical display. 

```python
    image_pattern_search = "im_lab_0*.jpeg"
    crop = [3840, 2160, 128, 1936]
    frame_size = (1920, 1080)
    movie_filename = 'rgb_movie_3840x2160_1920x1080'
    fps = 30
    # Encode movie
    command = visualization.encode_video(aia_mixer.outputdir, movie_filename, crop=crop, frame_size=frame_size, image_pattern_search=image_pattern_search)

```
Video at: https://youtu.be/CqIhzQMOLRw

As we have more information in this 1920x1080 than above, the file size is a bit bigger, ~15 MB. Considering we had 225 images of ~4.1 MB, that would result in nearly 1 GB of uncompressed video size for the full sun, for virtually no difference in how the images are rendedered on a typical full HD display!








