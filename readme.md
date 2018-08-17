# aia-reloaded
We are a group of solar physicist working with the Solar Dynamics Observatory (SDO), a NASA mission. This project is a singular attempt at delivering a new view of our nearest star through the combination of advanced image processing methods with more awareness on how we, as humans and scientists, associate color with information on the Sun. 

This repository provides python scripts and functions to create calibrated and enhanced RGB image series and movies from the full resolution images of SDO/AIA at 3 different wavebands. 

[![RGB image from SDO/AIA](images/im_rgb_gamma_2.8_2.8_2.4_btf_0.3_000_default.jpeg)](https://youtu.be/hC3gCB7ergY)
See youtube video at https://youtu.be/hC3gCB7ergY


## Installation

Download and unpack files into an empty parent directory.

Main dependencies:

* Python 3 (tested on 3.5 and above)
* astropy
* opencv (version 3.x.x)
* numpy (will be installed, updated or downgraded when you install opencv)
* multiprocessing (optional, for parallel processing)
* functools (if using multiprocessing)
* [optional] pytest if you wish to run the unit tests. 

 
If you wish to run the test functions with **pytest**: 

* create an **aia_data** directory in the parent directory

* Download the 3 test raw AIA fits files and place them into the **aia_data** directory:
  * https://drive.google.com/open?id=1-qT9RFb8NXFWlhbvNVVXm52JJ0cjLJyV
  * https://drive.google.com/open?id=1NCVU91LQoFfmZMdg9eFec6nzdDf6q7Qx
  * https://drive.google.com/open?id=1lCoKH_BghuCuFwsrTVbtBz76ynVG_NQP
  
* To fully automate the video creation from the rgb image series, you will need to install **ffmpeg**(https://www.ffmpeg.org)

### How does it work? 

This framework assume you know how to download the raw fits files from SDO/AIA. 
An example of the rgb image processing is given in **aia_rgb_images.py**. It processes the raw fits files by rescaling each channel which correspond to one waveband. Our examples use the wavebands centered at 304 (red channel), 171 (green channel) and 193 Anstrom (blue channel).
The assignment of these wavebands to these colors is chosen in accordance to general human perception of colors. 
To create movies, see examples in **aia_rgb_movies.py**