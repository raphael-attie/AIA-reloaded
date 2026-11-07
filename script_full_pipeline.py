"""
Script automating the creation of AIA RGB images and movies from JPEG2000 (.jp2) files.
This script assumes that AIA .jp2 files from the 3 different wavelengths are in three subdirectories
under the same parent directory `data_dir` (e.g. created by get_data.py).

The script first creates RGB/CIELab JPEG images written to `<data_dir>/rgb_images` (by default),
then calls ffmpeg to encode the movie directly under `data_dir`.
"""

import argparse
import os
import visualization


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full pipeline: convert AIA .jp2 wavelength series to RGB JPEG images and encode an MP4 movie."
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="~/data/AIA/event_2012-08-31_1800_2012-08-31_2200",
        help="Parent directory containing the 3 wavelength subdirectories (default: ~/data/AIA/event_2012-08-31_1800_2012-08-31_2200)",
    )
    parser.add_argument(
        "-w",
        "--wavelengths",
        nargs=3,
        default=["304", "171", "193"],
        metavar=("RED", "GREEN", "BLUE"),
        help="Three AIA wavelength subdirectory names in R G B order (default: 304 171 193)",
    )
    parser.add_argument(
        "-o",
        "--outputdir",
        default=None,
        help="Output directory for RGB JPEG images (default: <data_dir>/rgb_images)",
    )
    parser.add_argument(
        "-n",
        "--ncores",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4; set to 1 to disable multiprocessing)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.wavelengths) != 3:
        raise ValueError("Exactly 3 wavelengths must be provided for (R, G, B).")

    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    outputdir = (
        os.path.abspath(os.path.expanduser(args.outputdir))
        if args.outputdir is not None
        else os.path.join(data_dir, "rgb_images")
    )
    os.makedirs(outputdir, exist_ok=True)

    aia_mixer = visualization.RGBMixer(
        data_dir=data_dir,
        wavel_dirs=list(args.wavelengths),
        jp2=True,
        outputdir=outputdir,
    )
    aia_mixer.set_jp2_default()

    n_frames = min(len(files) for files in aia_mixer.data_files)
    file_range = range(n_frames)

    print(f"Input directory  : {data_dir}")
    print(f"Wavelengths (RGB): {args.wavelengths}")
    print(f"Images directory : {outputdir}")
    print(f"Processing {n_frames} frames using {args.ncores} worker(s)...")

    aia_mixer.process_rgb_list(args.ncores, file_range)

    ##### Create .mp4 videos right under data_dir

    ## full sun rescaled to 1080x1080 px
    frame_size = (1080, 1080)
    movie_filepath = os.path.join(data_dir, "rgb_movie_full_sun_1080x1080")
    fps = 30  # Number of frames per second
    image_pattern_search = f"{aia_mixer.filename_lab}_*.jpeg"
    visualization.encode_video(
        aia_mixer.outputdir,
        movie_filepath,
        fps=fps,
        frame_size=frame_size,
        image_pattern_search=image_pattern_search,
    )

    ## Rectangularly padded to fit 16:9 1920x1080p for optimized youtube streaming
    # padded_size = (1920, 1080)
    # movie_filepath = os.path.join(data_dir, 'rgb_movie_full_sun_1080x1080_padded_1920_1080')
    # visualization.encode_video(aia_mixer.outputdir, movie_filepath, fps=fps, frame_size=frame_size, padded_size=padded_size, image_pattern_search=image_pattern_search)

    ### Create video With a 16:9 aspect ratio, crop over 3840 x 2160 around bottom half
    ### and output at full HD resolution (1920 x 1080) at 30 frames per second.
    # crop = [3840, 2160, 128, 1936]
    # frame_size = (1920, 1080)
    # movie_filepath = os.path.join(data_dir, 'rgb_movie_3840x2160_1920x1080')
    # fps = 30
    # command = visualization.encode_video(aia_mixer.outputdir, movie_filepath, crop=crop, frame_size=frame_size, image_pattern_search=image_pattern_search)


if __name__ == "__main__":
    main()
