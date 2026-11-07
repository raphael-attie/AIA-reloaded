import argparse
import os
import sys

# Ensure project root is on sys.path when running from dev_Lab/ or project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from visualization import RGBMixer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process AIA JPEG2000 (.jp2) wavelength series into RGB/CIELab JPEG images in parallel."
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

    aia_mixer = RGBMixer(
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
    print(f"Output directory : {outputdir}")
    print(f"Processing {n_frames} frames using {args.ncores} worker(s)...")

    aia_mixer.process_rgb_list(args.ncores, file_range)


if __name__ == "__main__":
    main()
