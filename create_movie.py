"""
Create an MP4 movie from a directory of RGB JPEG images (e.g., after running
dev_Lab/script_RGBMixer_parallel.py), with optional cropping, resizing, and padding.
"""

import argparse
import os
import time
import visualization


def build_default_movie_name(crop, frame_size, padded_size):
    """Construct a descriptive default movie basename from crop/frame_size/padded_size."""
    parts = ["rgb_movie"]
    if crop is not None:
        parts.append(f"crop_{crop[0]}x{crop[1]}_{crop[2]}_{crop[3]}")
    else:
        parts.append("full")
    if frame_size is not None:
        parts.append(f"{frame_size[0]}x{frame_size[1]}")
    if padded_size is not None:
        parts.append(f"padded_{padded_size[0]}x{padded_size[1]}")
    return "_".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encode an MP4 movie from a directory of RGB JPEG images using ffmpeg.",
        epilog=(
            "Examples:\n"
            "  python create_movie.py ~/data/AIA/event_2012-08-31_1800_2012-08-31_2200/rgb_images -s 1080 1080\n"
            "  python create_movie.py ~/data/AIA/event_2012-08-31_1800_2012-08-31_2200/rgb_images -c 3840 2160 128 1935 -s 1920 1080\n"
            "  python create_movie.py ~/data/AIA/event_2012-08-31_1800_2012-08-31_2200/rgb_images -s 1080 1080 -p 1920 1080"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "images_dir",
        nargs="?",
        default="~/data/AIA/event_2012-08-31_1800_2012-08-31_2200/rgb_images",
        help="Directory containing the RGB JPEG images (default: ~/data/AIA/event_2012-08-31_1800_2012-08-31_2200/rgb_images)",
    )
    parser.add_argument(
        "-c",
        "--crop",
        nargs=4,
        type=int,
        default=None,
        metavar=("WIDTH", "HEIGHT", "X", "Y"),
        help="Crop window [width height x y] where (x, y) is the top-left corner (e.g. 3840 2160 128 1935)",
    )
    parser.add_argument(
        "-s",
        "--frame-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Output video frame size [width height] in pixels (e.g. 1080 1080 or 1920 1080)",
    )
    parser.add_argument(
        "-p",
        "--padded-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Optional padded video size [width height] in pixels (e.g. 1920 1080)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)",
    )
    parser.add_argument(
        "--pattern",
        default="*.jpeg",
        help='Glob pattern for input JPEG images inside images_dir (default: "*.jpeg")',
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output movie filename or path without .mp4 extension (default: auto-named in parent data_dir)",
    )
    return parser.parse_args()


def create_movie(images_dir, output=None, crop=None, frame_size=None, padded_size=None, fps=30, pattern="*.jpeg"):
    """Encode images_dir/<pattern> into an MP4 and return its path.

    output: movie path with or without .mp4 (default: auto-named, in <data_dir> when images_dir is
    <data_dir>/rgb_images, else in images_dir). Raises RuntimeError if ffmpeg did not write the movie."""
    images_dir = os.path.abspath(os.path.expanduser(images_dir))
    if not os.path.isdir(images_dir):
        raise ValueError(f"Images directory does not exist: {images_dir}")

    crop = list(crop) if crop is not None else None
    frame_size = tuple(frame_size) if frame_size is not None else None
    padded_size = tuple(padded_size) if padded_size is not None else None

    if output is not None:
        movie_filename = os.path.expanduser(output)
        if movie_filename.endswith(".mp4"):
            movie_filename = movie_filename[:-4]
    else:
        default_name = build_default_movie_name(crop, frame_size, padded_size)
        # If images_dir is a subdirectory like <data_dir>/rgb_images, save the movie right under <data_dir>
        parent_dir = (
            os.path.dirname(images_dir)
            if os.path.basename(images_dir) == "rgb_images"
            else images_dir
        )
        movie_filename = os.path.join(parent_dir, default_name)
    movie_filename = os.path.abspath(movie_filename)  # encode_video runs ffmpeg with cwd=images_dir

    print(f"Images directory : {images_dir}")
    print(f"Crop             : {crop}")
    print(f"Frame size       : {frame_size}")
    print(f"Padded size      : {padded_size}")
    print(f"Output movie     : {movie_filename}.mp4")

    out_path = movie_filename + ".mp4"
    start = time.time()
    visualization.encode_video(
        images_dir,
        movie_filename,
        fps=fps,
        crop=crop,
        frame_size=frame_size,
        padded_size=padded_size,
        image_pattern_search=pattern,
    )
    if not os.path.isfile(out_path) or os.path.getmtime(out_path) < start:
        raise RuntimeError(f"ffmpeg did not write {out_path}")
    return out_path


def main():
    args = parse_args()
    create_movie(args.images_dir, args.output, args.crop, args.frame_size, args.padded_size, args.fps, args.pattern)


if __name__ == "__main__":
    main()
