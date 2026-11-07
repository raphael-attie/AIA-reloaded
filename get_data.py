import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
from hvpy import DataSource, getJP2Image, save_file


AIA_SOURCES = {
    "94": DataSource.AIA_94,
    "131": DataSource.AIA_131,
    "171": DataSource.AIA_171,
    "193": DataSource.AIA_193,
    "211": DataSource.AIA_211,
    "304": DataSource.AIA_304,
    "335": DataSource.AIA_335,
    "1600": DataSource.AIA_1600,
    "1700": DataSource.AIA_1700,
    "4500": DataSource.AIA_4500,
}


def parse_utc_datetime(dt_value):
    """Parse a date/time string or datetime object into a naive UTC datetime."""
    if isinstance(dt_value, datetime):
        return dt_value
    text = dt_value.strip().rstrip("Z")
    return datetime.fromisoformat(text)


def format_event_dirname(start_dt, end_dt):
    """Build event directory name, e.g. event_2012-08-31_1800_2012-08-31_2200."""
    return f"event_{start_dt.strftime('%Y-%m-%d_%H%M')}_{end_dt.strftime('%Y-%m-%d_%H%M')}"


def download_aia_jp2(
    start_date,
    end_date,
    wavelengths=("304", "171", "193"),
    cadence=36,
    base_dir="~/data/AIA",
    overwrite=False,
):
    """
    Download SDO/AIA JPEG2000 (.jp2) images over a UTC date/time range using hvpy.

    Files are saved under:
        <base_dir>/event_<start>_<end>/<wavelength>/aia_<wavelength>_<timestamp>.jp2

    :param start_date: UTC start date/time (str such as "2012-08-31 18:00" or datetime)
    :param end_date: UTC end date/time (str such as "2012-08-31 22:00" or datetime)
    :param wavelengths: sequence of AIA wavelength strings (default: ('304', '171', '193'))
    :param cadence: time step in seconds between consecutive frames (default: 36)
    :param base_dir: parent directory for event folders (default: '~/data/AIA')
    :param overwrite: whether to overwrite existing .jp2 files (default: False)
    :return: Path to the created event directory
    """
    start_dt = parse_utc_datetime(start_date)
    end_dt = parse_utc_datetime(end_date)
    if end_dt < start_dt:
        raise ValueError(f"end_date ({end_dt}) must be >= start_date ({start_dt})")

    for wavel in wavelengths:
        if str(wavel) not in AIA_SOURCES:
            raise ValueError(f"Unsupported AIA wavelength '{wavel}'. Supported: {list(AIA_SOURCES.keys())}")

    event_dir = Path(os.path.expanduser(base_dir)) / format_event_dirname(start_dt, end_dt)
    wavel_dirs = {}
    for wavel in wavelengths:
        wdir = event_dir / str(wavel)
        wdir.mkdir(parents=True, exist_ok=True)
        wavel_dirs[str(wavel)] = wdir

    # Generate list of timestamps from start_dt to end_dt at the requested cadence
    timestamps = []
    cur_dt = start_dt
    step = timedelta(seconds=cadence)
    while cur_dt <= end_dt:
        timestamps.append(cur_dt)
        cur_dt += step

    print(f"Output directory : {event_dir}")
    print(f"Time range (UTC) : {start_dt} -> {end_dt} ({len(timestamps)} timestamps @ {cadence}s cadence)")
    print(f"Wavelengths      : {', '.join(str(w) for w in wavelengths)}")

    for dt in tqdm(timestamps, desc="Downloading JP2 frames", unit="step"):
        ts_str = dt.strftime("%Y-%m-%dT%H%M%SZ")
        for wavel in wavelengths:
            w_str = str(wavel)
            out_file = wavel_dirs[w_str] / f"aia_{w_str}_{ts_str}.jp2"
            if out_file.exists() and not overwrite:
                continue
            jp2_bytes = getJP2Image(date=dt, sourceId=AIA_SOURCES[w_str])
            save_file(jp2_bytes, out_file, overwrite=True)

    return event_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download SDO/AIA JPEG2000 (.jp2) images for a UTC date/time range using hvpy."
    )
    parser.add_argument(
        "start_date",
        nargs="?",
        default="2012-08-31 18:00",
        help='Start date/time in UTC (default: "2012-08-31 18:00")',
    )
    parser.add_argument(
        "end_date",
        nargs="?",
        default="2012-08-31 22:00",
        help='End date/time in UTC (default: "2012-08-31 22:00")',
    )
    parser.add_argument(
        "--cadence",
        type=int,
        default=36,
        help="Cadence in seconds between frames (default: 36)",
    )
    parser.add_argument(
        "--wavelengths",
        nargs="+",
        default=["304", "171", "193"],
        help="AIA wavelengths to download (default: 304 171 193)",
    )
    parser.add_argument(
        "--base-dir",
        default="~/data/AIA",
        help="Base output directory (default: ~/data/AIA)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files if already downloaded",
    )

    args = parser.parse_args()
    download_aia_jp2(
        start_date=args.start_date,
        end_date=args.end_date,
        wavelengths=args.wavelengths,
        cadence=args.cadence,
        base_dir=args.base_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
