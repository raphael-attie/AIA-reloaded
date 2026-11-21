import argparse
import bisect
import os
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
from hvpy import DataSource, getClosestImage, getJP2Image, save_file


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

HMI_SOURCES = {
    "continuum": DataSource.HMI_INT,
    "magnetogram": DataSource.HMI_MAG,
}

# Every downloadable channel: AIA wavelengths and HMI products, which share the same event folder layout.
SOURCES = {**AIA_SOURCES, **HMI_SOURCES}

# Helioviewer serves HMI images every 45 s: a finer time grid would fetch the same image repeatedly.
HMI_CADENCE = 45


def instrument(channel):
    """'aia' or 'hmi': file name prefix of a channel."""
    return "hmi" if str(channel) in HMI_SOURCES else "aia"


def channel_cadence(channel, cadence):
    """Time step in seconds used for a channel: `cadence`, but never finer than HMI_CADENCE for HMI."""
    return max(cadence, HMI_CADENCE) if instrument(channel) == "hmi" else cadence


def parse_utc_datetime(dt_value):
    """Parse a date/time string or datetime object into a naive UTC datetime."""
    if isinstance(dt_value, datetime):
        return dt_value
    text = dt_value.strip().rstrip("Z")
    return datetime.fromisoformat(text)


def format_event_dirname(start_dt, end_dt):
    """Build event directory name, e.g. event_2012-08-31_1800_2012-08-31_2200."""
    return f"event_{start_dt.strftime('%Y-%m-%d_%H%M')}_{end_dt.strftime('%Y-%m-%d_%H%M')}"


_FILE_TS_RE = re.compile(r"(?:aia|hmi)_([a-z0-9]+)_(\d{4}-\d{2}-\d{2}T\d{6})Z?\.jp2$")


def jp2_filename(wavel, dt):
    """File name of one frame, e.g. aia_171_2012-08-31T180000Z.jp2 or hmi_magnetogram_2012-08-31T180000Z.jp2."""
    return f"{instrument(wavel)}_{wavel}_{dt.strftime('%Y-%m-%dT%H%M%SZ')}.jp2"


def frame_timestamps(start_dt, end_dt, cadence):
    """Timestamps from start_dt to end_dt (inclusive) at the given cadence in seconds."""
    step = timedelta(seconds=cadence)
    out, cur = [], start_dt
    while cur <= end_dt:
        out.append(cur)
        cur += step
    return out


def closest_image_dates(date, wavelengths):
    """Date/time of the Helioviewer image closest to `date`, per wavelength (or HMI channel)."""
    dt = parse_utc_datetime(date)
    return {
        str(w): datetime.fromisoformat(getClosestImage(date=dt, sourceId=SOURCES[str(w)])["date"])
        for w in wavelengths
    }


def local_jp2_index(base_dir, wavelengths, exclude=None):
    """Sorted (timestamp, path) of every jp2 already in an event folder under base_dir, per wavelength."""
    base = Path(os.path.expanduser(base_dir))
    index = {str(w): [] for w in wavelengths}
    for event_dir in base.glob("event_*"):
        if exclude is not None and event_dir.resolve() == Path(exclude).resolve():
            continue
        for w in index:
            for f in (event_dir / w).glob("*.jp2"):
                m = _FILE_TS_RE.search(f.name)
                if m and m.group(1) == w:
                    index[w].append((datetime.strptime(m.group(2), "%Y-%m-%dT%H%M%S"), f))
    for files in index.values():
        files.sort()
    return index


def nearest_local(files, dt, tolerance):
    """Path in a sorted (timestamp, path) list closest to dt, if within tolerance (timedelta)."""
    i = bisect.bisect_left(files, (dt,))
    best = min(files[max(i - 1, 0): i + 1], key=lambda tf: abs(tf[0] - dt), default=None)
    return best[1] if best is not None and abs(best[0] - dt) <= tolerance else None


def download_sdo_jp2(
    start_date,
    end_date,
    wavelengths=("304", "171", "193"),
    cadence=36,
    base_dir="~/data/AIA",
    overwrite=False,
    reuse_existing=True,
    workers=4,
    progress=None,
    cancel=None,
):
    """
    Download SDO/AIA and SDO/HMI JPEG2000 (.jp2) images over a UTC date/time range using hvpy.

    Files are saved under:
        <base_dir>/event_<start>_<end>/<wavelength>/aia_<wavelength>_<timestamp>.jp2
    or, for the HMI channels 'continuum' (intensity continuum) and 'magnetogram':
        <base_dir>/event_<start>_<end>/<channel>/hmi_<channel>_<timestamp>.jp2

    Each instrument has its own time grid: AIA channels at `cadence`, HMI channels at
    channel_cadence() (at least HMI_CADENCE = 45 s), so no image is downloaded twice.

    Frames already present in another event folder under base_dir (within half the channel's cadence
    of a requested timestamp) are copied from there instead of being downloaded again.

    :param start_date: UTC start date/time (str such as "2012-08-31 18:00" or datetime)
    :param end_date: UTC end date/time (str such as "2012-08-31 22:00" or datetime)
    :param wavelengths: sequence of AIA wavelength strings and/or HMI channels 'continuum', 'magnetogram'
        (default: ('304', '171', '193'))
    :param cadence: time step in seconds between consecutive frames (default: 36; HMI: at least 45)
    :param base_dir: parent directory for event folders (default: '~/data/AIA')
    :param overwrite: whether to overwrite existing .jp2 files (default: False)
    :param reuse_existing: copy matching frames from other event folders instead of downloading them (default: True)
    :param workers: number of parallel downloads (default: 4)
    :param progress: optional callable(done, total) called after each file
    :param cancel: optional threading.Event; when set, remaining files are skipped
    :return: Path to the created event directory
    """
    start_dt = parse_utc_datetime(start_date)
    end_dt = parse_utc_datetime(end_date)
    if end_dt < start_dt:
        raise ValueError(f"end_date ({end_dt}) must be >= start_date ({start_dt})")

    for wavel in wavelengths:
        if str(wavel) not in SOURCES:
            raise ValueError(f"Unsupported wavelength/channel '{wavel}'. Supported: {list(SOURCES.keys())}")
    wavelengths = [str(w) for w in wavelengths]

    event_dir = Path(os.path.expanduser(base_dir)) / format_event_dirname(start_dt, end_dt)
    for wavel in wavelengths:
        (event_dir / wavel).mkdir(parents=True, exist_ok=True)

    cadences = {w: channel_cadence(w, cadence) for w in wavelengths}
    local = local_jp2_index(base_dir, wavelengths, exclude=event_dir) if reuse_existing else {}
    tolerance = {w: timedelta(seconds=c / 2) for w, c in cadences.items()}

    print(f"Output directory : {event_dir}")
    print(f"Time range (UTC) : {start_dt} -> {end_dt}")
    for c in sorted(set(cadences.values())):
        chans = [w for w in wavelengths if cadences[w] == c]
        print(f"  @ {c}s cadence : {', '.join(chans)} ({len(frame_timestamps(start_dt, end_dt, c))} timestamps each)")

    # In time order across channels, so a cancelled download keeps a complete leading stretch.
    jobs = sorted(((dt, w) for w in wavelengths for dt in frame_timestamps(start_dt, end_dt, cadences[w])),
                  key=lambda job: job[0])
    done = 0
    lock = threading.Lock()

    def fetch(job):
        nonlocal done
        dt, w = job
        if cancel is not None and cancel.is_set():
            return
        out_file = event_dir / w / jp2_filename(w, dt)
        if overwrite or not out_file.exists():
            src = nearest_local(local.get(w, []), dt, tolerance[w]) if reuse_existing else None
            if src is not None:
                shutil.copy2(src, out_file)
            else:
                save_file(getJP2Image(date=dt, sourceId=SOURCES[w]), out_file, overwrite=True)
        with lock:
            done += 1
            if progress is not None:
                progress(done, len(jobs))

    with ThreadPoolExecutor(max(1, workers)) as pool:
        for _ in tqdm(pool.map(fetch, jobs), total=len(jobs), desc="Downloading JP2 files", unit="file"):
            pass

    return event_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download SDO/AIA and SDO/HMI JPEG2000 (.jp2) images for a UTC date/time range using hvpy."
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
        help="Cadence in seconds between frames (default: 36; HMI channels use at least 45)",
    )
    parser.add_argument(
        "--wavelengths",
        nargs="+",
        default=["304", "171", "193"],
        help="AIA wavelengths and/or HMI channels (continuum, magnetogram) to download (default: 304 171 193)",
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
    download_sdo_jp2(
        start_date=args.start_date,
        end_date=args.end_date,
        wavelengths=args.wavelengths,
        cadence=args.cadence,
        base_dir=args.base_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
