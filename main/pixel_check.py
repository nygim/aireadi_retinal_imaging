#!/usr/bin/env python3
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Optional

import pydicom
from pydicom.uid import \
    UID  # optional import; not strictly needed, but handy for typing
from tqdm import tqdm


# --------------------- Discovery ---------------------
def get_filtered_all_file_names(folder_path: str) -> List[str]:
    """
    Recursively collect .dcm files, skipping names starting with '._'.
    """
    files = []
    for root, _, names in os.walk(folder_path):
        for name in names:
            if name.endswith(".dcm") and not name.startswith("._"):
                files.append(os.path.join(root, name))
    return files

# --------------------- Probing ---------------------
def _is_compressed_file(path: str) -> bool:
    """
    Fast header-only check to see if a DICOM appears to have compressed PixelData.
    Uses the Transfer Syntax UID property available on pydicom UIDs.
    """
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        if "PixelData" not in ds:
            return False
        ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
        return bool(getattr(ts, "is_compressed", False))
    except Exception:
        # If header can't be read, treat as not compressed for the heuristic.
        return False
# --------------------- Validation ---------------------
def _check_one_file(path: str, decode_pixels: bool) -> Optional[str]:
    """
    Read-only validation of a DICOM file.
    Returns None on success, or an error string on failure.
    """
    try:
        # Fast header read first
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        if "PixelData" not in ds:
            # Not an image; treat as OK unless you want to flag these
            return None

        if decode_pixels:
            # Read full dataset and force pixel decode (read-only)
            ds_full = pydicom.dcmread(path, force=True)
            _ = ds_full.pixel_array  # triggers handlers; stays in memory
        return None

    except Exception as e:
        return f"{path}\t{type(e).__name__}: {e}"

# --------------------- Runner ---------------------
def check_pixel_array(
    folder: str,
    max_workers: Optional[int] = None,
    chunksize: int = 64,
    sample: int = 200,
    decode_pixels: bool = True,
    force_threads: bool = False,
    force_processes: bool = False,
):
    """
    Validate DICOMs under `folder`. Read-only; no changes to original files.

    - If decode_pixels=True, forces pixel decode via .pixel_array (slow but thorough).
    - Auto chooses ThreadPool vs ProcessPool unless forced.
    """
    filelist = get_filtered_all_file_names(folder)
    total = len(filelist)
    if total == 0:
        print("No .dcm files found.")
        return 0

    # Decide executor type
    use_processes = False
    if not force_threads and not force_processes:
        # Heuristic: sample for compression ratio
        sample_files = filelist[: min(sample, total)]
        t0 = time.time()
        comp_count = sum(_is_compressed_file(f) for f in sample_files)
        comp_ratio = comp_count / len(sample_files) if sample_files else 0.0
        t1 = time.time()
        use_processes = comp_ratio >= 0.20 if decode_pixels else False  # processes only help when decoding
        print(
            f"Sampled {len(sample_files)} files in {t1 - t0:.2f}s; "
            f"compressed ≈ {comp_ratio:.0%}. "
            f"Executor: {'ProcessPool' if use_processes else 'ThreadPool'}."
        )
    else:
        use_processes = force_processes and not force_threads

    # Worker count defaults
    if max_workers is None:
        if use_processes:
            max_workers = max(1, (os.cpu_count() or 4))
        else:
            # I/O-bound default: more threads can help hide latency
            max_workers = max(8, 2 * (os.cpu_count() or 4))

    Exec = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    errors: List[str] = []
    # Wrap the callable to pass 'decode_pixels' when using executor.map
    def _runner(path):
        return _check_one_file(path, decode_pixels)

    print(f"Scanning {total} files with {Exec.__name__}(max_workers={max_workers}), chunksize={chunksize}")
    with Exec(max_workers=max_workers) as ex, \
         tqdm(total=total, desc="Checking DICOMs", unit="file", smoothing=0.3) as pbar:
        for result in ex.map(_runner, filelist, chunksize=chunksize):
            if result:
                errors.append(result)
            pbar.update(1)

    if not errors:
        print("✅ All files read successfully.")
    else:
        print(f"❌ {len(errors)} files failed to validate; listing below (tab-separated path and error):")
        for err in errors:
            print(err)

    # Return count of failures as exit code (0 = success)
    return len(errors)

# --------------------- CLI ---------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Read-only DICOM validation (recursive). Decodes pixel data unless --light."
    )
    p.add_argument("folder", help="Root folder to scan")
    p.add_argument("--workers", type=int, default=None, help="Max workers (threads or processes)")
    p.add_argument("--chunksize", type=int, default=64, help="Executor map chunksize (larger reduces overhead)")
    p.add_argument("--sample", type=int, default=200, help="Files to sample for compression heuristic")
    p.add_argument("--light", action="store_true", help="Header-only check (skip pixel decode; much faster)")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--threads", action="store_true", help="Force ThreadPoolExecutor")
    group.add_argument("--processes", action="store_true", help="Force ProcessPoolExecutor")
    return p.parse_args(argv)

def main(argv: List[str]) -> int:
    args = parse_args(argv)
    # NOTE: This script never writes to any DICOM or changes metadata.
    # It only reads files and decodes pixels in memory.
    return check_pixel_array(
        folder=args.folder,
        max_workers=args.workers,
        chunksize=args.chunksize,
        sample=args.sample,
        decode_pixels=not args.light,
        force_threads=args.threads,
        force_processes=args.processes,
    )

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
