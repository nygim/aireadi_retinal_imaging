#!/usr/bin/env python3
"""
Script to compare file paths and contents between two folders.
Compares contents for .dcm (DICOM) and .tsv files, ignores .DS_Store files.
"""

import os
import argparse
from typing import Dict, Set, List, Tuple
import pydicom
import pandas as pd


def get_all_files(root_dir: str, ignore_patterns: Set[str] = None) -> Dict[str, str]:
    """
    Recursively get all files in a directory, returning relative paths.

    Args:
        root_dir: Root directory to scan
        ignore_patterns: Set of file names/patterns to ignore (e.g., {'.DS_Store'})

    Returns:
        Dictionary mapping relative paths to absolute paths
    """
    if ignore_patterns is None:
        ignore_patterns = {".DS_Store"}

    files = {}

    for root, dirs, filenames in os.walk(root_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for filename in filenames:
            # Skip ignored files
            if filename in ignore_patterns:
                continue

            abs_path = os.path.join(root, filename)
            rel_path = os.path.relpath(abs_path, root_dir)
            files[rel_path] = abs_path

    return files


def compare_dicom_files(file1: str, file2: str) -> Tuple[bool, List[str]]:
    """
    Compare two DICOM files by comparing their metadata.

    Args:
        file1: Path to first DICOM file
        file2: Path to second DICOM file

    Returns:
        Tuple of (are_equal, list_of_differences)
    """
    differences = []

    try:
        ds1 = pydicom.dcmread(file1)
        ds2 = pydicom.dcmread(file2)

        # Get all tags from both files
        tags1 = set(ds1.keys())
        tags2 = set(ds2.keys())

        # Check for tags only in file1
        only_in_file1 = tags1 - tags2
        if only_in_file1:
            differences.append(f"Tags only in {file1}: {sorted(only_in_file1)}")

        # Check for tags only in file2
        only_in_file2 = tags2 - tags1
        if only_in_file2:
            differences.append(f"Tags only in {file2}: {sorted(only_in_file2)}")

        # Compare common tags
        common_tags = tags1 & tags2
        for tag in sorted(common_tags):
            try:
                val1 = ds1[tag].value
                val2 = ds2[tag].value

                # Handle sequences and complex types
                if isinstance(val1, (list, tuple)) and isinstance(val2, (list, tuple)):
                    if len(val1) != len(val2):
                        differences.append(
                            f"Tag {tag}: different lengths ({len(val1)} vs {len(val2)})"
                        )
                    else:
                        for i, (v1, v2) in enumerate(zip(val1, val2)):
                            if v1 != v2:
                                differences.append(f"Tag {tag}[{i}]: {v1} != {v2}")
                elif val1 != val2:
                    # Skip pixel data comparison (too large and may differ due to compression)
                    if tag.keyword == "PixelData":
                        continue
                    differences.append(f"Tag {tag}: {val1} != {val2}")
            except Exception as e:
                differences.append(f"Error comparing tag {tag}: {e}")

        return len(differences) == 0, differences

    except Exception as e:
        return False, [f"Error reading DICOM files: {e}"]


def compare_tsv_files(file1: str, file2: str) -> Tuple[bool, List[str]]:
    """
    Compare two TSV files by comparing their data.

    Args:
        file1: Path to first TSV file
        file2: Path to second TSV file

    Returns:
        Tuple of (are_equal, list_of_differences)
    """
    differences = []

    try:
        # Read TSV files
        df1 = pd.read_csv(file1, sep="\t", dtype=str, keep_default_na=False)
        df2 = pd.read_csv(file2, sep="\t", dtype=str, keep_default_na=False)

        # Compare shapes
        if df1.shape != df2.shape:
            differences.append(f"Different shapes: {df1.shape} vs {df2.shape}")

        # Compare columns
        if list(df1.columns) != list(df2.columns):
            differences.append(
                f"Different columns: {list(df1.columns)} vs {list(df2.columns)}"
            )

        # Compare data (only if shapes and columns match)
        if df1.shape == df2.shape and list(df1.columns) == list(df2.columns):
            # Find rows that differ
            diff_mask = ~df1.equals(df2)
            if diff_mask.any().any():
                # Find specific differences
                for col in df1.columns:
                    if not df1[col].equals(df2[col]):
                        diff_rows = df1[col] != df2[col]
                        num_diffs = diff_rows.sum()
                        differences.append(f"Column '{col}': {num_diffs} rows differ")
                        # Show first few differences
                        diff_indices = df1[diff_rows].index[:5].tolist()
                        if diff_indices:
                            differences.append(
                                f"  First differences at rows: {diff_indices}"
                            )
        else:
            differences.append("Cannot compare data due to shape/column mismatch")

        return len(differences) == 0, differences

    except Exception as e:
        return False, [f"Error reading TSV files: {e}"]


def compare_folders(folder1: str, folder2: str, verbose: bool = False) -> None:
    """
    Compare two folders: file paths and contents (for .dcm and .tsv files).

    Args:
        folder1: Path to first folder
        folder2: Path to second folder
        verbose: If True, print detailed information
    """
    if not os.path.exists(folder1):
        print(f"Error: Folder 1 does not exist: {folder1}")
        return

    if not os.path.exists(folder2):
        print(f"Error: Folder 2 does not exist: {folder2}")
        return

    print(f"Scanning folder 1: {folder1}")
    files1 = get_all_files(folder1)
    print(f"Found {len(files1)} files in folder 1")

    print(f"Scanning folder 2: {folder2}")
    files2 = get_all_files(folder2)
    print(f"Found {len(files2)} files in folder 2")
    print()

    # Get sets of relative paths
    paths1 = set(files1.keys())
    paths2 = set(files2.keys())

    # Find differences in file paths
    only_in_folder1 = paths1 - paths2
    only_in_folder2 = paths2 - paths1
    common_paths = paths1 & paths2

    print("=" * 80)
    print("FILE PATH COMPARISON")
    print("=" * 80)
    print(f"Files only in folder 1: {len(only_in_folder1)}")
    print(f"Files only in folder 2: {len(only_in_folder2)}")
    print(f"Files in both folders: {len(common_paths)}")
    print()

    if only_in_folder1:
        print(f"\nFiles only in folder 1 ({len(only_in_folder1)}):")
        for path in sorted(only_in_folder1)[:20]:  # Show first 20
            print(f"  {path}")
        if len(only_in_folder1) > 20:
            print(f"  ... and {len(only_in_folder1) - 20} more")

    if only_in_folder2:
        print(f"\nFiles only in folder 2 ({len(only_in_folder2)}):")
        for path in sorted(only_in_folder2)[:20]:  # Show first 20
            print(f"  {path}")
        if len(only_in_folder2) > 20:
            print(f"  ... and {len(only_in_folder2) - 20} more")

    # Compare contents for common files
    print("\n" + "=" * 80)
    print("CONTENT COMPARISON (for .dcm and .tsv files)")
    print("=" * 80)

    dcm_files = [p for p in common_paths if p.lower().endswith(".dcm")]
    tsv_files = [p for p in common_paths if p.lower().endswith(".tsv")]

    print(f"\nComparing {len(dcm_files)} DICOM files...")
    dcm_differences = []
    for rel_path in sorted(dcm_files):
        file1 = files1[rel_path]
        file2 = files2[rel_path]
        are_equal, diffs = compare_dicom_files(file1, file2)
        if not are_equal:
            dcm_differences.append((rel_path, diffs))
            if verbose:
                print(f"\n  Differences in {rel_path}:")
                for diff in diffs:
                    print(f"    {diff}")

    print(f"  DICOM files with differences: {len(dcm_differences)}")
    if dcm_differences and not verbose:
        print("  (Use --verbose to see details)")
        for rel_path, _ in dcm_differences[:10]:  # Show first 10
            print(f"    {rel_path}")
        if len(dcm_differences) > 10:
            print(f"    ... and {len(dcm_differences) - 10} more")

    print(f"\nComparing {len(tsv_files)} TSV files...")
    tsv_differences = []
    for rel_path in sorted(tsv_files):
        file1 = files1[rel_path]
        file2 = files2[rel_path]
        are_equal, diffs = compare_tsv_files(file1, file2)
        if not are_equal:
            tsv_differences.append((rel_path, diffs))
            if verbose:
                print(f"\n  Differences in {rel_path}:")
                for diff in diffs:
                    print(f"    {diff}")

    print(f"  TSV files with differences: {len(tsv_differences)}")
    if tsv_differences and not verbose:
        print("  (Use --verbose to see details)")
        for rel_path, _ in tsv_differences[:10]:  # Show first 10
            print(f"    {rel_path}")
        if len(tsv_differences) > 10:
            print(f"    ... and {len(tsv_differences) - 10} more")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_differences = (
        len(only_in_folder1)
        + len(only_in_folder2)
        + len(dcm_differences)
        + len(tsv_differences)
    )
    if total_differences == 0:
        print("✓ Folders are identical!")
    else:
        print(f"✗ Found {total_differences} differences:")
        print(f"  - {len(only_in_folder1)} files only in folder 1")
        print(f"  - {len(only_in_folder2)} files only in folder 2")
        print(f"  - {len(dcm_differences)} DICOM files with content differences")
        print(f"  - {len(tsv_differences)} TSV files with content differences")


def main():
    folder1 = (
        "C:\\Users\\sanjay\\Downloads\\sample_data\\eidon\\final\\retinal_photography"
    )
    folder2 = "C:\\Users\\sanjay\\Downloads\\retinal_photography"

    compare_folders(folder1, folder2, verbose=True)


if __name__ == "__main__":
    main()
