"""
Fast version of Dryad downloader using parallel threads and aria2c fallback.
Run from the top-level of nejm-brain-to-text repository:

    conda activate b2txt25
    python download_data_fast.py
"""

import os
import sys
import json
import urllib.request
import zipfile
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

########################################################################################
# Helper utilities
########################################################################################

def display_progress_bar(block_num, block_size, total_size, message=""):
    bytes_downloaded = block_num * block_size
    MB_downloaded = bytes_downloaded / 1e6
    MB_total = total_size / 1e6 if total_size > 0 else 0
    sys.stdout.write(f"\r{message}\t{MB_downloaded:.1f} MB / {MB_total:.1f} MB")
    sys.stdout.flush()


def aria2c_download(url, dest_path, max_conn=16):
    """Try to download using aria2c for high speed."""
    try:
        subprocess.run(
            [
                "aria2c",
                "-x", str(max_conn),
                "-s", str(max_conn),
                "-k", "1M",
                "-o", os.path.basename(dest_path),
                "-d", os.path.dirname(dest_path),
                "--allow-overwrite=true",
                "--quiet=false",
                url,
            ],
            check=True
        )
        return True
    except FileNotFoundError:
        print("aria2c not found — falling back to urllib.")
        return False
    except subprocess.CalledProcessError:
        print(f"aria2c failed for {url}, fallback to urllib.")
        return False


def urllib_download(url, dest_path, message="Downloading"):
    urllib.request.urlretrieve(
        url,
        dest_path,
        reporthook=lambda *args: display_progress_bar(*args, message=message),
    )
    sys.stdout.write("\n")


########################################################################################
# Main logic
########################################################################################

def main():
    DRYAD_DOI = "10.5061/dryad.dncjsxm85"
    DRYAD_ROOT = "https://datadryad.org"

    # Ensure we're in correct repo
    assert os.getcwd().endswith("nejm-brain-to-text"), \
        f"Run this script from the nejm-brain-to-text directory (currently {os.getcwd()})"

    data_dir = os.path.abspath("data")
    os.makedirs(data_dir, exist_ok=True)

    # Fetch dataset version info
    urlified_doi = DRYAD_DOI.replace("/", "%2F")
    versions_url = f"{DRYAD_ROOT}/api/v2/datasets/doi:{urlified_doi}/versions"

    print("Fetching dataset versions...")
    with urllib.request.urlopen(versions_url) as resp:
        versions_info = json.loads(resp.read().decode())

    files_url_path = versions_info["_embedded"]["stash:versions"][-1]["_links"]["stash:files"]["href"]
    files_url = f"{DRYAD_ROOT}{files_url_path}"

    print("Fetching file list...")
    with urllib.request.urlopen(files_url) as resp:
        files_info = json.loads(resp.read().decode())

    file_infos = files_info["_embedded"]["stash:files"]

    # Download worker
    def worker(file_info):
        filename = file_info["path"]
        if filename == "README.md":
            return f"Skipped {filename}"

        url = DRYAD_ROOT + file_info["_links"]["stash:download"]["href"]
        dest_path = os.path.join(data_dir, filename)

        # Download with aria2c or fallback
        if not aria2c_download(url, dest_path):
            urllib_download(url, dest_path, message=f"Downloading {filename}")

        # Unzip if needed
        if file_info["mimeType"] == "application/zip":
            print(f"Extracting {filename} ...")
            with zipfile.ZipFile(dest_path, "r") as zf:
                zf.extractall(data_dir)
        return f"Done {filename}"

    # Parallel downloads
    print("Starting parallel downloads...\n")
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(worker, f): f for f in file_infos}
        for future in as_completed(futures):
            print(future.result())

    print(f"\n✅ All downloads complete. See data in: {data_dir}\n")


if __name__ == "__main__":
    main()