"""
Hybrid fast Dryad downloader for nejm-brain-to-text:

- 大檔案 ( >= threshold MB ) → 交給 aria2c 多連線下載
- 小檔案 / 無檔案大小 → aiohttp + asyncio 並行下載
- 自動 retry + exponential backoff (429 / 5xx / timeout)
- Dryad 檔案清單會 cache 到 data/.dryad_files_cache.json
  - 可用 --offline 只讀 cache，不再打 Dryad API
- 支援 resume (aria2c 的 -c + aiohttp Range)
- tqdm 顯示整體下載進度 (bytes)

執行方式（在 nejm-brain-to-text repo 根目錄）:

    conda activate b2txt25
    python download_data_fast.py

常用選項:

    --max-workers N          # 同時下載幾個檔案 (default: 4)
    --offline                # 用 cache 的檔案清單，不再打 Dryad API
    --force                  # 強制重下，即使檔案看起來已經完整
    --no-extract             # 不自動解壓 zip
    --no-aria2c              # 全部都用 aiohttp，不用 aria2c
    --aria2c-threshold-mb M  # 大於等於 M MB 才用 aria2c (default: 50)
"""

import os
import sys
import json
import asyncio
import argparse
import zipfile
import subprocess
import functools
from dataclasses import dataclass
from typing import Optional, List

import aiohttp
from aiohttp import ClientSession
from tqdm import tqdm


DRYAD_DOI = "10.5061/dryad.dncjsxm85"
DRYAD_ROOT = "https://datadryad.org"


@dataclass
class FileTask:
    filename: str
    url: str
    size: Optional[int]
    mime_type: str


########################################################################################
# Helpers: repo / data dir / cache
########################################################################################

def ensure_in_repo():
    assert os.getcwd().endswith("nejm-brain-to-text"), (
        f"Run this script from the nejm-brain-to-text directory "
        f"(currently {os.getcwd()})"
    )


def get_data_dir() -> str:
    data_dir = os.path.abspath("data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_cache_path(data_dir: str) -> str:
    return os.path.join(data_dir, ".dryad_files_cache.json")


def load_cached_file_infos(cache_path: str) -> List[dict]:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Offline mode requested but cache file not found: {cache_path}\n"
            "Run once online (without --offline) to populate the cache."
        )
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cached_file_infos(cache_path: str, file_infos: List[dict]) -> None:
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(file_infos, f, indent=2)
    print(f"[cache] Saved file list to {cache_path}")


########################################################################################
# Helpers: HTTP metadata with retry
########################################################################################

async def fetch_json_with_retry(
    session: ClientSession,
    url: str,
    *,
    max_retries: int = 5,
    backoff_factor: float = 1.5,
    timeout: float = 30.0,
) -> dict:
    delay = 1.0
    for attempt in range(max_retries):
        try:
            async with session.get(url, timeout=timeout) as resp:
                status = resp.status
                if status in (429,) or (500 <= status < 600):
                    # treat as transient
                    raise aiohttp.ClientError(f"Transient HTTP {status} on {url}")
                resp.raise_for_status()
                return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == max_retries - 1:
                print(f"[error] Failed to fetch {url} after {max_retries} attempts: {e}")
                raise
            else:
                print(f"[warn] Error fetching {url}: {e} — retry in {delay:.1f}s")
                await asyncio.sleep(delay)
                delay *= backoff_factor


async def fetch_file_infos_online(session: ClientSession, doi: str) -> List[dict]:
    urlified_doi = doi.replace("/", "%2F")
    versions_url = f"{DRYAD_ROOT}/api/v2/datasets/doi:{urlified_doi}/versions"

    print("[meta] Fetching dataset versions...")
    versions_info = await fetch_json_with_retry(session, versions_url)

    files_url_path = versions_info["_embedded"]["stash:versions"][-1]["_links"]["stash:files"]["href"]
    files_url = f"{DRYAD_ROOT}{files_url_path}"

    print("[meta] Fetching file list...")
    files_info = await fetch_json_with_retry(session, files_url)
    return files_info["_embedded"]["stash:files"]


def build_tasks_from_infos(file_infos: List[dict]) -> List[FileTask]:
    tasks: List[FileTask] = []
    for info in file_infos:
        filename = info["path"]
        if filename == "README.md":
            continue
        url = DRYAD_ROOT + info["_links"]["stash:download"]["href"]
        size = info.get("size")  # may be None
        mime_type = info.get("mimeType", "")
        tasks.append(FileTask(filename=filename, url=url, size=size, mime_type=mime_type))
    return tasks


########################################################################################
# Download logic: aiohttp stream with retry
########################################################################################

async def stream_download_aiohttp(
    session: ClientSession,
    url: str,
    dest_path: str,
    *,
    resume_from: int,
    progress: tqdm,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
    backoff_factor: float = 1.5,
) -> None:
    attempt = 0
    delay = 1.0

    while attempt < max_retries:
        attempt += 1
        try:
            headers = {}
            mode = "wb"
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"
                mode = "ab"

            async with semaphore:
                async with session.get(url, headers=headers) as resp:
                    status = resp.status

                    if status in (429,) or (500 <= status < 600):
                        raise aiohttp.ClientError(f"Transient HTTP {status} on {url}")

                    resp.raise_for_status()

                    # 如果 server 忽略 Range，我們這裡就從頭覆蓋
                    if resume_from > 0 and status == 200:
                        print(f"[warn] Server ignored Range for {os.path.basename(dest_path)}, restarting from scratch.")
                        resume_from = 0
                        mode = "wb"

                    with open(dest_path, mode) as f:
                        async for chunk in resp.content.iter_chunked(1 << 14):
                            if not chunk:
                                continue
                            f.write(chunk)
                            progress.update(len(chunk))
            # 成功就離開
            return

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == max_retries:
                print(f"[error] Download failed for {url} after {max_retries} attempts: {e}")
                raise
            else:
                print(f"[warn] Download error for {url}: {e} — retry in {delay:.1f}s")
                await asyncio.sleep(delay)
                delay *= backoff_factor


async def download_via_aiohttp(
    session: ClientSession,
    task: FileTask,
    data_dir: str,
    progress: tqdm,
    semaphore: asyncio.Semaphore,
    *,
    force: bool = False,
) -> str:
    dest_path = os.path.join(data_dir, task.filename)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if force and os.path.exists(dest_path):
        os.remove(dest_path)

    resume_from = 0
    if os.path.exists(dest_path) and not force:
        resume_from = os.path.getsize(dest_path)

    await stream_download_aiohttp(
        session,
        task.url,
        dest_path,
        resume_from=resume_from,
        progress=progress,
        semaphore=semaphore,
    )

    # sanity check
    if task.size is not None:
        final_size = os.path.getsize(dest_path)
        if final_size != task.size:
            return f"[warn] {task.filename} downloaded but size mismatch ({final_size} vs {task.size})"
    return f"[done aiohttp] {task.filename}"


########################################################################################
# Download logic: aria2c via thread executor
########################################################################################

async def download_via_aria2c(
    task: FileTask,
    data_dir: str,
    progress: tqdm,
    semaphore: asyncio.Semaphore,
    *,
    force: bool = False,
    max_conn: int = 16,
) -> str:
    if task.size is None:
        raise RuntimeError("aria2c downloader requires known file size.")

    dest_path = os.path.join(data_dir, task.filename)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    existing_before = 0
    if os.path.exists(dest_path):
        if force:
            os.remove(dest_path)
        else:
            existing_before = os.path.getsize(dest_path)

    # aria2c 命令
    cmd = [
        "aria2c",
        "--console-log-level=warn",
        "--summary-interval=0",
        "-x", str(max_conn),
        "-s", str(max_conn),
        "-k", "1M",
        "-o", os.path.basename(dest_path),
        "-d", os.path.dirname(dest_path),
        task.url,
    ]
    if force:
        cmd.append("--allow-overwrite=true")
    else:
        cmd.append("-c")  # continue from partial

    loop = asyncio.get_running_loop()

    async with semaphore:
        try:
            # 在 thread pool 裡同步跑 aria2c，stdout/stderr 直接輸出到 terminal
            run_fn = functools.partial(subprocess.run, cmd, check=True)
            await loop.run_in_executor(None, run_fn)
        except FileNotFoundError:
            raise RuntimeError("aria2c not found. Install aria2c or run with --no-aria2c.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"aria2c failed for {task.filename}: {e}")

    if not os.path.exists(dest_path):
        raise RuntimeError(f"aria2c reported success but file not found: {dest_path}")

    final_size = os.path.getsize(dest_path)
    delta = max(0, final_size - existing_before)
    progress.update(delta)

    if final_size != task.size:
        return f"[warn] {task.filename} via aria2c size mismatch ({final_size} vs {task.size})"
    return f"[done aria2c] {task.filename}"


########################################################################################
# Dispatch: choose aria2c or aiohttp per file
########################################################################################

async def download_one_file(
    session: ClientSession,
    task: FileTask,
    data_dir: str,
    progress: tqdm,
    semaphore: asyncio.Semaphore,
    *,
    use_aria2c: bool,
    force: bool,
) -> str:
    dest_path = os.path.join(data_dir, task.filename)

    # 已完整就跳過
    if not force and os.path.exists(dest_path) and task.size is not None:
        existing_size = os.path.getsize(dest_path)
        if existing_size >= task.size:
            return f"[skip] {task.filename} (already complete)"

    if use_aria2c:
        return await download_via_aria2c(
            task,
            data_dir,
            progress,
            semaphore,
            force=force,
        )
    else:
        return await download_via_aiohttp(
            session,
            task,
            data_dir,
            progress,
            semaphore,
            force=force,
        )


def extract_if_needed(task: FileTask, data_dir: str, *, no_extract: bool = False) -> Optional[str]:
    if no_extract:
        return None
    if task.mime_type == "application/zip" or task.filename.lower().endswith(".zip"):
        dest_path = os.path.join(data_dir, task.filename)
        print(f"[zip] Extracting {task.filename} ...")
        with zipfile.ZipFile(dest_path, "r") as zf:
            zf.extractall(data_dir)
        return f"[unzipped] {task.filename}"
    return None


########################################################################################
# Main async entry
########################################################################################

async def async_main(args: argparse.Namespace) -> None:
    ensure_in_repo()
    data_dir = get_data_dir()
    cache_path = get_cache_path(data_dir)

    # 取得檔案 metadata（線上 or cache）
    if args.offline:
        print("[mode] Offline — using cached file list.")
        file_infos = load_cached_file_infos(cache_path)
    else:
        timeout = aiohttp.ClientTimeout(total=None, connect=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            file_infos = await fetch_file_infos_online(session, DRYAD_DOI)
        save_cached_file_infos(cache_path, file_infos)

    tasks = build_tasks_from_infos(file_infos)
    if not tasks:
        print("[info] No files to download (file list empty).")
        return

    # 計算總 bytes（只算已知 size 的）
    known_sizes = [t.size for t in tasks if t.size is not None]
    total_bytes = sum(known_sizes) if len(known_sizes) == len(tasks) else None

    # 已經完整存在的檔案，預先加進 progress
    already_bytes = 0
    if total_bytes is not None:
        for t in tasks:
            if t.size is None:
                continue
            dest = os.path.join(data_dir, t.filename)
            if os.path.exists(dest):
                sz = os.path.getsize(dest)
                if sz >= t.size:
                    already_bytes += t.size

    print(f"[info] {len(tasks)} files in dataset.")
    if total_bytes is not None:
        print(f"[info] Total known size: {total_bytes / 1e6:.1f} MB")
        if already_bytes:
            print(f"[info] Already present: {already_bytes / 1e6:.1f} MB")

    timeout = aiohttp.ClientTimeout(total=None, connect=30)
    semaphore = asyncio.Semaphore(args.max_workers)
    aria2c_threshold_bytes = args.aria2c_threshold_mb * 1024 * 1024

    async with aiohttp.ClientSession(timeout=timeout) as session:
        with tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            desc="Downloading",
            ascii=True,
        ) as progress:
            if already_bytes and total_bytes is not None:
                progress.update(already_bytes)

            coros = []
            for t in tasks:
                # 是否要給 aria2c 處理這個檔案
                use_aria2c = (
                    (not args.no_aria2c)
                    and (t.size is not None)
                    and (t.size >= aria2c_threshold_bytes)
                )
                coros.append(
                    download_one_file(
                        session=session,
                        task=t,
                        data_dir=data_dir,
                        progress=progress,
                        semaphore=semaphore,
                        use_aria2c=use_aria2c,
                        force=args.force,
                    )
                )

            results = await asyncio.gather(*coros, return_exceptions=True)

    print()
    for t, r in zip(tasks, results):
        if isinstance(r, Exception):
            print(f"[fail] {t.filename}: {r}")
        else:
            print(r)
        msg = extract_if_needed(t, data_dir, no_extract=args.no_extract)
        if msg:
            print(msg)

    print(f"\n✅ All done. See data in: {data_dir}\n")


########################################################################################
# CLI entry
########################################################################################

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid fast Dryad downloader (aria2c + aiohttp)")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum concurrent downloads (default: 4)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached file list; don't call Dryad API.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they appear complete.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Don't automatically extract .zip files.",
    )
    parser.add_argument(
        "--no-aria2c",
        action="store_true",
        help="Disable aria2c and download everything with aiohttp.",
    )
    parser.add_argument(
        "--aria2c-threshold-mb",
        type=int,
        default=50,
        help="Files >= this size (MB) will be downloaded with aria2c if available (default: 50).",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\n[abort] Interrupted by user.")


if __name__ == "__main__":
    main()