"""
download_xfund.py
-----------------
Downloads the XFUND v1.0 dataset from GitHub releases.
Each language has a training JSON and test JSON (with document images embedded as base64).

Usage:
    python data/download_xfund.py --output_dir ./xfund_data
"""

import os
import json
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm


XFUND_BASE_URL = "https://github.com/doc-analysis/XFUND/releases/download/v1.0"

LANGUAGES = {
    "zh": ("ZH.train.json", "ZH.val.json"),
    "ja": ("JA.train.json", "JA.val.json"),
    "es": ("ES.train.json", "ES.val.json"),
    "fr": ("FR.train.json", "FR.val.json"),
    "it": ("IT.train.json", "IT.val.json"),
    "de": ("DE.train.json", "DE.val.json"),
    "pt": ("PT.train.json", "PT.val.json"),
}

# FUNSD (English) is part of the benchmark too
FUNSD_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest_path: str) -> bool:
    """Download a file with a progress bar. Returns True on success."""
    try:
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest_path) as t:
            urllib.request.urlretrieve(url, filename=dest_path, reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to download {url}: {e}")
        return False


def verify_xfund_file(path: str) -> bool:
    """Basic structural check for XFUND JSON files."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "documents" in data, "Missing 'documents' key"
        assert len(data["documents"]) > 0, "Empty documents list"
        doc = data["documents"][0]
        assert "document" in doc, "Missing 'document' key in first item"
        print(f"  OK  {path}: {len(data['documents'])} documents verified")
        return True
    except Exception as e:
        print(f"  FAIL  {path} failed verification: {e}")
        return False


def download_xfund(output_dir: str, languages: list = None, force: bool = False):
    """
    Download XFUND dataset for specified languages.

    Args:
        output_dir: Root directory to save data.
        languages: List of language codes, e.g. ['zh', 'ja']. None = all.
        force: Re-download even if file already exists.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    langs_to_download = languages or list(LANGUAGES.keys())
    print(f"\n{'='*60}")
    print(f"XFUND Dataset Downloader - {len(langs_to_download)} language(s)")
    print(f"Output directory: {output_path.resolve()}")
    print(f"{'='*60}\n")

    results = {}

    for lang in langs_to_download:
        if lang not in LANGUAGES:
            print(f"[WARN] Unknown language code: {lang}, skipping.")
            continue

        lang_dir = output_path / lang
        lang_dir.mkdir(exist_ok=True)
        train_file, val_file = LANGUAGES[lang]

        print(f"\n[{lang.upper()}] Downloading...")

        for filename in (train_file, val_file):
            dest = lang_dir / filename
            if dest.exists() and not force:
                print(f"  -> {filename} already exists, skipping (use --force to re-download)")
                results[f"{lang}/{filename}"] = "skipped"
                continue

            url = f"{XFUND_BASE_URL}/{filename}"
            print(f"  -> {filename}")
            success = download_file(url, str(dest))

            if success:
                verified = verify_xfund_file(str(dest))
                results[f"{lang}/{filename}"] = "ok" if verified else "corrupt"
            else:
                results[f"{lang}/{filename}"] = "failed"

    # Summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    ok = sum(1 for v in results.values() if v in ("ok", "skipped"))
    failed = sum(1 for v in results.values() if v == "failed")
    corrupt = sum(1 for v in results.values() if v == "corrupt")
    print(f"  OK / Skipped : {ok}")
    print(f"  Failed       : {failed}")
    print(f"  Corrupt      : {corrupt}")

    # Write manifest
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "version": "1.0",
            "languages": langs_to_download,
            "files": results,
        }, f, indent=2)
    print(f"\nManifest written to: {manifest_path}")

    return results


def build_dataset_index(data_dir: str) -> dict:
    """
    Build a quick-access index of all downloaded XFUND files.

    Returns:
        dict mapping lang -> {train: path, val: path}
    """
    data_path = Path(data_dir)
    index = {}

    for lang, (train_file, val_file) in LANGUAGES.items():
        lang_dir = data_path / lang
        entry = {}
        train_path = lang_dir / train_file
        val_path = lang_dir / val_file

        if train_path.exists():
            entry["train"] = str(train_path)
        if val_path.exists():
            entry["val"] = str(val_path)

        if entry:
            index[lang] = entry

    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download XFUND dataset")
    parser.add_argument("--output_dir", type=str, default="./xfund_data",
                        help="Root directory for downloaded data")
    parser.add_argument("--languages", nargs="+", default=None,
                        help="Language codes to download (default: all). E.g.: --languages zh ja es")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files already exist")
    args = parser.parse_args()

    results = download_xfund(args.output_dir, args.languages, args.force)

    # Print index
    print("\nDataset Index:")
    index = build_dataset_index(args.output_dir)
    for lang, paths in index.items():
        for split, path in paths.items():
            print(f"  {lang}/{split}: {path}")
