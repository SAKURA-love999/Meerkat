#!/usr/bin/env python
"""Download Egyptian fruit bat metadata CSVs from the Figshare collection.

This downloads only small metadata files such as Annotations.csv and
FileInfo.csv. It does not download the full WAV archive.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


BASE = "https://api.figshare.com/v2"
COLLECTION_ID = 3666502


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as r:
        path.write_bytes(r.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data_bat"))
    parser.add_argument("--collection-id", type=int, default=COLLECTION_ID)
    parser.add_argument("--wanted", default="Annotations.csv,FileInfo.csv,Metadata.pdf")
    args = parser.parse_args()

    wanted = {x.strip().lower() for x in args.wanted.split(",") if x.strip()}
    articles = get_json(f"{BASE}/collections/{args.collection_id}/articles?page_size=100")
    manifest = []
    downloaded = []

    for article in articles:
        article_id = article.get("id")
        detail = get_json(f"{BASE}/articles/{article_id}")
        for file_info in detail.get("files", []):
            name = file_info.get("name", "")
            record = {
                "article_id": article_id,
                "article_title": detail.get("title"),
                "file_id": file_info.get("id"),
                "name": name,
                "size": file_info.get("size"),
                "download_url": file_info.get("download_url"),
            }
            manifest.append(record)
            if name.lower() in wanted:
                out_path = args.output_dir / name
                if not out_path.exists() or out_path.stat().st_size != file_info.get("size"):
                    print(f"downloading {name} -> {out_path}")
                    download(file_info["download_url"], out_path)
                downloaded.append(str(out_path))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figshare_file_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"downloaded": downloaded, "n_files_seen": len(manifest)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
