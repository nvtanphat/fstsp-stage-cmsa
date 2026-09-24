from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/pcbouman-eur/TSP-D-Instances/archive/refs/heads/master.zip"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/external/agatz")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "TSP-D-Instances-master.zip"
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out)
    print(f"Extracted to {out}")

if __name__ == "__main__":
    main()
