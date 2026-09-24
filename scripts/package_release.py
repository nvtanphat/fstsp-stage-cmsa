"""Create clean release ZIP for FSTSP paper reproduction."""
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ZIP = ROOT / "FSTSP_PAPER_REPRODUCTION_FINAL.zip"

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    ".kaggle",
}

EXCLUDE_EXTS = {
    ".pyc",
    ".pyo",
}

EXCLUDE_FILES = {
    "scratch_paper_text.txt",
    "scratch_sec2.txt",
    "scratch_sec34.txt",
    "FSTSP_PAPER_REPRODUCTION_FINAL.zip",
}

def should_include(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return False
    if path.name in EXCLUDE_FILES:
        return False
    if path.suffix in EXCLUDE_EXTS:
        return False
    return True

def make_zip():
    print(f"Creating {OUTPUT_ZIP.name} from {ROOT}...")
    count = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in ROOT.rglob("*"):
            if file_path.is_file() and should_include(file_path):
                rel_path = file_path.relative_to(ROOT)
                zf.write(file_path, arcname=str(rel_path))
                count += 1
    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"Added {count} files. Total ZIP size: {size_mb:.2f} MB")

if __name__ == "__main__":
    make_zip()
