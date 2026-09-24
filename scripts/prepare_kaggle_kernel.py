from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a self-contained Kaggle script-kernel folder")
    ap.add_argument("--username", required=True)
    ap.add_argument("--slug", default="fstsp-stage-cmsa-reimplementation")
    ap.add_argument("--private", action="store_true")
    ap.add_argument(
        "--mode",
        choices=[
            "synthetic", "agatz",
            "paper_table1", "paper_table2", "paper_tables_1_and_2",
            "paper_table3", "paper_table4", "paper_tables_3_and_4",
            "paper_full"
        ],
        default="synthetic"
    )
    ap.add_argument("--dataset-source", action="append", default=[])
    ap.add_argument("--kernel-source", action="append", default=[])
    ap.add_argument("--sizes", default="6,10")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--total-time", type=float, default=45.0)
    ap.add_argument("--mip-time", type=float, default=8.0)
    ap.add_argument("--age-limit", type=int, default=2)
    ap.add_argument("--max-instances", type=int, default=6)
    ap.add_argument("--exact-time-limit", type=float, default=45.0)
    ap.add_argument("--instances-per-setting", type=int, default=3)
    ap.add_argument("--table3-sizes", default="20,30,40,50")
    ap.add_argument("--table3-seeds", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--resume", action="store_true", default=True, help="Resume from existing checkpoints if available")
    ap.add_argument("--no-resume", action="store_false", dest="resume", help="Force recomputation from scratch")
    args = ap.parse_args()

    if args.mode in ("agatz", "paper_table1", "paper_table2", "paper_tables_1_and_2", "paper_full") and not args.dataset_source:
        raise SystemExit(f"--mode {args.mode} requires at least one --dataset-source owner/dataset-slug")

    out = ROOT / "dist" / "kaggle_kernel"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(ROOT / "src" / "fstsp", out / "fstsp")
    if (ROOT / "configs").exists():
        shutil.copytree(ROOT / "configs", out / "configs")
    settings = {
        "mode": args.mode,
        "sizes": _parse_int_list(args.sizes),
        "seeds": _parse_int_list(args.seeds),
        "total_time": args.total_time,
        "mip_time": args.mip_time,
        "age_limit": args.age_limit,
        "max_instances": args.max_instances,
        "exact_time_limit": args.exact_time_limit,
        "instances_per_setting": args.instances_per_setting,
        "table3_sizes": _parse_int_list(args.table3_sizes),
        "table3_seeds": _parse_int_list(args.table3_seeds),
        "resume": args.resume,
    }
    (out / "experiment-settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    import base64
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for py_file in (ROOT / "src" / "fstsp").rglob("*.py"):
            arcname = py_file.relative_to(ROOT / "src").as_posix()
            zf.write(py_file, arcname)
        if (ROOT / "configs").exists():
            for cfg_file in (ROOT / "configs").rglob("*.*"):
                arcname = cfg_file.relative_to(ROOT).as_posix()
                zf.write(cfg_file, arcname)
    fstsp_bundle_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    bootstrap = (
        "from __future__ import annotations\n\n"
        "# --- AUTO-GENERATED SELF-CONTAINED FSTSP BUNDLE BOOTSTRAP ---\n"
        "import base64\n"
        "import io\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "import tempfile\n"
        "import zipfile\n"
        "from pathlib import Path\n\n"
        f"_EMBEDDED_SETTINGS = {repr(settings)}\n"
        f'_FSTSP_ZIP_B64 = "{fstsp_bundle_b64}"\n\n'
        "try:\n"
        "    import fstsp\n"
        "except ImportError:\n"
        '    _bundle_dir = Path(tempfile.gettempdir()) / "_fstsp_extracted_bundle"\n'
        '    if not (_bundle_dir / "fstsp").exists():\n'
        "        _bundle_dir.mkdir(parents=True, exist_ok=True)\n"
        "        with zipfile.ZipFile(io.BytesIO(base64.b64decode(_FSTSP_ZIP_B64))) as _zf:\n"
        "            _zf.extractall(_bundle_dir)\n"
        "    if str(_bundle_dir) not in sys.path:\n"
        "        sys.path.insert(0, str(_bundle_dir))\n"
        "# --- END BOOTSTRAP ---\n\n"
    )
    original_code = (ROOT / "kaggle" / "run_experiment.py").read_text(encoding="utf-8")
    original_code = original_code.replace("from __future__ import annotations\n", "")
    (out / "run_experiment.py").write_text(bootstrap + original_code, encoding="utf-8")

    metadata = {
        "id": f"{args.username}/{args.slug}",
        "title": "FSTSP Stage CMSA Reimplementation",
        "code_file": "run_experiment.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true" if args.private else "false",
        "enable_gpu": "false",
        "enable_internet": "false",
        "dataset_sources": args.dataset_source,
        "competition_sources": [],
        "kernel_sources": args.kernel_source,
        "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
