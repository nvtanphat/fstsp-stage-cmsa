# Kaggle CLI execution guide — verified against current official CLI docs (Sep 2026)

This project is a CPU optimization workload, not neural-network training.

## 1. Install and authenticate

```bash
pip install kaggle
kaggle --help
kaggle auth login
```

The current official CLI also supports `KAGGLE_API_TOKEN`, `~/.kaggle/access_token`, and legacy `~/.kaggle/kaggle.json` credentials.

## 2. Build a self-contained script-kernel bundle

Synthetic smoke run:

```bash
python scripts/prepare_kaggle_kernel.py \
  --username YOUR_USERNAME \
  --private \
  --mode synthetic \
  --sizes 6,10 \
  --seeds 1,2 \
  --total-time 45 \
  --mip-time 8
```

Generated folder:

```text
dist/kaggle_kernel/
├── kernel-metadata.json
├── experiment-settings.json
├── run_experiment.py
└── fstsp/
```

The metadata uses a Python **script** kernel, CPU only, with internet disabled.

## 3. Push and execute

```bash
kaggle kernels push -p dist/kaggle_kernel -t 1200
```

Current official CLI behavior is to upload the kernel and attempt to run it. The current documented `kernels push` options include `-p/--path`, `-t/--timeout`, and accelerator selection.

## 4. Check status

```bash
kaggle kernels status YOUR_USERNAME/fstsp-stage-cmsa-reimplementation
```

## 5. Download outputs

```bash
kaggle kernels output YOUR_USERNAME/fstsp-stage-cmsa-reimplementation \
  -p artifacts/kaggle_download -o
```

Expected outputs include:

```text
benchmark.csv
run_settings.json
<run>/instance.json
<run>/solution.json
<run>/route.png
```

If any produced solution fails independent schedule certification, the Kaggle script intentionally exits with an error after writing diagnostic artifacts.

## 6. Run with the public Agatz benchmark on Kaggle

The kernel has internet disabled, so attach the benchmark as a Kaggle Dataset.

After downloading the public repository locally, create/upload a Kaggle dataset using the official dataset CLI workflow (`kaggle datasets init/create`) or an existing dataset source.

Then build the kernel with the dataset source attached:

```bash
python scripts/prepare_kaggle_kernel.py \
  --username YOUR_USERNAME \
  --private \
  --mode agatz \
  --dataset-source YOUR_USERNAME/YOUR_DATASET_SLUG \
  --max-instances 6 \
  --total-time 60 \
  --mip-time 10
```

Push as usual:

```bash
kaggle kernels push -p dist/kaggle_kernel -t 1800
```

The Kaggle entry point searches attached input datasets recursively for `maxradius` and `novisit` `.txt` files.

## 7. Windows PowerShell

The convenience wrapper runs the default synthetic smoke test:

```powershell
./scripts/kaggle_run.ps1 -Username YOUR_USERNAME
```

For Agatz mode, call `prepare_kaggle_kernel.py` directly with `--mode agatz` and `--dataset-source` before `kaggle kernels push`.

## Important research note

The supplied paper used CPLEX 22.11. This repository defaults to SciPy/HiGHS for license-free Kaggle execution. Do not present HiGHS runtimes as strict reproductions of the paper's CPLEX timings.
