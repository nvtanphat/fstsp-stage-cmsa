# Hướng dẫn nghiên cứu FSTSP Stage-based + CMSA

## 1. Repo này làm gì?

Repo là **reimplementation (tái hiện từ bài báo)** của bài toán FSTSP (Flying Sidekick Traveling Salesman Problem - bài toán giao hàng phối hợp xe tải–drone), gồm:

- 2-index stage-based MILP (mô hình quy hoạch tuyến tính nguyên hỗn hợp theo giai đoạn);
- CMSA - Construct-Merge-Solve & Adapt (Khởi tạo–Gộp–Giải–Thích nghi);
- parser benchmark Agatz;
- kiểm định nghiệm độc lập;
- brute-force oracle (bộ giải vét cạn đối chứng) cho instance rất nhỏ;
- pipeline chạy Kaggle CLI.

Đây **không phải source code gốc của tác giả**.

## 2. “Train” ở đây thực chất là gì?

Bài này không train neural network. Kaggle được dùng như môi trường compute để **solve/benchmark (giải và đánh giá)** bài toán tối ưu. Tải chính là CPU/RAM; GPU không cần.

## 3. Cấu trúc chính

```text
src/fstsp/
  domain/                  Instance / Solution / Sortie
  data/                    generator + parser benchmark
  formulation/             2-index stage-based MILP
  algorithms/tsp/          MTZ + heuristic TSP
  algorithms/cmsa/         CMSA
  evaluation/              schedule validation + brute-force oracle + metrics
  visualization/           vẽ route

experiments/               protocol chạy thí nghiệm
notebooks/eda/             chỉ EDA/phân tích
scripts/                   verify release + Kaggle CLI + download data
artifacts/                 kết quả / log / verification report
docs/                      tài liệu nghiên cứu
```

## 4. Chạy local

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
pytest -q
```

Không bắt buộc `pip install -e .` khi chạy script trong repo; các experiment script đã trỏ `src/`. Nếu muốn import package từ mọi vị trí thì cài editable package.

Exact nhỏ:

```bash
python experiments/run_exact.py --n 6 --time-limit 30
```

CMSA:

```bash
python experiments/run_cmsa.py --n 12 --total-time 60 --mip-time 10 --age-limit 2
```

## 5. Map paper → code

`src/fstsp/formulation/stage_based.py`:

```text
Eq. (1)       objective min d_E
Eq. (2)-(9)   truck routing
Eq. (10)-(15) drone consistency/endurance/non-overlap
Eq. (16)-(26) forcing truck-drone
Eq. (27)-(33) time synchronization
Eq. (34)-(40) strengthening
Eq. (41)-(55) variable domains/bounds
```

Stage trong Python dùng 0-based; stage 1 của paper tương ứng stage 0 trong code.

## 6. Cơ chế bảo đảm nghiệm không “ảo”

Solver trả nghiệm chưa đủ. Repo còn **tái dựng lịch chạy độc lập** từ truck route + drone sorties bằng `evaluation/schedule.py`.

Kiểm tra gồm:

- mỗi customer được phục vụ đúng một lần;
- launch/recovery nằm trên truck route;
- recovery xảy ra sau launch;
- không có hai sortie chồng nhau;
- drone đủ endurance cho quãng bay;
- truck tới điểm rendezvous đủ sớm để drone không vượt endurance;
- objective báo cáo khớp completion time được tái dựng.

Nếu fail, experiment/Kaggle run bị đánh dấu lỗi thay vì âm thầm ghi kết quả.

## 7. Kiểm chứng formulation bằng brute force

Repo có `evaluation/bruteforce.py` để vét cạn instance rất nhỏ. Regression test so objective exact MILP với optimum brute-force độc lập.

Chạy toàn bộ bộ kiểm định release:

```bash
python scripts/verify_release.py
```

Report:

```text
artifacts/verification/release_verification.json
```

Đây là bước nên chạy trước khi commit/tag release hoặc trước khi lấy số liệu báo cáo.

## 8. CMSA

```text
Construct
   ↓
Truck route + drone sorties đã certify
   ↓
Trích component x, phi, A, B
   ↓
Merge + age
   ↓
Fix component inactive
   ↓
Restricted stage-based MILP
   ↓
Adapt age
   ↓
Giữ best certified solution
```

Paper dùng `age_limit=2`, `t_MIP=15s`. Repo cho đổi tham số để smoke test.

## 9. Solver

Paper dùng CPLEX 22.11. Repo mặc định dùng `scipy.optimize.milp` + HiGHS (solver mã nguồn mở) để Kaggle chạy không cần commercial license.

Vì solver khác, không được lấy runtime HiGHS rồi nói là reproduce chính xác runtime CPLEX của paper.

## 10. Dataset

Public benchmark:

```bash
python scripts/download_agatz_data.py
```

Parser hỗ trợ `#MAXFLY` và `#NOVISIT`. Với file không có `#MAXFLY`, repo dùng ngưỡng hữu hạn tương đương 200% maximum pairwise distance thay vì `1e9`, giúp Big-M ổn định số hơn.

Paper không cung cấp exact 40 generated instance files, nên dữ liệu synthetic trong repo chỉ là **dữ liệu tái hiện có seed**, không phải 40 file gốc.

## 11. Kaggle CLI

Cài/đăng nhập:

```bash
pip install kaggle
kaggle auth login
```

Smoke test synthetic:

```bash
python scripts/prepare_kaggle_kernel.py \
  --username TEN_KAGGLE \
  --private \
  --mode synthetic \
  --sizes 6,10 \
  --seeds 1,2

kaggle kernels push -p dist/kaggle_kernel -t 1200
kaggle kernels status TEN_KAGGLE/fstsp-stage-cmsa-reimplementation
kaggle kernels output TEN_KAGGLE/fstsp-stage-cmsa-reimplementation \
  -p artifacts/kaggle_download -o
```

Muốn chạy public benchmark thì upload/attach dataset trên Kaggle, sau đó prepare với:

```bash
python scripts/prepare_kaggle_kernel.py \
  --username TEN_KAGGLE \
  --private \
  --mode agatz \
  --dataset-source TEN_KAGGLE/TEN_DATASET
```

## 12. Notebook

Notebook chỉ dùng cho EDA (khám phá dữ liệu) và phân tích kết quả. Không để formulation/CMSA core logic trong notebook.

## 13. Cách diễn đạt đúng trong báo cáo

Nên viết:

> “Chúng tôi tái hiện 2-index stage-based formulation và kiến trúc CMSA từ bài báo, sử dụng HiGHS làm solver mã nguồn mở, đồng thời bổ sung cơ chế kiểm định nghiệm độc lập và kiểm chứng exact model bằng brute-force trên instance nhỏ.”

Không nên viết:

> “Đây là source code gốc của tác giả” hoặc “đã reproduce 100% mọi con số trong paper”.

Xem thêm `docs/AUDIT_REPORT.md` và `docs/VALIDATION.md`.
