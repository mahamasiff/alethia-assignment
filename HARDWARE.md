# Hardware and software used for the measurements

All runs were on a free Kaggle notebook, not on the grading machine (RTX 4090, 32 threads, 64 GB RAM).

| | |
|---|---|
| GPUs | 2 × NVIDIA Tesla T4 (16 GB), driver 580.178.04 |
| CPU | 4 vCPUs (Kaggle) |
| Python | 3.13.16 |
| torch / torchvision | 2.11.0+cu130 / 0.26.0+cu130 |
| ultralytics | 8.4.216 |
| av (PyAV) | 19.0.2 |
| pyarrow | 23.0.1 |
| numpy | 2.1.3 |
| opencv-python | 4.14.0.94 |

`requirements.txt` pins Python 3.11-compatible versions for the grader: torch 2.8.0 (CUDA 12.8 build), torchvision 0.23.0, and av 18.1.1 (av 19 needs Python ≥ 3.12).
The other pins match the table.
So the measured runs used newer torch/PyAV builds than the graded run will.

`run.py` settings for all timed runs: the defaults, i.e. FP16, 5 frames per second, batch 32, one worker per GPU with 2 decoder threads each.

## Data covered

Only the FineVideo parquet files listed here were processed. Everything else in the first 50 files (`train-00000` … `train-00049`) was not.

| parquet files | videos | status |
|---|---|---|
| `train-00000-of-01357.parquet`, `train-00001-of-01357.parquet` | 66 (5.22 video-hours) | complete, exit 0, 0 errors; run.py 447.9 s |
| _batch files from `batches.log` are added after the batch run_ | | |
