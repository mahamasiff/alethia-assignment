# Video indexing at speed

`run.py` reads FineVideo parquet files and, for every second of every video, writes which of
person / car / dog / laptop / cell phone are present (YOLO11l, imgsz 640, conf ≥ 0.4) and the people count.

## Install (once, with internet, untimed)

```bash
pip install -r requirements.txt
bash prepare.sh
```

`prepare.sh` downloads the official `yolo11l.pt` (Ultralytics assets release, about 51 MB, no login) into the folder that holds `run.py`.
It skips the download if the file is already there.
It checks the file size and that Ultralytics loads the file as a COCO model, and exits non-zero with a message if anything is wrong.

## Run (timed, offline)

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

`run.py` loads the weights from its own folder by absolute path, whatever the current directory is.
It never downloads anything: if the weights are missing it exits and asks you to run `prepare.sh`.
It sets `YOLO_OFFLINE=1`, so Ultralytics doesn't probe the network either.

## How it works

One process per GPU. In each process:

1. **Reader thread** reads the `mp4` column one parquet row group at a time and hands out videos.
   Videos are claimed one at a time through a shared array, so the GPUs balance their load dynamically.
2. **Decoder threads** (PyAV, in memory) handle each video:
   - apply the container rotation tag, matching ffmpeg's autorotate
   - optionally keep only *k* frames per second
   - resize each frame to the exact size the Ultralytics letterbox would. Ultralytics then skips its own resize, so the pixels are identical but the work moves off the GPU thread.
   - batch frames from one video only, so Ultralytics keeps the same rectangular letterbox as single-video inference
3. **Main thread** runs YOLO on the batches and aggregates per second:
   - second N = frames whose `(pts − pts_first_frame) · time_base` is in [N, N+1)
   - an object is present if it is detected in ≥ half of that second's frames
   - people = median of the per-frame person counts, rounded down at .5
   - every second from 0 up to the last frame's second gets a line; seconds without frames get `[]` and 0

**Failures.** A damaged video keeps the lines for the seconds it did decode, then gets one `error` line.
An unreadable parquet file or row group gets error lines for the affected videos.
Any video that no worker finished gets an error line during the final merge.

## Validation (`check.py`)

- `compare ref.jsonl cand.jsonl`: the grader's metrics (object F1, people exact, people within 1).
- `sweep --ref ... --cand ... --k ...`: uses per-frame dumps (`run.py --dump-frames`) from one every-frame run to simulate any frames-per-second setting offline, without the GPU.
  Simulating FP16 needs one FP16 every-frame dump.
- `rotation --input ...`: checks our upright frames against OpenCV's autorotate.

## Development tools (Kaggle only, not used by the graded run)

- `kaggle_batches.py`: resumable batch driver. For each batch of 3 parquet files it downloads them to `/tmp`, runs `run.py` with default settings, renames `res_NN.jsonl.tmp` to `res_NN.jsonl` only if `run.py` exits 0, and deletes the batch's data.
  It appends one line per batch to `batches.log` (files, videos, video-hours, elapsed), skips batches whose `res_NN.jsonl` exists, and stops at an optional hard cutoff (`--stop-at HH:MM`).
- `merge_results.py`: merges `results.jsonl` and the complete `res_NN.jsonl` files. It refuses to write the result if a video is duplicated or missing, judged against the row counts in `manifest.jsonl`, or if a video's seconds have gaps.

## Results

Hardware and software for these runs: see [HARDWARE.md](HARDWARE.md).

### Accuracy and chosen settings

Reference: FP32, every frame, on the first 40 s of the first 32 videos of `train-00000` (1,270 seconds).
Frame sampling and FP16 were simulated from per-frame dumps (`check.py sweep`).
Thresholds: F1 ≥ 0.95, people exact ≥ 0.85, people within 1 ≥ 0.95.

| frames/sec | FP32: F1 / exact / within 1 | FP16: F1 / exact / within 1 |
|---|---|---|
| all | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 0.9984 / 1.0000 |
| 1 | 0.9369 / 0.8732 / 0.9717 (FAIL) | 0.9369 / 0.8732 / 0.9717 (FAIL) |
| 2 | 0.9584 / 0.9063 / 0.9882 | 0.9584 / 0.9071 / 0.9882 |
| 3 | 0.9699 / 0.9425 / 0.9953 | 0.9699 / 0.9425 / 0.9953 |
| 4 | 0.9794 / 0.9425 / 0.9937 | 0.9794 / 0.9417 / 0.9937 |
| **5** | 0.9825 / 0.9535 / 0.9984 | **0.9819 / 0.9543 / 0.9984** |
| 6 | 0.9822 / 0.9551 / 0.9976 | 0.9822 / 0.9583 / 0.9976 |
| 8 | 0.9893 / 0.9661 / 0.9992 | 0.9899 / 0.9646 / 0.9992 |
| 10 | 0.9899 / 0.9740 / 0.9992 | 0.9910 / 0.9740 / 0.9992 |

**Defaults: FP16, 5 frames per second.** FP16 changes nothing measurable.
5 frames per second keeps a margin of about 0.03 on F1, the tightest metric, for unseen videos.
On the T4s, every-frame inference ran at about 50 fps per GPU in FP32 and 114 fps in FP16.

**Rotation:** 64 videos from `train-00000`/`train-00001` matched OpenCV's autorotated frames, with 0 mismatches.
None of them carries a rotation tag, so the rotation path itself is untested on real data.

### Coverage

Only the parquet files listed here were processed. Everything else in the first 50 files was not.

| batch | parquet files | videos | video-hours | run.py elapsed | status |
|---|---|---|---|---|---|
| initial | `train-00000`, `train-00001` | 66 | _TBD_ | _TBD_ | complete, exit 0 |
| _batches from `batches.log` are added after the batch run_ | | | | | |

**Total covered:** _TBD from `merge_results.py`_.

**Limitation:** with about 4 h of free Kaggle 2×T4, only part of the 50 files was processed, and the FP32 every-frame reference was built on a sample.
