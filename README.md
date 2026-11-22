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

## Results

_TODO: hardware (Kaggle 2×T4), files processed, fps, sweep table, chosen defaults._

**Limitation:** with ~4 h of free Kaggle 2×T4, only the first few parquet files were processed,
and the FP32 every-frame reference was built on a sample.
