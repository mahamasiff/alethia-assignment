#!/usr/bin/env python3
"""Video indexing at speed.

For every second of every video in a folder of FineVideo parquet files, record
which of {person, car, dog, laptop, cell phone} are present (YOLO11l, conf>=0.4)
and how many people are visible.

    python run.py --input <folder of .parquet> --output results.jsonl

Pipeline (one process per GPU):
    reader thread   : reads parquet row groups, hands out one video at a time
    decoder threads : PyAV decode -> rotate -> (optional) sample frames per second
                      -> resize to the exact letterbox size -> batches
    main thread     : YOLO inference on batches, per-second aggregation, writes JSONL

Per-second rules (must match the grader's reference):
    second N = frames with (pts - pts_of_first_decoded_frame) in [N, N+1)
    object present  <=> detected in at least half of the second's frames
    people          =  median of per-frame person counts, rounded down at .5
"""
import argparse
import io
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
from fractions import Fraction
from pathlib import Path

CLASS_IDS = (0, 2, 16, 63, 67)
CLASS_NAMES = ("person", "car", "dog", "laptop", "cell phone")
CLASS_INDEX = {c: i for i, c in enumerate(CLASS_IDS)}
IMGSZ = 640
CONF = 0.4

# Defaults used when the grader runs `python run.py --input ... --output ...`.
# Change these only after check.py shows the faster setting still passes.
DEFAULT_SAMPLES_PER_SEC = 0  # 0 = every frame
DEFAULT_HALF = False         # FP16 inference


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Sampling + aggregation (also imported by check.py so the offline simulation
# uses exactly the same logic as the real run)
# --------------------------------------------------------------------------
class Sampler:
    """Keep at most k frames per second: the first frame falling in each of k
    equal time slots of the second. k <= 0 keeps every frame. Every second that
    has at least one frame keeps at least one."""

    def __init__(self, k):
        self.k = k
        self.sec = None
        self.slots = set()

    def keep(self, sec, t):
        if self.k <= 0:
            return True
        if sec != self.sec:
            self.sec = sec
            self.slots = set()
        slot = min(int((t - sec) * self.k + 1e-6), self.k - 1)
        if slot in self.slots:
            return False
        self.slots.add(slot)
        return True


def median_floor(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) // 2


class SecondAgg:
    __slots__ = ("n", "present", "people")

    def __init__(self):
        self.n = 0
        self.present = [0] * len(CLASS_IDS)
        self.people = []

    def add(self, counts):
        self.n += 1
        for i, c in enumerate(counts):
            if c:
                self.present[i] += 1
        self.people.append(counts[0])


def video_lines(vid, secs, last_sec):
    """One dict per second 0..last_sec; seconds without frames are empty."""
    out = []
    for s in range(last_sec + 1):
        a = secs.get(s)
        if a is None or a.n == 0:
            objs, ppl = [], 0
        else:
            objs = [CLASS_NAMES[i] for i in range(len(CLASS_IDS)) if 2 * a.present[i] >= a.n]
            ppl = median_floor(a.people)
        out.append({"video": vid, "second": s, "objects": objs, "people": ppl})
    return out


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------
def letterbox_size(h, w):
    """(w, h) the Ultralytics LetterBox resizes to before padding."""
    r = min(IMGSZ / h, IMGSZ / w)
    return int(round(w * r)), int(round(h * r))


def rotation_code(frame, stream):
    """cv2 rotate code that makes the frame upright, like ffmpeg autorotate."""
    import cv2

    cw = 0
    r = getattr(frame, "rotation", None)  # PyAV>=14: display-matrix angle, CCW degrees
    if r:
        cw = (-int(round(r))) % 360
    else:
        tag = (stream.metadata or {}).get("rotate")  # legacy tag, clockwise degrees
        if tag:
            try:
                cw = int(tag) % 360
            except ValueError:
                cw = 0
    return {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(cw)


class VideoDecoder:
    """Iterate (sec, t, bgr_upright_full_res) over the kept frames of one video.
    After iteration (or a failure), .last_sec is the second of the last decoded frame
    (-1 if none)."""

    def __init__(self, data, samples_per_sec=0, max_seconds=0, codec_threads=2):
        self.data = data
        self.k = samples_per_sec
        self.max_seconds = max_seconds
        self.codec_threads = codec_threads
        self.last_sec = -1
        self.rot = None

    def __iter__(self):
        import av
        import cv2

        container = av.open(io.BytesIO(self.data), mode="r")
        try:
            stream = container.streams.video[0]
            cc = stream.codec_context
            cc.thread_type = "AUTO"
            cc.thread_count = self.codec_threads
            tb = stream.time_base
            rate = stream.average_rate or stream.guessed_rate or Fraction(30)
            sampler = Sampler(self.k)
            pts0 = None
            idx = 0
            first = True
            for frame in container.decode(stream):
                if frame.pts is not None and tb is not None:
                    if pts0 is None:
                        pts0 = frame.pts
                    t = (frame.pts - pts0) * tb  # Fraction, exact
                else:
                    t = Fraction(idx) / Fraction(rate)
                idx += 1
                if t < 0:
                    t = Fraction(0)
                sec = math.floor(t)
                if self.max_seconds and sec >= self.max_seconds:
                    break
                if sec > self.last_sec:
                    self.last_sec = sec
                tf = float(t)
                if not sampler.keep(sec, tf):
                    continue
                if first:
                    self.rot = rotation_code(frame, stream)
                    first = False
                img = frame.to_ndarray(format="bgr24")
                if self.rot is not None:
                    img = cv2.rotate(img, self.rot)
                yield sec, tf, img
        finally:
            container.close()


def decode_video(vid, data, args, out_q):
    """Decode one video into ('frames', ...) batches, then ('end', ...) or ('error', ...)."""
    import cv2

    dec = VideoDecoder(data, args.samples_per_sec, args.max_seconds, args.codec_threads)
    imgs, meta, shape = [], [], None

    def flush():
        nonlocal imgs, meta
        if imgs:
            out_q.put(("frames", (vid, imgs, meta)))
            imgs, meta = [], []

    try:
        for sec, t, img in dec:
            h, w = img.shape[:2]
            tw, th = letterbox_size(h, w)
            if (w, h) != (tw, th):
                # Same op Ultralytics LetterBox would do; it then skips its own resize.
                img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
            # batches must hold one frame shape so Ultralytics keeps the same
            # (rectangular) letterbox the reference used
            if shape is not None and img.shape != shape:
                flush()
            shape = img.shape
            imgs.append(img)
            meta.append((sec, t))
            if len(imgs) >= args.batch:
                flush()
        flush()
        if dec.last_sec < 0:
            out_q.put(("error", (vid, -1, "no decodable video frames")))
        else:
            out_q.put(("end", (vid, dec.last_sec)))
    except Exception as e:  # damaged video: keep what we have, then an error line
        flush()
        out_q.put(("error", (vid, dec.last_sec, f"decode failed: {type(e).__name__}: {e}")))


# --------------------------------------------------------------------------
# Input listing
# --------------------------------------------------------------------------
def list_inputs(input_dir, max_files=0, max_videos=0):
    """Return [(path, name, row_group or None, row0, nrows)] in global video order.
    row_group None = file unreadable."""
    import pyarrow.parquet as pq

    files = sorted(Path(input_dir).glob("*.parquet"))
    if max_files:
        files = files[:max_files]
    units, total = [], 0
    for p in files:
        if max_videos and total >= max_videos:
            break
        try:
            md = pq.ParquetFile(p).metadata
            row0 = 0
            for rg in range(md.num_row_groups):
                n = md.row_group(rg).num_rows
                if max_videos:
                    n = min(n, max_videos - total)
                if n <= 0:
                    break
                units.append((str(p), p.name, rg, row0, n))
                row0 += md.row_group(rg).num_rows
                total += n
        except Exception as e:
            log(f"cannot open {p.name}: {e}")
            units.append((str(p), p.name, None, 0, 1))
            total += 1
    return units


# --------------------------------------------------------------------------
# Worker (one per GPU)
# --------------------------------------------------------------------------
def reader_thread(units, claimed, video_q, n_decoders):
    """Feed (vid, bytes | Exception) to decoders. `claimed` (shared across GPU
    processes) gives each video to whichever process reaches it first."""
    import pyarrow.parquet as pq

    def claim(g):
        # claim lazily, right before handing the video out, so a process that
        # is ahead does not reserve work it will only reach much later
        if claimed is None:
            return True
        with claimed.get_lock():
            if claimed[g]:
                return False
            claimed[g] = 1
            return True

    def unclaimed(g0, n):
        if claimed is None:
            return True
        with claimed.get_lock():
            return any(not claimed[g] for g in range(g0, g0 + n))

    g0 = 0
    try:
        for path, name, rg, row0, nrows in units:
            base, g0 = g0, g0 + nrows
            if not unclaimed(base, nrows):
                continue
            if rg is None:
                if claim(base):
                    video_q.put((f"{name}:0", RuntimeError("could not open parquet file")))
                continue
            try:
                col = pq.ParquetFile(path).read_row_group(rg, columns=["mp4"]).column("mp4")
            except Exception as e:
                for i in range(nrows):
                    if claim(base + i):
                        video_q.put((f"{name}:{row0 + i}", RuntimeError(f"could not read parquet: {e}")))
                continue
            for i in range(nrows):
                if not claim(base + i):
                    continue
                vid = f"{name}:{row0 + i}"
                try:
                    data = col[i].as_py()
                    if not data:
                        raise RuntimeError("empty mp4 bytes")
                except Exception as e:
                    data = RuntimeError(f"could not read mp4 bytes: {e}")
                video_q.put((vid, data))
            del col
    finally:
        for _ in range(n_decoders):
            video_q.put(None)


def decoder_thread(video_q, out_q, args):
    while True:
        item = video_q.get()
        if item is None:
            break
        vid, data = item
        if isinstance(data, Exception):
            out_q.put(("error", (vid, -1, str(data))))
            continue
        decode_video(vid, data, args, out_q)
    out_q.put(("done", None))


def run_worker(rank, args, units, claimed, part_path, dump_path, device):
    import cv2

    cv2.setNumThreads(1)
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    model = YOLO(args.model)
    half = args.half and device != "cpu"

    video_q = queue.Queue(maxsize=args.decoders * 2)
    out_q = queue.Queue(maxsize=args.decoders * 4)
    threads = [threading.Thread(target=reader_thread,
                                args=(units, claimed, video_q, args.decoders), daemon=True)]
    threads += [threading.Thread(target=decoder_thread, args=(video_q, out_q, args), daemon=True)
                for _ in range(args.decoders)]
    for t in threads:
        t.start()

    states = {}
    n_frames, n_videos, t0 = 0, 0, time.time()
    done = 0
    with open(part_path, "w") as out, (open(dump_path, "w") if dump_path else open(os.devnull, "w")) as dump:
        while done < args.decoders:
            kind, p = out_q.get()
            if kind == "done":
                done += 1
            elif kind == "frames":
                vid, imgs, meta = p
                res = model.predict(imgs, imgsz=IMGSZ, conf=CONF, half=half,
                                    device=device, verbose=False)
                st = states.setdefault(vid, {})
                for (sec, t), r in zip(meta, res):
                    c = [0] * len(CLASS_IDS)
                    for k in r.boxes.cls.int().tolist():
                        j = CLASS_INDEX.get(k)
                        if j is not None:
                            c[j] += 1
                    a = st.get(sec)
                    if a is None:
                        a = st[sec] = SecondAgg()
                    a.add(c)
                    if dump_path:
                        dump.write(json.dumps({"v": vid, "s": sec, "t": t, "c": c}) + "\n")
                n_frames += len(imgs)
            else:
                if kind == "end":
                    vid, last_sec = p
                    err = None
                else:
                    vid, last_sec, err = p
                st = states.pop(vid, {})
                lines = video_lines(vid, st, last_sec)
                if err is not None:
                    lines.append({"video": vid, "error": err})
                    log(f"[gpu{rank}] {vid}: {err}")
                out.write("".join(json.dumps(l) + "\n" for l in lines))
                out.flush()
                if dump_path:
                    dump.write(json.dumps({"v": vid, "last": last_sec}) + "\n")
                n_videos += 1
                el = time.time() - t0
                log(f"[gpu{rank}] videos {n_videos}  frames {n_frames}  "
                    f"{n_frames / max(el, 1e-9):.1f} fps  {el:.0f}s")


def _child(rank, args, units, claimed, part_path, dump_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    run_worker(rank, args, units, claimed, part_path, dump_path, "cuda:0")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of .parquet files")
    ap.add_argument("--output", required=True, help="results .jsonl")
    ap.add_argument("--model", default=str(Path(__file__).with_name("yolo11l.pt")))
    ap.add_argument("--samples-per-sec", type=int, default=DEFAULT_SAMPLES_PER_SEC,
                    help="frames analysed per second of video (0 = all)")
    ap.add_argument("--half", action=argparse.BooleanOptionalAction, default=DEFAULT_HALF)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gpus", type=int, default=-1, help="-1 = all visible")
    ap.add_argument("--decoders", type=int, default=0, help="decoder threads per GPU (0 = auto)")
    ap.add_argument("--codec-threads", type=int, default=2, help="ffmpeg threads per decoder")
    # subset options for testing
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--max-seconds", type=int, default=0, help="only the first N seconds of each video")
    ap.add_argument("--dump-frames", default="", help="prefix for per-frame dumps (for check.py sweep)")
    args = ap.parse_args()

    if not Path(args.model).exists():
        args.model = "yolo11l.pt"  # let Ultralytics resolve/download it (dev only)

    import torch

    ngpu = torch.cuda.device_count() if args.gpus < 0 else args.gpus
    nproc = max(1, ngpu)
    if args.decoders <= 0:
        args.decoders = max(2, (os.cpu_count() or 4) // (nproc * args.codec_threads))

    units = list_inputs(args.input, args.max_files, args.max_videos)
    n_videos = sum(u[4] for u in units)
    log(f"{len(units)} row groups, {n_videos} videos, {nproc} worker(s) x {args.decoders} decoders, "
        f"samples/sec={args.samples_per_sec or 'all'}, half={args.half}")

    out_path = Path(args.output)
    parts = [Path(f"{out_path}.part{r}") for r in range(nproc)]
    dumps = [f"{args.dump_frames}.{r}.jsonl" if args.dump_frames else "" for r in range(nproc)]
    t0 = time.time()

    if nproc == 1:
        run_worker(0, args, units, None, parts[0], dumps[0], "cuda:0" if ngpu else "cpu")
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        claimed = ctx.Array("b", n_videos, lock=True)
        procs = [ctx.Process(target=_child, args=(r, args, units, claimed, parts[r], dumps[r]))
                 for r in range(nproc)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
            if p.exitcode:
                log(f"worker exited with code {p.exitcode}")

    # merge parts; any video no worker finished gets an error line
    seen = set()
    with open(out_path, "w") as out:
        for pp in parts:
            if not pp.exists():
                continue
            with open(pp) as f:
                for line in f:
                    if line.endswith("\n"):
                        out.write(line)
                        seen.add(json.loads(line)["video"])
            pp.unlink()
        for path, name, rg, row0, nrows in units:
            for i in range(1 if rg is None else nrows):
                vid = f"{name}:{row0 + i}"
                if vid not in seen:
                    out.write(json.dumps({"video": vid, "error": "not processed (worker failure)"}) + "\n")
    log(f"done in {time.time() - t0:.1f}s -> {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
