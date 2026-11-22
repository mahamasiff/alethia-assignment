#!/usr/bin/env python3
"""Accuracy checks against an every-frame FP32 reference.

  # 1) compare two results files with the grader's metrics
  python check.py compare ref.jsonl cand.jsonl

  # 2) simulate frame sampling (and FP16) offline from per-frame dumps written by
  #    `run.py --dump-frames PREFIX` (every-frame runs). No GPU needed.
  python check.py sweep --ref "ref_frames.*.jsonl" --cand "fp16_frames.*.jsonl" --k 0 1 2 3 5 8

  # 3) check our rotation handling against OpenCV's autorotate on real videos
  python check.py rotation --input /path/to/parquets --max-videos 100

Pass thresholds: object F1 >= 0.95, people exact >= 85%, people within 1 >= 95%.
"""
import argparse
import glob
import json
import os
import sys
import tempfile
from collections import defaultdict

from run import CLASS_NAMES, Sampler, SecondAgg, video_lines

THRESH = {"f1": 0.95, "exact": 0.85, "within1": 0.95}


def load_results(path):
    """-> {vid: {sec: (frozenset(objects), people)}}, {vid: error}"""
    res, errs = defaultdict(dict), {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "error" in d:
                errs[d["video"]] = d["error"]
            else:
                res[d["video"]][d["second"]] = (frozenset(d["objects"]), d["people"])
    return res, errs


def score(ref, cand, quiet=False, clip=False):
    tp = fp = fn = 0
    n = exact = within1 = 0
    per_class = {c: [0, 0, 0] for c in CLASS_NAMES}
    missing = 0
    for vid, secs in ref.items():
        csecs = cand.get(vid, {})
        # clip: only score seconds the reference has (reference built with --max-seconds)
        for s in (set(secs) if clip else set(secs) | set(csecs)):
            r_obj, r_ppl = secs.get(s, (frozenset(), None))
            c_obj, c_ppl = csecs.get(s, (frozenset(), None))
            tp += len(r_obj & c_obj)
            fp += len(c_obj - r_obj)
            fn += len(r_obj - c_obj)
            for c in CLASS_NAMES:
                pc = per_class[c]
                pc[0] += (c in r_obj and c in c_obj)
                pc[1] += (c in c_obj and c not in r_obj)
                pc[2] += (c in r_obj and c not in c_obj)
            if r_ppl is None:
                continue
            n += 1
            if c_ppl is None:
                missing += 1
                continue
            exact += (c_ppl == r_ppl)
            within1 += (abs(c_ppl - r_ppl) <= 1)
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    m = {"f1": f1, "exact": exact / n if n else 1.0, "within1": within1 / n if n else 1.0}
    ok = all(m[k] >= THRESH[k] for k in THRESH)
    if not quiet:
        print(f"videos {len(ref)}  seconds {n}  missing seconds in candidate {missing}")
        print(f"object F1 {m['f1']:.4f}  people exact {m['exact']:.4f}  within1 {m['within1']:.4f}"
              f"  -> {'PASS' if ok else 'FAIL'}")
        for c, (a, b, d) in per_class.items():
            f = 2 * a / (2 * a + b + d) if a + b + d else float('nan')
            print(f"   {c:<11} F1 {f:.4f}  (tp {a}, fp {b}, fn {d})")
    return m, ok


def cmd_compare(a):
    ref, ref_err = load_results(a.ref)
    cand, cand_err = load_results(a.cand)
    print(f"errors: ref {len(ref_err)}, cand {len(cand_err)}")
    for v in sorted(set(cand_err) - set(ref_err))[:20]:
        print(f"   cand-only error {v}: {cand_err[v]}")
    _, ok = score(ref, cand, clip=a.clip)
    sys.exit(0 if ok else 1)


def load_frames(pattern):
    """-> {vid: (sorted [(t, sec, counts)], last_sec)}"""
    frames, last = defaultdict(list), {}
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no files match {pattern}")
    for fn in files:
        with open(fn) as f:
            for line in f:
                d = json.loads(line)
                if "last" in d:
                    last[d["v"]] = d["last"]
                else:
                    frames[d["v"]].append((d["t"], d["s"], d["c"]))
    return {v: (sorted(frames.get(v, [])), last[v]) for v in last}


def aggregate(frames, k):
    out = {}
    for vid, (fr, last_sec) in frames.items():
        sm, secs = Sampler(k), {}
        for t, s, c in fr:
            if sm.keep(s, t):
                secs.setdefault(s, SecondAgg()).add(c)
        out[vid] = {d["second"]: (frozenset(d["objects"]), d["people"])
                    for d in video_lines(vid, secs, last_sec)}
    return out


def cmd_sweep(a):
    ref_frames = load_frames(a.ref)
    cand_frames = load_frames(a.cand) if a.cand else ref_frames
    common = set(ref_frames) & set(cand_frames)
    ref_frames = {v: ref_frames[v] for v in common}
    cand_frames = {v: cand_frames[v] for v in common}
    ref = aggregate(ref_frames, 0)
    print(f"{len(common)} videos, {sum(len(v) for v in ref.values())} seconds")
    print(f"{'k':>4} {'F1':>8} {'exact':>8} {'within1':>8}  result")
    for k in a.k:
        m, ok = score(ref, aggregate(cand_frames, k), quiet=True)
        print(f"{k if k else 'all':>4} {m['f1']:8.4f} {m['exact']:8.4f} {m['within1']:8.4f}  "
              f"{'PASS' if ok else 'FAIL'}")


def cmd_rotation(a):
    """Compare first frame of PyAV+our rotation with OpenCV (which autorotates
    like most reference pipelines)."""
    import cv2
    import numpy as np
    import pyarrow.parquet as pq

    from run import VideoDecoder, list_inputs

    n = bad = rotated = 0
    for path, name, rg, row0, nrows in list_inputs(a.input, a.max_files, a.max_videos):
        if rg is None:
            continue
        col = pq.ParquetFile(path).read_row_group(rg, columns=["mp4"]).column("mp4")
        for i in range(nrows):
            data = col[i].as_py()
            try:
                dec = VideoDecoder(data)
                _, _, ours = next(iter(dec))
            except Exception as e:
                print(f"{name}:{row0 + i} pyav failed: {e}")
                continue
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                tf.write(data)
            cap = cv2.VideoCapture(tf.name)
            okr, theirs = cap.read()
            cap.release()
            os.unlink(tf.name)
            if not okr:
                print(f"{name}:{row0 + i} opencv failed")
                continue
            n += 1
            rotated += dec.rot is not None
            same_shape = ours.shape == theirs.shape
            diff = float(np.abs(ours.astype(int) - theirs.astype(int)).mean()) if same_shape else -1
            if not same_shape or diff > 8:
                bad += 1
                print(f"{name}:{row0 + i} MISMATCH ours {ours.shape} rot={dec.rot} "
                      f"opencv {theirs.shape} meandiff {diff:.1f}")
            elif dec.rot is not None:
                print(f"{name}:{row0 + i} rotated OK (rot={dec.rot}, meandiff {diff:.2f})")
    print(f"checked {n}, rotated {rotated}, mismatches {bad}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compare")
    p.add_argument("ref")
    p.add_argument("cand")
    p.add_argument("--clip", action="store_true",
                   help="score only seconds present in ref (ref made with --max-seconds)")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("sweep")
    p.add_argument("--ref", required=True, help="glob of FP32 every-frame dumps")
    p.add_argument("--cand", default="", help="glob of candidate every-frame dumps (e.g. FP16)")
    p.add_argument("--k", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 8, 10])
    p.set_defaults(fn=cmd_sweep)
    p = sub.add_parser("rotation")
    p.add_argument("--input", required=True)
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--max-videos", type=int, default=50)
    p.set_defaults(fn=cmd_rotation)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
