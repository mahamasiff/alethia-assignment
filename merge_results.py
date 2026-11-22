#!/usr/bin/env python3
"""Merge results.jsonl + complete res_NN.jsonl batch outputs into one file, with checks.

  python merge_results.py --dir /kaggle/working --out /kaggle/working/results_all.jsonl

Inputs: DIR/results.jsonl and DIR/res_NN.jsonl (never *.tmp). Checks:
  - no video appears in more than one input file
  - no duplicate (video, second) line, at most one error line per video
  - each video's seconds are exactly 0..max
  - every row of every parquet file in DIR/manifest.jsonl has a video (missing rows),
    and every covered file is in the manifest
The merged file is written only if all checks pass (or with --force).
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def ranges(nums):
    nums = sorted(nums)
    out, s = [], None
    for i, n in enumerate(nums):
        if s is None:
            s = n
        if i + 1 == len(nums) or nums[i + 1] != n + 1:
            out.append(f"{s}" if s == n else f"{s}-{n}")
            s = None
    return ", ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/kaggle/working")
    ap.add_argument("--out", default="/kaggle/working/results_all.jsonl")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    d = Path(a.dir)

    inputs = [d / "results.jsonl"] if (d / "results.jsonl").exists() else []
    inputs += sorted(p for p in d.glob("res_*.jsonl") if re.fullmatch(r"res_\d+\.jsonl", p.name))
    if not inputs:
        sys.exit("no inputs found")

    problems = []
    owner = {}                       # video -> input file
    secs = defaultdict(list)         # video -> seconds
    errs = defaultdict(int)
    rows = defaultdict(set)          # parquet file -> row indices seen
    lines_total = 0
    for p in inputs:
        for i, line in enumerate(open(p), 1):
            rec = json.loads(line)
            v = rec["video"]
            if owner.setdefault(v, p.name) != p.name:
                msg = f"DUPLICATE video {v} in {owner[v]} and {p.name}"
                if msg not in problems:
                    problems.append(msg)
                continue
            lines_total += 1
            name, row = v.rsplit(":", 1)
            rows[name].add(int(row))
            if "error" in rec:
                errs[v] += 1
            else:
                secs[v].append(rec["second"])

    dup_reported = set()
    for v, s in secs.items():
        if len(s) != len(set(s)) and v not in dup_reported:
            problems.append(f"DUPLICATE seconds in {v}")
            dup_reported.add(v)
        if sorted(set(s)) != list(range(max(s) + 1)):
            problems.append(f"GAP in seconds of {v} (max {max(s)}, {len(set(s))} distinct)")
    for v, n in errs.items():
        if n > 1:
            problems.append(f"{v} has {n} error lines")

    manifest = {}
    mp = d / "manifest.jsonl"
    if mp.exists():
        for line in open(mp):
            m = json.loads(line)
            manifest[m["file"]] = m["rows"]
    for name, seen in sorted(rows.items()):
        n = manifest.get(name)
        if name not in manifest:
            problems.append(f"{name} not in manifest; can only check rows 0..{max(seen)}")
            n = max(seen) + 1
        if n is None:
            continue  # unreadable file
        missing = sorted(set(range(n)) - seen)
        if missing:
            problems.append(f"MISSING videos in {name}: rows {ranges(missing)}")
    for name in sorted(set(manifest) - set(rows)):
        problems.append(f"MISSING file {name}: in manifest but no output lines")

    files = sorted(rows)
    idx = [int(re.search(r"train-(\d+)-", f).group(1)) for f in files]
    hours = sum(len(set(s)) for s in secs.values()) / 3600
    err_videos = sum(1 for v in owner if errs[v])
    print(f"inputs: {', '.join(p.name for p in inputs)}")
    print(f"files covered: {len(files)} (indices {ranges(idx)})")
    print(f"videos: {len(owner)}   video-hours: {hours:.2f}   videos with error line: {err_videos}"
          f"   lines: {lines_total}")
    if problems:
        print(f"CHECK FAILED ({len(problems)} problems):")
        for p in problems[:50]:
            print("  ", p)
        if not a.force:
            sys.exit(1)
    else:
        print("checks: no duplicate or missing videos, seconds contiguous -> OK")

    with open(a.out, "w") as out:
        for p in inputs:
            with open(p) as f:
                for line in f:
                    out.write(line)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
