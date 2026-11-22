#!/usr/bin/env python3
"""Resumable batch driver for Kaggle (development tool, not part of the graded run).

For each batch of parquet files (default: files 2-4, 5-7, ..., 47-49):
  1. skip it if OUT/res_NN.jsonl already exists (NN = first file index)
  2. download its files from Hugging Face into WORK/NN/data  (token from $HF_TOKEN, never printed)
  3. run `run.py --input WORK/NN/data --output OUT/res_NN.jsonl.tmp` with default settings
  4. rename to res_NN.jsonl only if run.py exited 0
  5. delete WORK/NN
and append one line per batch to OUT/batches.log (files, videos, video-hours, elapsed).
Expected row counts per parquet file go to OUT/manifest.jsonl for merge_results.py.

  python kaggle_batches.py --stop-at 18:20      # optional hard cutoff, Kaggle clock (UTC)

SIGTERM/SIGINT or the cutoff kill the current run.py; its partial output is discarded and the
batch reruns on the next launch.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ID = "HuggingFaceFV/finevideo"


def fname(i):
    return f"train-{i:05d}-of-01357.parquet"


class Runner:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out)
        self.work = Path(a.work)
        self.logp = self.out / "batches.log"
        self.manifest = self.out / "manifest.jsonl"
        self.child = None
        self.tmp = None

    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.logp, "a") as f:
            f.write(line + "\n")

    def kill_child(self):
        if self.child is not None and self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signal.SIGTERM)  # run.py + its GPU workers
                self.child.wait(timeout=30)
            except Exception:
                try:
                    os.killpg(self.child.pid, signal.SIGKILL)
                except Exception:
                    pass

    def discard_tmp(self, nn):
        for p in self.out.glob(f"res_{nn}.jsonl.tmp*"):
            p.unlink(missing_ok=True)

    def on_signal(self, signum, frame):
        self.log(f"stopped by signal {signum}; current batch discarded")
        self.kill_child()
        if self.tmp:
            self.discard_tmp(self.tmp)
        sys.exit(1)

    def manifest_add(self, files):
        import pyarrow.parquet as pq

        known = set()
        if self.manifest.exists():
            known = {json.loads(l)["file"] for l in open(self.manifest)}
        with open(self.manifest, "a") as f:
            for p in files:
                if p.name in known:
                    continue
                try:
                    n = pq.ParquetFile(p).metadata.num_rows
                except Exception:
                    n = None  # unreadable file: run.py writes an error line for it
                f.write(json.dumps({"file": p.name, "rows": n}) + "\n")

    def run(self):
        a = self.a
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "batch_logs").mkdir(exist_ok=True)
        signal.signal(signal.SIGTERM, self.on_signal)
        signal.signal(signal.SIGINT, self.on_signal)

        deadline = None
        if a.stop_at:
            hh, mm = map(int, a.stop_at.split(":"))
            deadline = datetime.now().replace(hour=hh, minute=mm, second=0, microsecond=0)
            if deadline < datetime.now() - timedelta(hours=12):
                deadline += timedelta(days=1)
        for d in a.seed_manifest:
            self.manifest_add(sorted(Path(d).glob("*.parquet")))

        self.log(f"start: files {a.first}-{a.last}, batches of {a.batch}, "
                 f"cutoff {deadline.strftime('%H:%M') if deadline else 'none'}")
        last_batch_s = None
        for b0 in range(a.first, a.last + 1, a.batch):
            idx = list(range(b0, min(b0 + a.batch - 1, a.last) + 1))
            nn = f"{b0:02d}"
            label = f"files {idx[0]}-{idx[-1]} ({fname(idx[0])} .. {fname(idx[-1])})"
            res = self.out / f"res_{nn}.jsonl"
            if res.exists():
                self.log(f"batch {nn} {label}: res_{nn}.jsonl exists, skipped")
                continue
            now = datetime.now()
            if deadline and (now >= deadline or
                             (last_batch_s and now + timedelta(seconds=1.1 * last_batch_s) > deadline)):
                self.log(f"cutoff {deadline:%H:%M}: not enough time for batch {nn}; stopping")
                break

            self.tmp = nn
            self.discard_tmp(nn)
            bdir = self.work / nn
            shutil.rmtree(bdir, ignore_errors=True)
            t0 = time.time()
            try:
                from huggingface_hub import hf_hub_download

                for i in idx:
                    hf_hub_download(REPO_ID, f"data/{fname(i)}", repo_type="dataset",
                                    local_dir=bdir, token=os.environ.get("HF_TOKEN") or None)
            except Exception as e:
                self.log(f"batch {nn} {label}: download failed ({type(e).__name__}); skipped")
                shutil.rmtree(bdir, ignore_errors=True)
                continue
            dl_s = time.time() - t0
            data = bdir / "data"
            self.manifest_add(sorted(data.glob("*.parquet")))

            tmp = self.out / f"res_{nn}.jsonl.tmp"
            cmd = [sys.executable, str(HERE / "run.py"), "--input", str(data), "--output", str(tmp)]
            t1 = time.time()
            with open(self.out / "batch_logs" / f"run_{nn}.log", "w") as lf:
                self.child = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                              start_new_session=True)
                cut = False
                while self.child.poll() is None:
                    if deadline and datetime.now() >= deadline:
                        cut = True
                        self.kill_child()
                        break
                    time.sleep(5)
            run_s = time.time() - t1
            rc = self.child.returncode
            self.child = None

            if cut:
                self.log(f"batch {nn} {label}: cutoff reached after {run_s:.0f}s; partial output discarded")
                self.discard_tmp(nn)
                shutil.rmtree(bdir, ignore_errors=True)
                break
            if rc == 0 and tmp.exists():
                os.replace(tmp, res)
                vids, secs, errs = set(), 0, 0
                for line in open(res):
                    d = json.loads(line)
                    vids.add(d["video"])
                    if "error" in d:
                        errs += 1
                    else:
                        secs += 1
                h = secs / 3600
                self.log(f"batch {nn} {label}: OK videos {len(vids)} video_hours {h:.2f} "
                         f"errors {errs} elapsed {run_s:.0f}s ({h * 3600 / run_s:.0f}x realtime) "
                         f"download {dl_s:.0f}s")
            else:
                self.log(f"batch {nn} {label}: run.py FAILED (exit {rc}) after {run_s:.0f}s; "
                         f"see batch_logs/run_{nn}.log")
                self.discard_tmp(nn)
            shutil.rmtree(bdir, ignore_errors=True)
            self.tmp = None
            last_batch_s = dl_s + run_s
        self.log("finished")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2)
    ap.add_argument("--last", type=int, default=49)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--work", default="/tmp/batches")
    ap.add_argument("--stop-at", default="", help="HH:MM on this machine's clock; hard cutoff")
    ap.add_argument("--seed-manifest", nargs="*", default=[],
                    help="folders of already-processed parquet files to add to the manifest")
    Runner(ap.parse_args()).run()


if __name__ == "__main__":
    main()
