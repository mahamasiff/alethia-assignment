#!/usr/bin/env bash
# One-time, untimed preparation (needs internet). Run after `pip install -r requirements.txt`:
#     bash prepare.sh
# Downloads the official YOLO11l weights next to run.py and checks them, so the
# timed run.py never needs the network.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # folder of this script (= folder of run.py)
WEIGHTS="$HERE/yolo11l.pt"
# Official Ultralytics release asset (public, no login). YOLO11 weights live under tag v8.3.0.
URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt"
MIN_BYTES=$((40 * 1024 * 1024))   # yolo11l.pt is ~49 MiB (~51 MB)
MAX_BYTES=$((70 * 1024 * 1024))
PY="${PYTHON:-python3}"

size_of() { stat -c %s "$1" 2>/dev/null || wc -c < "$1"; }

if [[ -f "$WEIGHTS" ]]; then
    echo "prepare: $WEIGHTS already present, skipping download"
else
    echo "prepare: downloading $URL"
    TMP="$WEIGHTS.part"
    rm -f "$TMP"
    # Python is guaranteed on the grader; curl may not be.
    "$PY" - "$URL" "$TMP" <<'EOF'
import sys, urllib.request
url, dst = sys.argv[1], sys.argv[2]
req = urllib.request.Request(url, headers={"User-Agent": "prepare-script"})
with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
    while True:
        b = r.read(1 << 20)
        if not b:
            break
        f.write(b)
EOF
    mv -f "$TMP" "$WEIGHTS"   # only a complete download ever gets the final name
fi

# 1) plausible size
if [[ ! -f "$WEIGHTS" ]]; then
    echo "prepare: ERROR: $WEIGHTS does not exist" >&2
    exit 1
fi
SIZE=$(size_of "$WEIGHTS")
if (( SIZE < MIN_BYTES || SIZE > MAX_BYTES )); then
    echo "prepare: ERROR: $WEIGHTS is $SIZE bytes, expected ~51 MB (40-70 MiB)." \
         "Delete it and rerun: rm '$WEIGHTS' && bash prepare.sh" >&2
    exit 1
fi
echo "prepare: $WEIGHTS ok ($SIZE bytes)"

# 2) it loads as YOLO11l with Ultralytics, from the absolute path, without downloading anything.
# Also creates Ultralytics' settings file now rather than during the timed run.
cd "$HERE"
YOLO_OFFLINE=1 "$PY" - "$WEIGHTS" <<'EOF'
import sys
try:
    from ultralytics import YOLO, __version__
except ImportError as e:
    sys.exit(f"prepare: ERROR: cannot import ultralytics ({e}); run `pip install -r requirements.txt` first")
try:
    m = YOLO(sys.argv[1])
except Exception as e:
    sys.exit(f"prepare: ERROR: {sys.argv[1]} does not load as a YOLO model ({e}); delete it and rerun")
names = m.names
assert names.get(0) == "person" and names.get(67) == "cell phone", names
print(f"prepare: ultralytics {__version__} loaded {sys.argv[1]} ({len(names)} classes)")
EOF
echo "prepare: done"
