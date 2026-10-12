#!/bin/bash
# Status of the VTe2 Chinook/GrizzlyME 4x5 matrix. Run on the run host:
#   ssh <host> 'RUN_DIR=<heavy_root>/chinook_gui_run <heavy_root>/chinook_gui_run/status.sh'
cd "${RUN_DIR:?set RUN_DIR to the chinook_gui_run directory}" || exit 1

DONE=$(ls -1 cube_*.npz 2>/dev/null | grep -v probe | wc -l | tr -d ' ')
echo "cubes finished : $DONE / 20"

if pgrep -f chinook_remote_runner >/dev/null 2>&1; then
  echo "state          : RUNNING"
  ps -eo etime,pcpu,args 2>/dev/null | grep -v grep | grep chinook_remote_runner | head -1 \
    | awk '{printf "  current cut  : elapsed %s   cpu %s%%\n", $1, $2}'
  grep "^=== " matrix.log 2>/dev/null | tail -1 | sed 's/^/  working on   : /'
elif [ "$DONE" -ge 20 ]; then
  echo "state          : FINISHED  -- all 20 cubes present"
else
  echo "state          : STOPPED at $DONE/20  <-- something ended it, see the tail below"
fi

python3 - <<'PY'
import glob, json, time
w = []
for f in glob.glob("cube_*.timing.jsonl"):
    if "probe" in f:
        continue
    for line in open(f):
        try:
            w.append(float(json.loads(line)["wall_s"]))
        except Exception:
            pass
if w:
    avg = sum(w) / len(w)
    print(f"per-cut wall   : mean {avg/3600:.2f} h   (min {min(w)/3600:.2f}, "
          f"max {max(w)/3600:.2f}, n={len(w)})")
    left = 20 - len(w)
    if left > 0:
        eta = time.time() + left * avg
        print(f"remaining      : {left} cuts, ~{left*avg/3600:.1f} h")
        print(f"projected end  : {time.strftime('%a %d %b %H:%M', time.localtime(eta))}")
    else:
        print(f"total          : {sum(w)/3600:.1f} h for all 20")
else:
    print("per-cut wall   : no cut has finished yet -- no estimate possible")
PY

echo "--- last 4 lines of matrix.log:"
tail -4 matrix.log 2>/dev/null || echo "  (no matrix.log)"
