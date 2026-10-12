#!/bin/bash
# Why the +/- deflector asymmetry? Run on the GPU host (RUN_DIR = chinook_gui_run).
#
# Chinook config A gives 8x more intensity at deflector +10.3 than at -10.3,
# where SPR-KKR gives a nearly symmetric, centre-peaked deflector dependence.
# Either something in the geometry is asymmetric that should not be (a bug on
# our side), or the asymmetry is real and is driven by the beam or the crystal.
#
# Two variants of config A, each run at deflector -10.3 and +10.3:
#   V1  azimuth 90 -> 270 : crystal rotated 180 deg about the surface normal,
#                           beam unchanged.  If the asymmetry FOLLOWS the crystal,
#                           the +/- totals swap.
#   V2  incidence 55 -> -55 : beam moved to the other side of the normal,
#                           crystal unchanged.  If the asymmetry FOLLOWS the beam,
#                           the +/- totals swap.
#
# A swap in either case means the asymmetry is physical. No swap in EITHER case
# means it is ours, because nothing left in the problem distinguishes +d from -d.
set -euo pipefail
cd "${RUN_DIR:?set RUN_DIR to the chinook_gui_run directory}"

PY="${PY:?set PY=/path/to/TensorSpec_env/bin/python on the run host}"

"$PY" - <<'PY'
import json
base = json.load(open("arpes_physics_th55_ph0.json"))
v1 = dict(base); v1["manip_azimuth"] = 270.0
v1["_note"] = "config A with the crystal rotated 180 deg about the normal"
json.dump(v1, open("arpes_physics_symA_az270.json", "w"), indent=2)
v2 = dict(base); v2["incidence_angle"] = -55.0
v2["_note"] = "config A with the beam on the other side of the normal"
json.dump(v2, open("arpes_physics_symA_inc-55.json", "w"), indent=2)
print("wrote arpes_physics_symA_az270.json and arpes_physics_symA_inc-55.json")
PY

for V in symA_az270 symA_inc-55; do
  for D in -10.3 10.3; do
    TAG=$(echo "$D" | sed 's/-/m/; s/\./p/')
    OUT="cube_${V}_defl_${TAG}.npz"
    if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
    echo "=== $V  deflector $D"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" -u chinook_remote_runner.py \
        --tb_file tb_data.npz --physics_file "arpes_physics_${V}.json" \
        --theta_min -10 --theta_max 10 --ntheta 100 \
        --phi_min "$D" --phi_max "$D" --nphi 1 \
        --e_min -1.0 --e_max 0.1 --ne 200 \
        --hv 84 --workf 4.5 --v0 12 --temp 10 \
        --engine grizzly --device cuda --layout full \
        --ngpus 1 --theta_chunk 50 \
        --out_file "$OUT" > "log_${V}_defl_${TAG}.txt" 2>&1
  done
done

echo
"$PY" - <<'PY'
import numpy as np

def tot(f):
    return float(np.asarray(np.load(f, allow_pickle=True)["cube"], float).sum())

rows = [
    ("baseline   A  (az  90, inc  55)", "cube_th55_ph0_defl_m10p3.npz",
     "cube_th55_ph0_defl_10p3.npz"),
    ("V1 crystal 180 (az 270, inc  55)", "cube_symA_az270_defl_m10p3.npz",
     "cube_symA_az270_defl_10p3.npz"),
    ("V2 beam flipped (az  90, inc -55)", "cube_symA_inc-55_defl_m10p3.npz",
     "cube_symA_inc-55_defl_10p3.npz"),
]
print(f"{'':34s} {'defl -10.3':>12s} {'defl +10.3':>12s}   ratio +/-")
base = None
for name, fm, fp in rows:
    m, p = tot(fm), tot(fp)
    r = p / m if m else float("nan")
    if base is None:
        base = r
    print(f"{name:34s} {m:12.4e} {p:12.4e}   {r:8.3f}")

print()
print("baseline ratio > 1 means +10.3 is the bright side.")
print("If V1 or V2 flips the ratio to < 1, the asymmetry is physical and follows")
print("the crystal (V1) or the beam (V2). If BOTH stay > 1, it is a convention")
print("bug on the deflector axis, because nothing else distinguishes +d from -d.")
PY
