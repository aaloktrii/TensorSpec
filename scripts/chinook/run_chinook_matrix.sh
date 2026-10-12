#!/bin/bash
# Run on the GPU host, inside the chinook_gui_run directory (<heavy_root>/chinook_gui_run).
#
# The VTe2 (-201) 4 x 5 matrix with Chinook / GrizzlyME, matching the SPR-KKR
# point-wise deflector cuts in scratch/sprkkr_e2e/cuts_vte2_th{55,20}_ph{0,90}.
#
# Each cut: 100 slit angles (-10..+10 deg) x 200 energies (-1.0..+0.1 eV) at one
# deflector position. The deflector is the phi axis collapsed to a single value,
# because Chinook has no separate deflector parameter.
#
#   PY=/path/to/TensorSpec_env/bin/python ./run_chinook_matrix.sh
#   ENGINE=chinook ONLY=th55_ph0:0 NTHETA=20 NE=40 SUFFIX=_probe ./run_chinook_matrix.sh
#       -> one small cut on the CPU reference engine, for the chinook-vs-grizzly check
#
set -euo pipefail

PY="${PY:?set PY=/path/to/TensorSpec_env/bin/python on the run host}"
ENGINE="${ENGINE:-grizzly}"      # grizzly | chinook
DEVICE="${DEVICE:-cuda}"         # cuda | cpu
LAYOUT="${LAYOUT:-full}"         # full needs grizzly+cuda; chinook must use slices
NGPUS="${NGPUS:-1}"              # GPUs on a shared host - take one
CORES="${CORES:-20}"             # only used by the slices layout
# theta_chunk MUST be < ntheta: the runner only hoists the GrizzlyME ME shell on its
# theta-chunked branch. With one chunk it silently falls back to chinook's CPU
# datacube, which is where the 2645 s of the first probe went.
THETA_CHUNK="${THETA_CHUNK:-0}"  # 0 = auto (= NTHETA/2 below)
ONLY="${ONLY:-}"                 # e.g. th55_ph0:0  -> just that one cut

NTHETA="${NTHETA:-100}"; THETA_MIN=-10; THETA_MAX=10
NE="${NE:-200}";          E_MIN=-1.0;    E_MAX=0.1
SUFFIX="${SUFFIX:-}"      # e.g. _probe, to keep a small test cut separate
HV=84; WORKF=4.5; V0=12; TEMP=10
if [ "$THETA_CHUNK" -le 0 ] || [ "$THETA_CHUNK" -ge "$NTHETA" ]; then
  THETA_CHUNK=$(( NTHETA / 2 )); [ "$THETA_CHUNK" -lt 1 ] && THETA_CHUNK=1
fi

CONFIGS="th55_ph0 th20_ph0 th55_ph90 th20_ph90"
DEFLECTORS="-10.3 -5 0 5 10.3"

if [ "$ENGINE" = "chinook" ]; then LAYOUT="slices"; fi

echo "python  : $PY"
echo "engine  : $ENGINE / $DEVICE / layout=$LAYOUT / ngpus=$NGPUS / theta_chunk=$THETA_CHUNK (ntheta=$NTHETA)"
"$PY" - <<'PRE' || { echo "!! $PY cannot import grizzly/torch - set PY= to one that can"; exit 1; }
import grizzly, torch
print("  grizzly", getattr(grizzly, "__version__", "(no __version__ attr)"),
      "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available(), torch.cuda.device_count(), "device(s)")
PRE

for CFG in $CONFIGS; do
  PHYS="arpes_physics_${CFG}.json"
  [ -f "$PHYS" ] || { echo "missing $PHYS"; exit 1; }
  for D in $DEFLECTORS; do
    TAG=$(echo "$D" | sed 's/-/m/; s/\./p/')
    NAME="${CFG}_defl_${TAG}"
    if [ -n "$ONLY" ] && [ "$ONLY" != "${CFG}:${D}" ]; then continue; fi
    OUT="cube_${NAME}${SUFFIX}.npz"
    if [ -f "$OUT" ]; then echo "skip $OUT (exists)"; continue; fi

    echo "=== $NAME   deflector $D deg"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" -u chinook_remote_runner.py \
        --tb_file tb_data.npz \
        --physics_file "$PHYS" \
        --theta_min $THETA_MIN --theta_max $THETA_MAX --ntheta $NTHETA \
        --phi_min "$D" --phi_max "$D" --nphi 1 \
        --e_min $E_MIN --e_max $E_MAX --ne $NE \
        --hv $HV --workf $WORKF --v0 $V0 --temp $TEMP \
        --engine "$ENGINE" --device "$DEVICE" --layout "$LAYOUT" \
        --ngpus $NGPUS --cores $CORES --theta_chunk $THETA_CHUNK \
        --out_file "$OUT" 2>&1 | tee "log_${NAME}${SUFFIX}.txt"
  done
done

echo
echo "cubes written:"
ls -lh cube_*.npz 2>/dev/null || echo "  (none)"
echo "timings:"
cat cube_*.timing.jsonl 2>/dev/null || echo "  (none)"
