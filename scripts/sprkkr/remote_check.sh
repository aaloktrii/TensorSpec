#!/usr/bin/env bash
# Quick remote health check for SPR-KKR runs. Run from your local machine:
#   REMOTE_ROOT=/abs/path/on/remote/SPRKKR bash scripts/sprkkr/remote_check.sh user@host
# Paste the output back to the planner.
set -uo pipefail
HOST="${1:?usage: REMOTE_ROOT=/abs/path remote_check.sh user@host}"
RUSER="${HOST%%@*}"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to an absolute path on the remote, e.g. <heavy_root>/SPRKKR}"
QE_ENV="${QE_ENV:-/home/${RUSER}/miniconda3/envs/qe}"

echo "== ssh key auth test (no password prompt expected)"
ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" 'echo "ssh ok: $(hostname)"' || { echo "!! key auth failed -> run: ssh-copy-id $HOST"; exit 1; }

ssh "$HOST" REMOTE_ROOT="$REMOTE_ROOT" QE_ENV="$QE_ENV" 'bash -s' <<'REMOTE'
echo "== host: $(hostname)  cores: $(nproc)  load: $(cut -d' ' -f1-3 /proc/loadavg)"
free -g | sed -n 2p
echo "== disk heavy_root:"; df -h "$(dirname "$REMOTE_ROOT")" | tail -1
echo "== SPR-KKR bins in $REMOTE_ROOT/bin:"; ls -la "$REMOTE_ROOT/bin" 2>/dev/null || echo "  (none)"
[ -d "$QE_ENV/bin" ] && { export PATH="$QE_ENV/bin:$PATH"; export LD_LIBRARY_PATH="$QE_ENV/lib:${LD_LIBRARY_PATH:-}"; }
echo "== mpif90: $(command -v mpif90 || echo none)   mpirun: $(command -v mpirun || echo none)"
for b in "$REMOTE_ROOT"/bin/kkrspec9.7 "$REMOTE_ROOT"/bin/kkrspec9.7MPI; do
  [ -x "$b" ] && { echo "== ldd $b:"; ldd "$b" | grep -Ei "not found|lapack|blas|mpi" | head -6; }
done
echo "== slurm? $(command -v sbatch >/dev/null && echo yes || echo no)"
echo "== job dir:"; ls -la "$(dirname "$REMOTE_ROOT")/sprkkr_gui_run" 2>/dev/null | head -5 || true
REMOTE
