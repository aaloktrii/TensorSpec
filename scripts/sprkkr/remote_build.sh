#!/usr/bin/env bash
# Build SPR-KKR 9.7 (serial + MPI) on a remote Linux host. Run from your local machine:
#
#   REMOTE_ROOT=/abs/path/on/remote/SPRKKR bash scripts/sprkkr/remote_build.sh user@host
#
# Env overrides:
#   REMOTE_ROOT  REQUIRED, absolute path on the remote   (bins land in $REMOTE_ROOT/bin)
#   TGZ          default SPRKKR/PUB9.7_260427_09h17.tgz (local, uploaded only if remote has no makefile)
#   QE_ENV       default /home/<user>/miniconda3/envs/qe  (conda env with mpif90 + lapack; skipped if absent)
#   JOBS         default 8
#
# Recipe = what worked in the sandbox (gfortran + openmpi): only make.inc edits, no source edits.
set -euo pipefail

HOST="${1:?usage: REMOTE_ROOT=/abs/path remote_build.sh user@host}"
RUSER="${HOST%%@*}"
REMOTE_ROOT="${REMOTE_ROOT:?set REMOTE_ROOT to an absolute path on the remote, e.g. <heavy_root>/SPRKKR}"
TGZ="${TGZ:-SPRKKR/PUB9.7_260427_09h17.tgz}"
QE_ENV="${QE_ENV:-/home/${RUSER}/miniconda3/envs/qe}"
JOBS="${JOBS:-8}"

echo "== target $HOST : $REMOTE_ROOT"
if ! ssh "$HOST" "test -f '$REMOTE_ROOT/makefile'"; then
  echo "== no makefile on remote, uploading $TGZ"
  [ -f "$TGZ" ] || { echo "local tgz not found: $TGZ"; exit 1; }
  ssh "$HOST" "mkdir -p '$REMOTE_ROOT'"
  scp "$TGZ" "$HOST:$REMOTE_ROOT/"
  ssh "$HOST" "cd '$REMOTE_ROOT' && tar xzf '$(basename "$TGZ")'"
fi

ssh "$HOST" REMOTE_ROOT="$REMOTE_ROOT" QE_ENV="$QE_ENV" JOBS="$JOBS" 'bash -s' <<'REMOTE'
set -euo pipefail
cd "$REMOTE_ROOT"
if [ -d "$QE_ENV/bin" ]; then
  export PATH="$QE_ENV/bin:$PATH"
  export LD_LIBRARY_PATH="$QE_ENV/lib:${LD_LIBRARY_PATH:-}"
  echo "== using conda env $QE_ENV"
fi
FC_BIN="$(command -v mpif90 || true)"
[ -n "$FC_BIN" ] || { echo "!! mpif90 not found (module load openmpi? or set QE_ENV)"; exit 2; }
echo "== mpif90 = $FC_BIN"; "$FC_BIN" --version | head -1
MPI_PREFIX="$(dirname "$(dirname "$FC_BIN")")"
INC="-I$MPI_PREFIX/include"
[ -f "$MPI_PREFIX/include/mpif.h" ] || INC="-I$(dirname "$(find / -name mpif.h -path '*openmpi*' 2>/dev/null | head -1)")"
echo "== mpif.h include: $INC"

# LAPACK/BLAS: conda env ships liblapack.so/libblas.so (-> openblas). Only use -lopenblas
# when the dev symlink libopenblas.so exists (libopenblas.so.0 alone is NOT linkable).
LIB="-llapack -lblas"
if [ -e "$MPI_PREFIX/lib/liblapack.so" ] || [ -e "$MPI_PREFIX/lib/libblas.so" ]; then
  LIB="-L$MPI_PREFIX/lib -llapack -lblas"
elif [ -e "$MPI_PREFIX/lib/libopenblas.so" ]; then
  LIB="-L$MPI_PREFIX/lib -lopenblas"
fi
echo "== LIB = $LIB"

mkdir -p bin
cp make.inc_example make.inc
sed -i "s|^BIN = .*|BIN = $REMOTE_ROOT/bin/|" make.inc
sed -i "s|^LIB = .*|LIB = $LIB|" make.inc
sed -i "s|^#INCLUDE = .*|INCLUDE = $INC|" make.inc
sed -i "s|^FC   = mpif90  -c \$(FFLAGS)|FC   = mpif90  -c \$(FFLAGS) \$(INCLUDE)|" make.inc
grep -E "^(BIN|LIB|INCLUDE|FC|LINK|FFLAGS) " make.inc

echo "== make (serial: scf gen spec)"; make -j"$JOBS" scf gen spec 2>&1 | tail -5
echo "== make (mpi: scfmpi specmpi)";  make -j"$JOBS" scfmpi specmpi 2>&1 | tail -5
echo "== bins:"; ls -la bin/
for b in bin/kkrscf9.7 bin/kkrspec9.7 bin/kkrscf9.7MPI bin/kkrspec9.7MPI; do
  [ -x "$b" ] || { echo "!! missing $b"; exit 3; }
  ldd "$b" | grep -i "not found" && { echo "!! $b has unresolved libs"; exit 4; } || true
done
echo "== OK. Put in ~/.tensorspec_clusters.json:  \"paths\": {\"sprkkr_bin\": \"$REMOTE_ROOT/bin\", \"qe_env\": \"$QE_ENV\"}"
REMOTE
