#!/bin/bash
set -e
export OMP_NUM_THREADS=1
export PYTHON="/Users/sandyai/Documents/GitHub/TensorSpec_GUI/TensorSpec_env/bin/python"
export PYTHONPATH="/Users/sandyai/Documents/GitHub/TensorSpec_GUI:${PYTHONPATH}"

# ==================================================================
# ADVANCED: HSE HYBRID FUNCTIONAL SWITCH
# By default, this script runs a standard PBE calculation.
# To run HSE, just uncomment (remove the '#') from the python command below.
# ==================================================================
# python -c "for f in ['scf.in','nscf.in']: d=open(f).read(); open(f,'w').write(d.replace('&SYSTEM','&SYSTEM\n    input_dft=\'hse\',\n    nqx1=6, nqx2=6, nqx3=6,'))"

mkdir -p out tmp
export TMPDIR=$(pwd)/tmp
# Edit this script to run specific parts of the pipeline
# pw.x ranks=4; pw2wannier90 ranks=4
echo "=== RELAX ==="
mpirun -np 4 pw.x -in relax.in | tee relax.out
"$PYTHON" -m tensorspec.core.dft.sync_after_relax --out-dir . --template-cif structure_template.cif
echo "=== SCF ==="
mpirun -np 4 pw.x -in scf.in | tee scf.out
mpirun -np 4 pw.x -in nscf.in | tee nscf.out
wannier90.x -pp wannier90
mpirun -np 4 pw2wannier90.x -in pw2wan.in | tee pw2wan.out
wannier90.x wannier90