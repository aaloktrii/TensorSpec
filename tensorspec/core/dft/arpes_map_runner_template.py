"""DEPRECATED (2026-09-09): replaced by tensorspec.core.dft.sprkkr.workflow (native NT×NP grid). Kept for reference."""
import os, sys, shutil, subprocess, time
import numpy as np
import concurrent.futures

def run_single_angle(idx, theta, phi, base_inp_text, bin_path):
    run_dir = f"run_pt_{idx}"
    os.makedirs(run_dir, exist_ok=True)
    
    # link potential
    if os.path.exists("scf.pot_new"):
        try:
            os.symlink(os.path.abspath("scf.pot_new"), os.path.join(run_dir, "scf.pot_new"))
        except:
            shutil.copy("scf.pot_new", os.path.join(run_dir, "scf.pot_new"))
            
    # rewrite input
    inp_lines = base_inp_text.split("\n")
    for i, line in enumerate(inp_lines):
        if line.strip().startswith("THETA"):
            inp_lines[i] = f"  THETA       = {theta:.4f}"
        elif line.strip().startswith("PHI"):
            inp_lines[i] = f"  PHI         = {phi:.4f}"
            
    with open(os.path.join(run_dir, "sys.inp"), "w") as f:
        f.write("\n".join(inp_lines))
        
    # run sprkkr
    env = os.environ.copy()
    subprocess.run(f"{bin_path} < sys.inp > sys.out", shell=True, cwd=run_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # read results
    spec_out = os.path.join(run_dir, "sys_ARPES_SPEC.out")
    energies, intensities = [], []
    if os.path.exists(spec_out):
        with open(spec_out, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 7 and "E" in parts[0] and "E" in parts[1]:
                    try:
                        energies.append(float(parts[0]))
                        intensities.append(float(parts[1]))
                    except ValueError:
                        pass
    
    shutil.rmtree(run_dir, ignore_errors=True)
    return idx, (energies, intensities)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta_min", type=float, required=True)
    parser.add_argument("--theta_max", type=float, required=True)
    parser.add_argument("--ntheta", type=int, required=True)
    parser.add_argument("--phi_min", type=float, required=True)
    parser.add_argument("--phi_max", type=float, required=True)
    parser.add_argument("--nphi", type=int, required=True)
    parser.add_argument("--bin", type=str, required=True)
    parser.add_argument("--cores", type=int, default=16)
    args = parser.parse_args()
    
    with open("sys.inp", "r") as f:
        base_inp = f.read()
        
    thetas = np.linspace(args.theta_min, args.theta_max, args.ntheta)
    phis = np.linspace(args.phi_min, args.phi_max, args.nphi)
    
    pts = []
    idx = 0
    for y in phis:
        for x in thetas:
            pts.append((idx, x, y))
            idx += 1
            
    print(f"Starting map of {len(pts)} points using {args.cores} cores...", flush=True)
    
    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.cores) as exec:
        futs = [exec.submit(run_single_angle, p[0], p[1], p[2], base_inp, args.bin) for p in pts]
        for i, fut in enumerate(concurrent.futures.as_completed(futs)):
            idx, res = fut.result()
            results[idx] = res
            if i % 10 == 0:
                print(f"Progress: {i}/{len(pts)} points...", flush=True)
                
    print("Gathering data...", flush=True)
    
    # find valid energy axis
    e_axis = None
    for idx in range(len(pts)):
        if results.get(idx):
            e_axis = np.array(results[idx][0])
            break
            
    if e_axis is None:
        print("ERROR: No valid data found!", flush=True)
        sys.exit(1)
        
    ne = len(e_axis)
    cube = np.zeros((args.ntheta, args.nphi, ne))
    
    idx = 0
    for j in range(args.nphi):
        for i in range(args.ntheta):
            res = results.get(idx)
            if res is not None and len(res[1]) == ne:
                cube[i, j, :] = res[1]
            idx += 1
            
    np.savez("arpes_cube.npz", intensity=cube, e_axis=e_axis, kx=thetas, ky=phis)
    print("Done! Saved to arpes_cube.npz", flush=True)

if __name__ == "__main__":
    main()
