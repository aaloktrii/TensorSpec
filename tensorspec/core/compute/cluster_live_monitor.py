"""SSH snapshot helpers for TensorSpec live cluster / task manager."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tensorspec.core.compute.cluster_paths import heavy_root, job_dir

# TensorSpec + ab-initio processes we care about in the job list.
_JOB_GREP = (
    "kkrscf|kkrspec|kkrgen|pw\\.x|wannier90|pw2wannier|run_pipeline|"
    "chinook_remote_runner|tb_remote_runner"
)

_REMOTE_SNAPSHOT_BASH = r"""
set +e
USER_NAME="__USER__"
HEAVY="__HEAVY__"

echo '###GPU_SUMMARY###'
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null

echo '###UUID_MAP###'
nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null

echo '###GPU_APPS###'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null

echo '###MY_JOBS###'
ps -u "${USER_NAME}" -o pid=,etimes=,stat=,%cpu=,%mem=,args= 2>/dev/null \
  | grep -E '__JOB_GREP__' | grep -v grep

echo '###GPU_HINT###'
grep -h "Using GPU ids" "${HEAVY}"/*/*.full 2>/dev/null | tail -n 1

echo '###LATEST_LOG###'
ls -t "${HEAVY}"/*/*.{full,out,log} 2>/dev/null | head -n 1
"""


def _infer_job_from_processes(jobs_text: str) -> Optional[str]:
    """Map a running command line to a job_dir key (qe, chinook, sprkkr, tb)."""
    for line in jobs_text.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        cmd = parts[5].lower()
        if "chinook_remote_runner" in cmd or "arpes_map_runner" in cmd:
            return "chinook"
        if "tb_remote_runner" in cmd:
            return "tb"
        if any(x in cmd for x in ("run_pipeline", "pw.x", "pw2wannier", "wannier90")):
            return "qe"
        if any(x in cmd for x in ("kkrscf", "kkrspec", "kkrgen")):
            return "sprkkr"
    return None


def _resolve_latest_log(
    ssh,
    cluster: Dict[str, Any],
    *,
    log_jobs: Optional[Sequence[str]] = None,
    my_jobs_raw: str = "",
    global_fallback: str = "",
) -> str:
    """Pick log file for tail display — suite-specific dirs before global newest."""
    candidates: List[str] = []

    jobs = list(log_jobs) if log_jobs else []
    if not jobs:
        inferred = _infer_job_from_processes(my_jobs_raw)
        if inferred:
            jobs = [inferred]

    find_dirs: List[str] = []
    for job in jobs:
        run_dir = job_dir(cluster, job)
        candidates.extend(
            [
                f"{run_dir}/sys.out.full",
                f"{run_dir}/sys.out",
            ]
        )
        if job == "sprkkr":
            # New SPR-KKR runs (core/dft/sprkkr.workflow) live in per-run subdirs:
            # sprkkr_gui_run/{scf_<ts>,arpes_<ts>,e2e_<ts>/scf,...}/<dataset>_{SCF,ARPES_SPEC}.out
            find_dirs.append(run_dir)

    if candidates or find_dirs:
        quoted = " ".join(f'"{p}"' for p in candidates)
        find_expr = ""
        if find_dirs:
            fd = " ".join(f'"{d}"' for d in find_dirs)
            find_expr = (
                f" $(find {fd} -maxdepth 3 -type f "
                f"\( -name '*_SCF.out' -o -name '*_SPEC.out' -o -name '*.log' \) 2>/dev/null)"
            )
        _, stdout, _ = ssh.exec_command(
            f"ls -t {quoted}{find_expr} 2>/dev/null | head -n 1",
            timeout=8,
        )
        picked = stdout.read().decode(errors="replace").strip()
        if picked:
            return picked

    return global_fallback.strip()


def _classify_job(cmd: str) -> str:
    c = cmd.lower()
    if "tb_remote_runner" in c:
        return "TB bands (remote)"
    if "chinook_remote_runner" in c:
        if "grizzly" in c:
            dev = "CUDA" if "device cuda" in c or "--device=cuda" in c else "CPU"
            return f"ARPES GrizzlyME ({dev})"
        return "ARPES Chinook"
    if "pw.x" in c:
        return "QE pw.x"
    if "pw2wannier" in c:
        return "QE pw2wannier"
    if "wannier90" in c:
        return "Wannier90"
    if "run_pipeline" in c:
        return "QE pipeline"
    if "kkrscf" in c:
        return "SPRKKR scf"
    if "kkrspec" in c:
        return "SPRKKR spec"
    if "kkrgen" in c:
        return "SPRKKR gen"
    return "job"


def _short_cmd(cmd: str, limit: int = 72) -> str:
    cmd = re.sub(r"\s+", " ", cmd.strip())
    if len(cmd) <= limit:
        return cmd
    return cmd[: limit - 3] + "..."


def _parse_section(raw: str, tag: str) -> str:
    marker = f"###{tag}###"
    if marker not in raw:
        return ""
    after = raw.split(marker, 1)[1]
    for other in (
        "GPU_SUMMARY",
        "UUID_MAP",
        "GPU_APPS",
        "MY_JOBS",
        "GPU_HINT",
        "LATEST_LOG",
    ):
        end = f"###{other}###"
        if other != tag and end in after:
            after = after.split(end, 1)[0]
    return after.strip()


def _parse_gpu_apps(
    apps_text: str, uuid_map_text: str, ssh, cluster_user: str
) -> Tuple[List[dict], List[dict]]:
    uuid_to_idx: Dict[str, str] = {}
    for line in uuid_map_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            uuid_to_idx[parts[1]] = parts[0]

    rows: List[dict] = []
    pids: List[str] = []
    for line in apps_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        gpu_uuid, pid, pname, mem = parts[0], parts[1], parts[2], parts[3]
        if not pid.isdigit():
            continue
        pids.append(pid)
        rows.append(
            {
                "gpu": uuid_to_idx.get(gpu_uuid, "?"),
                "pid": pid,
                "process": pname,
                "mem_mib": mem.replace(" MiB", "").strip(),
                "user": "",
            }
        )

    if pids:
        pid_list = ",".join(pids)
        _, stdout, _ = ssh.exec_command(
            f"ps -o user=,pid= -p {pid_list} 2>/dev/null", timeout=8
        )
        user_by_pid: Dict[str, str] = {}
        for line in stdout.read().decode(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            bits = line.split(None, 1)
            if len(bits) == 2:
                user_by_pid[bits[1].strip()] = bits[0].strip()
        for row in rows:
            row["user"] = user_by_pid.get(row["pid"], "?")

    mine = [r for r in rows if r["user"] == cluster_user]
    others = [r for r in rows if r["user"] and r["user"] != cluster_user]
    return mine, others


def _format_gpu_summary(gpu_text: str) -> str:
    if not gpu_text.strip():
        return "[ GPU Status ]\n(no nvidia-smi / no GPUs)\n\n"
    lines = ["[ GPU Status ]"]
    for line in gpu_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            idx, name, util, used, total = parts[:5]
            lines.append(
                f"  GPU {idx}: {name} | util {util} | VRAM {used} / {total}"
            )
        elif line.strip():
            lines.append(f"  {line}")
    lines.append("")
    return "\n".join(lines)


def _format_gpu_processes(title: str, rows: List[dict]) -> str:
    if not rows:
        return f"[ {title} ]\n(none)\n\n"
    by_user: Dict[str, List[dict]] = {}
    for row in rows:
        by_user.setdefault(row["user"] or "?", []).append(row)
    lines = [f"[ {title} ]"]
    for user in sorted(by_user):
        lines.append(f"  -- {user} --")
        for row in sorted(by_user[user], key=lambda r: (r["gpu"], r["pid"])):
            lines.append(
                f"    GPU {row['gpu']} | PID {row['pid']} | "
                f"{row['mem_mib']} MiB | {_short_cmd(row['process'], 56)}"
            )
    lines.append("")
    return "\n".join(lines)


def _format_my_jobs(jobs_text: str, gpu_hint: str) -> Tuple[str, int]:
    max_elapsed = 0
    if not jobs_text.strip():
        body = "(no active TensorSpec / QE / SPRKKR jobs)\n"
    else:
        body_lines = []
        for line in jobs_text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 5)
            if len(parts) < 6:
                continue
            pid, etimes, stat, cpu_p, mem_p, cmd = parts
            kind = _classify_job(cmd)
            body_lines.append(
                f"  {kind}\n"
                f"    PID {pid} | {etimes}s | CPU {cpu_p}% | MEM {mem_p}% | {stat}\n"
                f"    {_short_cmd(cmd, 90)}"
            )
            try:
                max_elapsed = max(max_elapsed, int(etimes))
            except ValueError:
                pass
        body = "\n".join(body_lines) if body_lines else "(no active jobs)\n"

    hint = gpu_hint.strip()
    if hint:
        body += f"\n  (latest log) {hint}\n"
    return "[ Your TensorSpec / DFT Jobs ]\n" + body + "\n", max_elapsed


def fetch_cluster_snapshot(
    ssh,
    cluster: Dict[str, Any],
    *,
    log_jobs: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """One polling cycle: GPUs, per-user GPU procs, jobs, log tail.

    log_jobs: prefer logs under these job dirs (``qe``, ``chinook``, ``sprkkr``, ``tb``).
    When omitted, infer from running processes; else fall back to newest log under heavy_root.
    """
    user = cluster.get("user") or "user"
    heavy = heavy_root(cluster)
    script = (
        _REMOTE_SNAPSHOT_BASH.replace("__USER__", user)
        .replace("__HEAVY__", heavy)
        .replace("__JOB_GREP__", _JOB_GREP)
    )
    _, stdout, stderr = ssh.exec_command(f"bash -s <<'EOSNAP'\n{script}\nEOSNAP", timeout=20)
    raw = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")

    gpu_summary = _parse_section(raw, "GPU_SUMMARY")
    uuid_map = _parse_section(raw, "UUID_MAP")
    gpu_apps = _parse_section(raw, "GPU_APPS")
    my_jobs_raw = _parse_section(raw, "MY_JOBS")
    gpu_hint = _parse_section(raw, "GPU_HINT")
    latest_log_global = _parse_section(raw, "LATEST_LOG")
    latest_log = _resolve_latest_log(
        ssh,
        cluster,
        log_jobs=log_jobs,
        my_jobs_raw=my_jobs_raw,
        global_fallback=latest_log_global,
    )

    my_gpu, other_gpu = _parse_gpu_apps(gpu_apps, uuid_map, ssh, user)

    my_jobs_text, max_elapsed = _format_my_jobs(my_jobs_raw, gpu_hint)

    text_info = ""
    text_info += _format_gpu_summary(gpu_summary)
    text_info += _format_gpu_processes("Your GPU Processes", my_gpu)
    text_info += _format_gpu_processes("Other Users' GPU Processes", other_gpu)
    text_info += my_jobs_text

    _, stdout, _ = ssh.exec_command("who", timeout=5)
    text_info += "[ Active SSH Sessions ]\n" + stdout.read().decode() + "\n"

    _, stdout, _ = ssh.exec_command(
        "ps -eo user,pid,stat,%cpu,%mem,command --sort=-%cpu | head -n 8",
        timeout=5,
    )
    text_info += "[ Top CPU (all users) ]\n" + stdout.read().decode() + "\n"

    full_log_tail = ""
    if latest_log:
        _, stdout, _ = ssh.exec_command(
            f"tail -n 50 {latest_log} 2>/dev/null", timeout=8
        )
        tail = stdout.read().decode(errors="replace")
        full_log_tail = f"=== Latest log: {latest_log} ===\n\n{tail}"
        small = "\n".join(tail.splitlines()[-12:])
        text_info += f"[ Last 12 lines: {latest_log} ]\n{small}\n"

    if err.strip() and not gpu_summary:
        text_info += f"[ monitor stderr ]\n{err.strip()}\n"

    return {
        "text_info": text_info,
        "full_log_tail": full_log_tail,
        "kkr_elapsed_seconds": max_elapsed,
        "dft_jobs_text": my_jobs_text,
        "latest_log": latest_log,
        "my_gpu_count": len({r["gpu"] for r in my_gpu}),
        "other_gpu_users": sorted({r["user"] for r in other_gpu if r["user"]}),
    }
