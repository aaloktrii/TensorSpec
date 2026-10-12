"""GUI helpers for sshproxy (MFA short-lived key) login; not used for Daemon / password hosts."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Mapping, Optional

from PySide6.QtWidgets import QMessageBox, QWidget

from tensorspec.core.compute import cluster_paths as cp


def refresh_sshproxy_login(parent: Optional[QWidget], cluster: Mapping[str, Any]) -> bool:
    """Run sshproxy for this cluster, then verify Paramiko can connect.

    Returns True if connection succeeds after the user completes MFA.
    """
    if not cp.uses_sshproxy(cluster):
        QMessageBox.information(
            parent,
            "Not an sshproxy cluster",
            "This cluster does not use sshproxy (auth: sshproxy in its config).\n"
            "Use Test Connection (password / SSH key) instead.",
        )
        return False

    try:
        cmd = cp.sshproxy_command(cluster)
    except Exception as exc:
        QMessageBox.critical(parent, "sshproxy", str(exc))
        return False

    user = cluster.get("user", "")
    key = cp.ssh_key_path(cluster) or os.path.expanduser("~/.ssh/sshproxy")
    cmd_str = " ".join(cmd)

    QMessageBox.information(
        parent,
        "sshproxy Login",
        f"Will run:\n  {cmd_str}\n\n"
        "A browser or Terminal window may open for MFA / OTP.\n"
        "Complete the login, then click OK here to test the connection.",
    )

    launched = False
    if sys.platform == "darwin":
        # Prefer Terminal so OTP prompts work when browser MFA is unavailable.
        escaped = cmd_str.replace("\\", "\\\\").replace('"', '\\"')
        apple = (
            'tell application "Terminal"\n'
            "  activate\n"
            f'  do script "{escaped}; echo; echo sshproxy login finished — you can close this window.; exit"\n'
            "end tell"
        )
        try:
            subprocess.Popen(["osascript", "-e", apple])
            launched = True
        except Exception:
            launched = False

    if not launched:
        try:
            # Non-macOS / fallback: let sshproxy open browser MFA if configured.
            subprocess.Popen(cmd)
            launched = True
        except Exception as exc:
            QMessageBox.critical(parent, "sshproxy", f"Failed to launch sshproxy:\n{exc}")
            return False

    reply = QMessageBox.question(
        parent,
        "Test sshproxy Connection",
        f"Finished MFA for user '{user}'?\n\n"
        f"Key file: {key}\n\n"
        "Click Yes to test Paramiko SSH to this cluster.",
        QMessageBox.Yes | QMessageBox.No,
    )
    if reply != QMessageBox.Yes:
        return False

    try:
        import paramiko

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        cp.ssh_connect(ssh, cluster, timeout=15)
        _, stdout, _ = ssh.exec_command("hostname", timeout=10)
        host_out = stdout.read().decode().strip()
        ssh.close()
        QMessageBox.information(
            parent,
            "sshproxy Connected",
            f"SSH OK to {cluster.get('host')}.\nRemote hostname: {host_out or '(ok)'}",
        )
        return True
    except Exception as exc:
        QMessageBox.critical(
            parent,
            "sshproxy Auth Failed",
            f"Could not connect after sshproxy.\n\n{exc}\n\n"
            "Complete MFA in Terminal/browser, wait a few seconds, then retry.",
        )
        return False
