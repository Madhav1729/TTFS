"""Run commands and move files on the Design Compiler machine (remote over ssh, or local)."""

import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import config


def _remote():
    return bool(config.DC_HOST)


def _target():
    return f"{config.DC_USER}@{config.DC_HOST}" if config.DC_USER else config.DC_HOST


def _ssh_options():
    opts = ["-o", "BatchMode=yes", "-p", config.DC_PORT]
    return opts + (["-i", config.DC_KEY] if config.DC_KEY else [])


def _scp_options():
    opts = ["-q", "-r", "-o", "BatchMode=yes", "-P", config.DC_PORT]
    return opts + (["-i", config.DC_KEY] if config.DC_KEY else [])


def _check_settings():
    if _remote() and not config.DC_WORKDIR:
        raise SystemExit("set TTFS_DC_WORKDIR: a writable directory on the DC machine (see dc.env.example)")


def run(command, check=True):
    _check_settings()
    if _remote():
        cmd = ["ssh", "-n", *_ssh_options(), _target(), command]
    else:
        cmd = ["bash", "-c", command]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed: {command}\n{result.stderr}")
    return result.stdout


def push(local, remote):
    if _remote():
        subprocess.run(["scp", *_scp_options(), str(local), f"{_target()}:{remote}"], check=True)
    else:
        (
            shutil.copy(local, remote)
            if Path(local).is_file()
            else shutil.copytree(local, remote, dirs_exist_ok=True)
        )


def pull_tree(remote_dir, local_dir):
    """Copy a directory's contents from the DC machine into local_dir."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    if not _remote():
        shutil.copytree(remote_dir, local_dir, dirs_exist_ok=True)
        return
    with tempfile.TemporaryFile() as buf:
        cmd = ["ssh", *_ssh_options(), _target(), f"tar czf - -C {remote_dir} ."]
        subprocess.run(cmd, stdout=buf, check=True)
        buf.seek(0)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            tar.extractall(local_dir, filter="data")


def new_run_dir(label):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"{config.DC_WORKDIR}/{stamp}_{label}"
    run(f"mkdir -p {path}")
    return path


def launch(remote_dir, script="run.sh"):
    run(f"cd {remote_dir} || exit 1; setsid nohup bash {script} > {script}.out 2>&1 < /dev/null &")


def is_done(remote_dir):
    return run(f"test -f {remote_dir}/done && echo yes || true").strip() == "yes"
