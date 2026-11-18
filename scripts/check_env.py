"""Check that this machine is ready to run the project safely.

Run it from the repository root:

    uv run python scripts/check_env.py

Each line starts with OK, WARN or FAIL. FAIL means that a privacy rule or a
hard requirement is broken; the script then exits with code 1.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import moodai
from moodai.config import ConfigError, load_config
from moodai.offline import is_offline

REPO_ROOT = Path(__file__).resolve().parent.parent
MIN_PYTHON = (3, 11)
MIN_FREE_DISK_GB = 30
# Folder names used by common cloud sync clients.
CLOUD_SYNC_MARKERS = ("onedrive", "dropbox", "google drive", "googledrive", "icloud")

OK, WARN, FAIL = "OK", "WARN", "FAIL"


def report(status: str, message: str) -> str:
    """Print one result line and return its status."""
    print(f"[{status:>4}] {message}")
    return status


def check_python() -> str:
    """Check the Python version and that the project's virtual environment is in use."""
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info < MIN_PYTHON:
        return report(FAIL, f"Python {version} is too old; the project needs 3.11 or newer.")
    if Path(sys.prefix).resolve() != (REPO_ROOT / ".venv").resolve():
        return report(WARN, f"Python {version} is not the project's .venv; run it with 'uv run'.")
    return report(OK, f"Python {version} from the project's .venv")


def check_offline() -> str:
    """Check that importing moodai turned on offline mode."""
    if not is_offline():
        return report(FAIL, "Offline mode is off: a library could reach the internet.")
    return report(OK, f"Offline mode is on (moodai {moodai.__version__})")


def find_cloud_sync_folder(path: Path) -> str | None:
    """Return the cloud-synced folder that contains ``path``, or None."""
    resolved = path.resolve()
    for variable in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        root = os.environ.get(variable)
        if root and resolved.is_relative_to(Path(root).resolve()):
            return root
    for part in resolved.parts:
        if any(marker in part.lower() for marker in CLOUD_SYNC_MARKERS):
            return part
    return None


def check_location() -> str:
    """Check that the project is not inside a cloud-synced folder."""
    synced = find_cloud_sync_folder(REPO_ROOT)
    if synced:
        return report(FAIL, f"The project is inside a cloud-synced folder ({synced}); move it.")
    return report(OK, f"Project folder is local, not cloud-synced: {REPO_ROOT}")


def check_data_paths_ignored(paths: object) -> str:
    """Check that git ignores every data folder listed under ``paths`` in the configuration."""
    if not isinstance(paths, dict) or not paths:
        return report(FAIL, "The configuration has no 'paths' section with data folders.")
    git = shutil.which("git")
    if git is None:
        return report(WARN, "git not found; could not check that the data folders are ignored.")
    not_ignored = []
    for folder in paths.values():
        # A made-up file inside the folder: git only compares names, nothing is created.
        probe = f"{str(folder).rstrip('/')}/probe.txt"
        result = subprocess.run([git, "check-ignore", "-q", probe], cwd=REPO_ROOT, check=False)
        if result.returncode == 1:
            not_ignored.append(str(folder))
        elif result.returncode != 0:
            return report(WARN, "git check-ignore failed; could not check the data folders.")
    if not_ignored:
        return report(FAIL, f"Not ignored by git: {', '.join(not_ignored)}. Fix .gitignore.")
    return report(OK, f"Ignored by git: {', '.join(str(folder) for folder in paths.values())}")


def check_gpu() -> str:
    """Report the NVIDIA GPU and its memory through nvidia-smi."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return report(WARN, "nvidia-smi not found; no NVIDIA GPU or driver detected.")
    query = "--query-gpu=name,memory.total,memory.used,driver_version"
    try:
        result = subprocess.run(
            [nvidia_smi, query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        name, total, used, driver = (field.strip() for field in result.stdout.split(","))
        total_mib, used_mib = int(total), int(used)
    except (subprocess.TimeoutExpired, ValueError):
        return report(WARN, "Could not read the GPU through nvidia-smi.")
    return report(OK, f"GPU {name}: {total_mib} MiB total, {used_mib} MiB in use, driver {driver}")


def check_disk() -> str:
    """Check the free disk space on the drive that holds the project."""
    free_gb = shutil.disk_usage(REPO_ROOT).free / 1024**3
    if free_gb < MIN_FREE_DISK_GB:
        return report(WARN, f"Only {free_gb:.0f} GB free; plan for {MIN_FREE_DISK_GB} GB or more.")
    return report(OK, f"{free_gb:.0f} GB of free disk space")


def check_long_paths() -> str:
    """On Windows, check whether paths longer than 260 characters are enabled."""
    if sys.platform != "win32":
        return report(OK, "Long paths: no 260-character limit on this system")
    import winreg

    key_path = r"SYSTEM\CurrentControlSet\Control\FileSystem"
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            enabled, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
    except OSError:
        enabled = 0
    if enabled != 1:
        return report(WARN, "Windows long paths are off; enable them if an install fails.")
    return report(OK, "Windows long paths are on")


def main() -> int:
    """Run every check and return the process exit code."""
    parser = argparse.ArgumentParser(description="Check that this machine is ready.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "default.yaml",
        help="YAML configuration file (default: configs/default.yaml)",
    )
    args = parser.parse_args()

    print("Environment check\n")
    results = [check_python(), check_offline(), check_location()]
    try:
        config = load_config(args.config)
    except ConfigError as error:
        results.append(report(FAIL, str(error)))
    else:
        results.append(check_data_paths_ignored(config.get("paths")))
    results += [check_gpu(), check_disk(), check_long_paths()]

    failures, warnings = results.count(FAIL), results.count(WARN)
    print(f"\n{failures} failure(s), {warnings} warning(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
