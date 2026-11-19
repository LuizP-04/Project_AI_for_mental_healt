"""Check that this machine is ready to run the project safely.

Run it from the repository root:

    uv run python scripts/check_env.py

Each line starts with OK, WARN or FAIL. FAIL means that a privacy rule or a
hard requirement is broken; the script then exits with code 1.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import moodai
from moodai.config import ConfigError, load_config
from moodai.llm.local import LocalLLMError, check_host, connect, require_models, server_version
from moodai.offline import is_offline

REPO_ROOT = Path(__file__).resolve().parent.parent
MIN_PYTHON = (3, 11)
MIN_FREE_DISK_GB = 30
FP16_TEST_SIZE = 2048  # side of the square matrix used in the GPU test (8 MiB in FP16)
# Folder names used by common cloud sync clients.
CLOUD_SYNC_MARKERS = ("onedrive", "dropbox", "google drive", "googledrive", "icloud")

OLLAMA_SERVER_JSON = Path.home() / ".ollama" / "server.json"
# Ollama reads its boolean variables with Go's strconv.ParseBool.
GO_TRUE = {"1", "t", "T", "TRUE", "true", "True"}
GO_FALSE = {"", "0", "f", "F", "FALSE", "false", "False"}
# Ollama settings that would write the analyzed text to logs or files.
OLLAMA_LOGGING_VARIABLES = ("OLLAMA_DEBUG", "OLLAMA_DEBUG_LOG_REQUESTS")

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


def query_gpu() -> tuple[str, int, int, str] | None:
    """Return name, total MiB, used MiB and driver of the first NVIDIA GPU, or None."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    query = "--query-gpu=name,memory.total,memory.used,driver_version"
    try:
        result = subprocess.run(
            [nvidia_smi, query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        first_gpu = result.stdout.strip().splitlines()[0]
        name, total, used, driver = (field.strip() for field in first_gpu.split(","))
        return name, int(total), int(used), driver
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def check_gpu() -> str:
    """Report the NVIDIA GPU and its memory through nvidia-smi."""
    gpu = query_gpu()
    if gpu is None:
        return report(WARN, "No NVIDIA GPU found (nvidia-smi is missing or failed).")
    name, total, used, driver = gpu
    return report(OK, f"GPU {name}: {total} MiB total, {used} MiB in use, driver {driver}")


def check_torch() -> str:
    """Check that PyTorch can use the GPU: run an FP16 matrix product and measure its VRAM."""
    try:
        import torch
    except ImportError:
        return report(WARN, "PyTorch is not installed.")
    if not torch.cuda.is_available():
        return report(WARN, f"PyTorch {torch.__version__} cannot use the GPU (CUDA unavailable).")
    before = query_gpu()
    matrix = torch.randn(FP16_TEST_SIZE, FP16_TEST_SIZE, device="cuda", dtype=torch.float16)
    product = matrix @ matrix
    torch.cuda.synchronize()
    after = query_gpu()
    if not torch.isfinite(product).all():
        return report(FAIL, "The FP16 test on the GPU returned invalid numbers.")
    message = f"PyTorch {torch.__version__} (CUDA {torch.version.cuda}) runs FP16 on the GPU"
    if before and after:
        # Includes the CUDA context, a fixed cost that every model on the GPU also pays.
        message += f"; this test used about {after[2] - before[2]} MiB of VRAM"
    return report(OK, message)


def user_env(name: str) -> str:
    """Value of an environment variable in this process or, on Windows, saved for the user.

    Programs started before a variable was saved (VS Code, for example) do not see it yet,
    so both places are checked.
    """
    value = os.environ.get(name, "")
    if value or sys.platform != "win32":
        return value
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            saved, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return ""
    return str(saved)


def ollama_cloud_disabled() -> bool:
    """True if Ollama's cloud features are off, in ~/.ollama/server.json or by variable."""
    try:
        settings = json.loads(OLLAMA_SERVER_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        settings = {}
    if isinstance(settings, dict) and settings.get("disable_ollama_cloud") is True:
        return True
    return user_env("OLLAMA_NO_CLOUD") in GO_TRUE


def check_ollama_privacy() -> str:
    """Check the Ollama settings that could send the text away or write it to disk."""
    if shutil.which("ollama") is None and not OLLAMA_SERVER_JSON.parent.exists():
        return report(WARN, "Ollama is not installed; the debate between LLMs needs it.")
    problems = []
    if not ollama_cloud_disabled():
        problems.append(f"cloud features are on (set disable_ollama_cloud in {OLLAMA_SERVER_JSON})")
    for name in OLLAMA_LOGGING_VARIABLES:
        if user_env(name) not in GO_FALSE:
            problems.append(f"{name} is set (it writes the analyzed text to logs or files)")
    host = user_env("OLLAMA_HOST")
    if host:
        address = host if "://" in host else f"http://{host}"
        if address.startswith("http://:"):  # ":11434" means 127.0.0.1:11434 for Ollama
            address = address.replace("http://:", "http://127.0.0.1:", 1)
        try:
            check_host(address)
        except LocalLLMError:
            problems.append(f"OLLAMA_HOST={host} makes Ollama reachable from other machines")
    if problems:
        return report(FAIL, "Ollama: " + "; ".join(problems) + ".")
    return report(OK, "Ollama privacy: cloud features off, no text logging, this machine only")


def check_ollama_server(llm: object) -> str:
    """Check that the configured Ollama server runs on this machine and has the models."""
    if not isinstance(llm, dict) or not llm.get("host"):
        return report(FAIL, "The configuration has no 'llm' section with a host.")
    try:
        check_host(llm["host"])
    except LocalLLMError as error:
        return report(FAIL, str(error))
    try:
        version = server_version(llm["host"], timeout_s=5)
        require_models(connect(llm["host"], timeout_s=10), llm.get("models", []))
    except LocalLLMError as error:
        return report(WARN, str(error))
    message = f"Ollama {version} at {llm['host']} with {', '.join(llm.get('models', []))}"
    expected = str(llm.get("ollama_version", ""))
    if expected and version != expected:
        return report(WARN, f"{message}; the project was measured with Ollama {expected}")
    return report(OK, message)


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
        config = None
        results.append(report(FAIL, str(error)))
    if config is not None:
        results.append(check_data_paths_ignored(config.get("paths")))
    results += [check_gpu(), check_torch(), check_ollama_privacy()]
    if config is not None:
        results.append(check_ollama_server(config.get("llm")))
    results += [check_disk(), check_long_paths()]

    failures, warnings = results.count(FAIL), results.count(WARN)
    print(f"\n{failures} failure(s), {warnings} warning(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
