"""Stop data files from being committed: health data never goes to git (LGPD).

Two ways to run it:

    python scripts/check_no_data.py FILE [FILE ...]   # pre-commit passes the staged files
    python scripts/check_no_data.py --all             # CI checks every file tracked by git

A file is blocked when it sits in a data folder (data/, models/, outputs/), has a
data, weight, image, audio or archive extension, is an environment file, or is
larger than 1 MB. This is the second layer of protection after .gitignore, and it
also catches files added with ``git add -f``. The rules are repository policy, so
they live here and not in configs/. Only the standard library is used.
"""

import argparse
import subprocess
import sys
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent
BLOCKED_DIRS = frozenset({"data", "models", "outputs"})

# File types blocked anywhere in the repository (the same list as .gitignore), by kind.
BLOCKED_TYPES = {
    "data file": "csv tsv jsonl parquet arrow feather pkl pickle npy npz h5 hdf5 xlsx xls",
    "database": "db sqlite sqlite3",
    "model weights": "safetensors bin pt pth ckpt gguf onnx",
    "image or document": "jpg jpeg png tif tiff bmp webp pdf",
    "audio or video": "wav mp3 flac m4a mp4 avi",
    "archive": "zip tar gz 7z rar",
    "notebook (cell outputs may hold dataset text)": "ipynb",
}
EXTENSION_KIND = {
    f".{extension}": kind
    for kind, extensions in BLOCKED_TYPES.items()
    for extension in extensions.split()
}

ALLOWED_NAMES = frozenset({".env.example"})  # empty template, no secrets
SIZE_LIMIT_BYTES = 1_000_000  # 1 MB
SIZE_LIMIT_EXEMPT = frozenset({"uv.lock"})  # the lock file grows with the dependencies


def find_problem(path: str) -> str | None:
    """Return why ``path`` must not be committed, or None if it may be committed."""
    posix = PurePosixPath(path.replace("\\", "/"))
    name = posix.name.lower()
    if name in ALLOWED_NAMES:
        return None
    if posix.parts and posix.parts[0].lower() in BLOCKED_DIRS:
        return f"inside the {posix.parts[0]}/ folder"
    if name in {".env", ".envrc"} or name.startswith(".env.") or name.endswith(".env"):
        return "environment file (may hold secrets)"
    kind = EXTENSION_KIND.get(posix.suffix.lower())
    if kind:
        return f"{kind} ({posix.suffix})"
    file = Path(path)
    if not file.is_absolute():
        file = REPO_ROOT / file
    too_big = file.is_file() and file.stat().st_size > SIZE_LIMIT_BYTES
    if too_big and name not in SIZE_LIMIT_EXEMPT:
        return f"larger than {SIZE_LIMIT_BYTES // 1_000_000} MB"
    return None


def tracked_files() -> list[str]:
    """Return every file tracked by git, relative to the repository root."""
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True
    )
    return [name for name in result.stdout.decode("utf-8").split("\0") if name]


def main(argv: list[str] | None = None) -> int:
    """Check the given files, or every tracked file, and return the exit code."""
    parser = argparse.ArgumentParser(description="Stop data files from being committed.")
    parser.add_argument("files", nargs="*", help="files to check, relative to the repository root")
    parser.add_argument("--all", action="store_true", help="check every file tracked by git")
    args = parser.parse_args(argv)

    files = tracked_files() if args.all else args.files
    blocked = [(file, reason) for file in files if (reason := find_problem(file))]
    for file, reason in blocked:
        print(f"BLOCKED {file}: {reason}")
    if blocked:
        print(f"\n{len(blocked)} file(s) must not be committed: health data never goes to git.")
        return 1
    print(f"OK: {len(files)} file(s) checked, no data files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
