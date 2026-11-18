"""Tests for scripts/check_no_data.py. Every file name here is made up."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_no_data.py"


def run_check(*args):
    """Run the script the way pre-commit or the CI would, and return the finished process."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/posts.txt",
        "models/classifier/config.json",
        "outputs/report.txt",
        "notes.csv",
        "samples/handwriting.png",
        "datasets.tar.gz",
        "exploration.ipynb",
        ".env",
        "prod.env",
        ".env.local",
    ],
)
def test_blocks_data_and_secret_files(path):
    result = run_check(path)
    assert result.returncode == 1
    assert f"BLOCKED {path}" in result.stdout


@pytest.mark.parametrize(
    "path",
    ["src/moodai/config.py", "src/moodai/models/bert.py", "README.md", ".env.example"],
)
def test_allows_code_and_docs(path):
    result = run_check(path)
    assert result.returncode == 0, result.stdout


def test_blocks_files_larger_than_1_mb(tmp_path):
    big_file = tmp_path / "big.txt"
    big_file.write_bytes(b"x" * 1_000_001)
    result = run_check(str(big_file))
    assert result.returncode == 1
    assert "larger than 1 MB" in result.stdout


def test_no_tracked_file_is_blocked():
    result = run_check("--all")
    assert result.returncode == 0, result.stdout
