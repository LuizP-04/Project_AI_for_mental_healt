"""Tests for offline mode and for the network block that guards every test run."""

import os
import socket
import subprocess
import sys

import pytest
from pytest_socket import SocketConnectBlockedError

from moodai.offline import OFFLINE_ENV, enforce_offline, is_offline


def test_importing_moodai_turns_on_offline_mode():
    # Start a fresh Python without the offline variables and only import the package.
    env = {name: value for name, value in os.environ.items() if name not in OFFLINE_ENV}
    code = "import moodai; from moodai.offline import is_offline; print(is_offline())"
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "True"


def test_enforce_offline_overrides_existing_values(monkeypatch):
    for name in OFFLINE_ENV:
        monkeypatch.setenv(name, "0")
    assert not is_offline()
    enforce_offline()
    assert is_offline()


# pytest-socket also warns when it blocks a connection; here that is the expected outcome.
@pytest.mark.filterwarnings("ignore:A test tried to use socket")
def test_internet_is_blocked_during_tests():
    # 1.1.1.1 is a public IP address, so no DNS lookup happens before the block.
    with pytest.raises(SocketConnectBlockedError):
        socket.create_connection(("1.1.1.1", 80), timeout=1)
