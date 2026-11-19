"""Tests for the guards in moodai.llm.local. They need no Ollama server, so they also run in CI."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from moodai.config import load_config
from moodai.llm.local import (
    LocalLLMError,
    check_host,
    check_model_name,
    connect,
    full_model_name,
    require_models,
    server_version,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434",
        "http://localhost:11434",
        "http://[::1]:11434",
        "http://127.0.0.2:11434",  # the whole 127.0.0.0/8 block is loopback
        "https://127.0.0.1:11434",
    ],
)
def test_loopback_addresses_are_accepted(url):
    assert check_host(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.0.10:11434",  # another computer on the local network
        "http://0.0.0.0:11434",  # "every interface", not this machine only
        "https://ollama.com",
        "http://localhost.example.com:11434",  # looks local, but is not
        "127.0.0.1:11434",  # no scheme
        "ftp://127.0.0.1:11434",
        "",
    ],
)
def test_other_addresses_are_refused(url):
    with pytest.raises(LocalLLMError):
        check_host(url)


@pytest.mark.parametrize("name", ["gemma4:cloud", "gemma4:31b-cloud", "GEMMA4:CLOUD", " "])
def test_cloud_and_empty_model_names_are_refused(name):
    with pytest.raises(LocalLLMError):
        check_model_name(name)


@pytest.mark.parametrize("name", ["qwen3.5:4b-q4_K_M", "gemma4:e2b-it-qat", "llama3.2"])
def test_local_model_names_are_accepted(name):
    assert check_model_name(name) == name


def test_full_model_name_adds_the_implicit_latest_tag():
    assert full_model_name("qwen3.5") == "qwen3.5:latest"
    assert full_model_name("qwen3.5:4b") == "qwen3.5:4b"
    assert full_model_name("hf.co/user/model") == "hf.co/user/model:latest"


def test_client_ignores_proxies_and_redirects():
    http = connect("http://127.0.0.1:11434", timeout_s=5)._client
    assert http.trust_env is False
    assert http.follow_redirects is False


def test_connect_refuses_a_remote_address():
    with pytest.raises(LocalLLMError):
        connect("https://ollama.com", timeout_s=5)


def test_server_version_refuses_a_remote_address_before_any_request():
    with pytest.raises(LocalLLMError, match="not on this machine"):
        server_version("https://ollama.com", timeout_s=5)


def test_server_version_explains_when_nothing_is_listening():
    # Port 9 ("discard") on this machine: nothing listens there, so the connection fails.
    with pytest.raises(LocalLLMError, match="ollama serve"):
        server_version("http://127.0.0.1:9", timeout_s=5)


class FakeClient:
    """Stands in for ollama.Client: lists the given models, or fails as if Ollama were off."""

    def __init__(self, models=(), running=True):
        self.models = models
        self.running = running
        self.calls = 0

    def list(self):
        self.calls += 1
        if not self.running:
            raise ConnectionError("Failed to connect to Ollama")
        return SimpleNamespace(models=[SimpleNamespace(model=name) for name in self.models])


def test_installed_models_pass():
    client = FakeClient(models=["qwen3.5:4b-q4_K_M", "llama3.2:latest"])
    require_models(client, ["qwen3.5:4b-q4_K_M", "llama3.2"])


def test_missing_model_tells_how_to_download_it():
    client = FakeClient(models=["qwen3.5:4b-q4_K_M"])
    with pytest.raises(LocalLLMError, match="ollama pull gemma4:e2b-it-qat"):
        require_models(client, ["qwen3.5:4b-q4_K_M", "gemma4:e2b-it-qat"])


def test_server_off_tells_how_to_start_it():
    with pytest.raises(LocalLLMError, match="ollama serve"):
        require_models(FakeClient(running=False), ["qwen3.5:4b-q4_K_M"])


def test_cloud_model_is_refused_before_contacting_the_server():
    client = FakeClient(models=["gemma4:cloud"])
    with pytest.raises(LocalLLMError, match="cloud"):
        require_models(client, ["gemma4:cloud"])
    assert client.calls == 0


def test_default_config_points_to_this_machine_and_local_models():
    llm = load_config(DEFAULT_CONFIG)["llm"]
    check_host(llm["host"])
    assert llm["models"], "the default config lists no LLM"
    for name in llm["models"]:
        check_model_name(name)
