"""Connection to the local Ollama server, with guards that keep the text on this machine.

Every connection goes through these checks first:
- the server address must be a loopback address (127.0.0.1, ::1 or localhost);
- the model must not be an Ollama cloud model (names ending in ":cloud" or "-cloud" run
  on Ollama's servers, so the text would leave the computer);
- the HTTP client ignores proxy settings and does not follow redirects, so a request
  cannot be sent anywhere other than the checked address.

Ollama itself also runs with its cloud features disabled (see the README). Models are never
downloaded from here: downloads happen only during setup, with `ollama pull`.
"""

import ipaddress
from collections.abc import Iterable
from urllib.parse import urlsplit

import httpx
import ollama

CLOUD_SUFFIXES = (":cloud", "-cloud")
NOT_RUNNING = (
    "Ollama is not running on this machine. Start it in another terminal with: ollama serve"
)


class LocalLLMError(Exception):
    """The LLM setup would break the "nothing leaves the machine" rule, or cannot be used."""


def check_host(url: str) -> str:
    """Return the Ollama address unchanged if it points to this machine; raise otherwise."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise LocalLLMError(
            f"Invalid Ollama address {url!r}: expected something like http://127.0.0.1:11434"
        )
    host = parts.hostname
    if host != "localhost":
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:  # a host name other than localhost
            is_loopback = False
        if not is_loopback:
            raise LocalLLMError(
                f"Ollama address {url!r} is not on this machine: only loopback addresses "
                "(127.0.0.1, ::1, localhost) are allowed, so the text never leaves the computer"
            )
    return url


def full_model_name(name: str) -> str:
    """Add Ollama's implicit ":latest" tag to a name without a tag (qwen3.5 -> qwen3.5:latest)."""
    last_part = name.rsplit("/", 1)[-1]
    return name if ":" in last_part else f"{name}:latest"


def check_model_name(name: str) -> str:
    """Return the model name unchanged if it is a local model name; raise otherwise."""
    if not name.strip():
        raise LocalLLMError("Empty model name")
    if name.lower().endswith(CLOUD_SUFFIXES):
        raise LocalLLMError(
            f"{name!r} is an Ollama cloud model: it would send the text to Ollama's servers"
        )
    return name


def connect(host: str, timeout_s: float) -> ollama.Client:
    """Create a client for the local Ollama server at `host`, after checking the address."""
    return ollama.Client(
        host=check_host(host),
        timeout=timeout_s,
        follow_redirects=False,  # a redirect could point to another machine
        trust_env=False,  # ignore HTTP(S)_PROXY settings: no request goes through a proxy
    )


def server_version(host: str, timeout_s: float) -> str:
    """Return the version of the Ollama server at `host`, after checking the address."""
    url = check_host(host).rstrip("/") + "/api/version"
    try:
        response = httpx.get(url, timeout=timeout_s, follow_redirects=False, trust_env=False)
        response.raise_for_status()
    except httpx.TransportError as error:  # nothing listening, or no answer in time
        raise LocalLLMError(NOT_RUNNING) from error
    except httpx.HTTPStatusError as error:
        raise LocalLLMError(f"Ollama at {host} answered with an error: {error}") from error
    return str(response.json().get("version", "unknown"))


def require_models(client: ollama.Client, names: Iterable[str]) -> None:
    """Check that every model is a local model already installed on the Ollama server."""
    wanted = [full_model_name(check_model_name(name)) for name in names]
    try:
        installed = {model.model for model in client.list().models}
    except ConnectionError as error:
        raise LocalLLMError(NOT_RUNNING) from error
    missing = [name for name in wanted if name not in installed]
    if missing:
        commands = "; ".join(f"ollama pull {name}" for name in missing)
        raise LocalLLMError(
            f"Model(s) not installed: {', '.join(missing)}. Download them during setup with: "
            f"{commands}"
        )
