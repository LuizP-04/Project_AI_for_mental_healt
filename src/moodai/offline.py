"""Offline mode: stop libraries from reaching the internet during an analysis.

Health data is sensitive personal data under the LGPD, so an analysis must
never make network calls. Hugging Face libraries read these environment
variables only once, when they are first imported. That is why ``moodai``
calls ``enforce_offline()`` as soon as the package itself is imported, before
any model library is loaded.
"""

import os

# Environment variables that switch libraries to offline mode or turn off telemetry.
OFFLINE_ENV: dict[str, str] = {
    # Hugging Face: never download models or datasets, never send usage telemetry.
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    # Generic telemetry opt-out, honored by several tools.
    "DO_NOT_TRACK": "1",
    # Streamlit (the interface, built in step 4): do not collect usage statistics.
    "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
}


def enforce_offline() -> None:
    """Set every variable in ``OFFLINE_ENV``, overriding any existing value."""
    os.environ.update(OFFLINE_ENV)


def is_offline() -> bool:
    """Return True if every variable in ``OFFLINE_ENV`` holds its offline value."""
    return all(os.environ.get(name) == value for name, value in OFFLINE_ENV.items())
