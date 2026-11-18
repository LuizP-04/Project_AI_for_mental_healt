"""Load YAML configuration files.

Paths, model names and experiment settings live in ``configs/*.yaml`` instead
of the code, so components can be swapped through configuration alone (RNF05).
"""

from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when a configuration file is missing or invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return its top-level mapping.

    Args:
        path: Path to the YAML file.

    Returns:
        The configuration as a dictionary.

    Raises:
        ConfigError: If the file does not exist, is not valid YAML, or its top
            level is not a mapping (``key: value`` pairs).
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    try:
        with config_path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in {config_path}: {error}") from error
    if not isinstance(data, dict):
        raise ConfigError(f"The top level of {config_path} must be a mapping (key: value pairs).")
    return data
