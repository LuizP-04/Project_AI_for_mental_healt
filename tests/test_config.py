"""Tests for moodai.config.load_config, using small YAML files written by each test."""

from pathlib import Path

import pytest

from moodai.config import ConfigError, load_config

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def test_loads_a_valid_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("seed: 42\nlanguages: [pt, en]\n", encoding="utf-8")
    assert load_config(path) == {"seed": 42, "languages": ["pt", "en"]}


def test_missing_file_raises_a_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_invalid_yaml_raises_a_clear_error(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("seed: [42\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(path)


def test_top_level_must_be_a_mapping(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- pt\n- en\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_default_config_lists_the_data_folders():
    paths = load_config(DEFAULT_CONFIG)["paths"]
    assert set(paths) == {"raw_data", "processed_data", "models", "outputs"}
