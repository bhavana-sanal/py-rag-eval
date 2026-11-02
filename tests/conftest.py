from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def example_config() -> Path:
    return ROOT / "experiments" / "example.toml"


@pytest.fixture
def demo_dir() -> Path:
    return ROOT / "data" / "demo"
