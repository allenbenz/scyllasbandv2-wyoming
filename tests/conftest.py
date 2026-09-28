"""Shared fixtures. FakeRuntime lives in tests/fakes.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from scyllasbandv2_wyoming.config import Settings
from scyllasbandv2_wyoming.runtime import SynthesisService

from .fakes import FakeRuntime


@pytest.fixture
def fake_runtime() -> FakeRuntime:
    return FakeRuntime()


@pytest.fixture
def fake_settings(tmp_path: Path) -> Settings:
    return Settings(model_dir=tmp_path, output_chunk_ms=50)


@pytest.fixture
def fake_service(fake_runtime: FakeRuntime, fake_settings: Settings) -> SynthesisService:
    return SynthesisService(
        fake_runtime,
        fake_settings,
        providers=["CPUExecutionProvider"],
        bundle_dir=fake_settings.model_dir / "v2" / fake_settings.bundle,
    )
