"""Unit tests for SynthesisService.create() bundle download/skip decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scyllasbandv2_wyoming import runtime as runtime_module
from scyllasbandv2_wyoming.config import Settings
from scyllasbandv2_wyoming.runtime import SynthesisService, SynthesisServiceError

from ..fakes import FakeRuntime


def make_bundle(root: Path, settings: Settings) -> Path:
    bundle_dir = root / "v2" / settings.bundle
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "manifest.json").write_text("{}", encoding="utf-8")
    return bundle_dir


class Harness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.downloads: list[dict[str, Any]] = []
        self.loads: list[Path] = []
        self.fail_first_n = 0
        self.tmp_path = tmp_path
        monkeypatch.setattr(runtime_module, "_download_bundle_sync", self._download)
        monkeypatch.setattr(runtime_module, "_load_runtime_sync", self._load)

    def _download(self, settings: Settings, *, force: bool) -> Path:
        self.downloads.append({"force": force})
        return make_bundle(self.tmp_path, settings)

    def _load(self, bundle_dir: Path, providers: list[str], settings: Settings) -> Any:
        self.loads.append(bundle_dir)
        if self.fail_first_n > 0:
            self.fail_first_n -= 1
            raise FileNotFoundError(f"missing weights in {bundle_dir}")
        return FakeRuntime()


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Harness:
    return Harness(monkeypatch, tmp_path)


async def test_auto_skips_download_when_bundle_present(harness: Harness, tmp_path: Path) -> None:
    settings = Settings(model_dir=tmp_path, device="cpu", warmup=False)
    make_bundle(tmp_path, settings)
    service = await SynthesisService.create(settings)
    assert harness.downloads == []  # nothing downloaded
    assert service.voices()  # loaded from the existing bundle
    assert len(harness.loads) == 1


async def test_auto_downloads_when_missing(harness: Harness, tmp_path: Path) -> None:
    settings = Settings(model_dir=tmp_path, device="cpu", warmup=False)
    service = await SynthesisService.create(settings)
    assert harness.downloads == [{"force": False}]
    assert service.voices()


async def test_never_raises_when_missing(harness: Harness, tmp_path: Path) -> None:
    settings = Settings(model_dir=tmp_path, device="cpu", download="never", warmup=False)
    with pytest.raises(SynthesisServiceError, match="SCYLLASBAND_DOWNLOAD=never"):
        await SynthesisService.create(settings)
    assert harness.downloads == []


async def test_never_skips_when_present(harness: Harness, tmp_path: Path) -> None:
    settings = Settings(model_dir=tmp_path, device="cpu", download="never", warmup=False)
    make_bundle(tmp_path, settings)
    await SynthesisService.create(settings)
    assert harness.downloads == []


async def test_always_forces_download(harness: Harness, tmp_path: Path) -> None:
    settings = Settings(model_dir=tmp_path, device="cpu", download="always", warmup=False)
    make_bundle(tmp_path, settings)  # bundle already present
    await SynthesisService.create(settings)
    assert harness.downloads == [{"force": True}]


async def test_auto_self_heals_corrupt_bundle(harness: Harness, tmp_path: Path) -> None:
    settings = Settings(model_dir=tmp_path, device="cpu", warmup=False)
    make_bundle(tmp_path, settings)
    harness.fail_first_n = 1  # first load fails as if weights were truncated
    service = await SynthesisService.create(settings)
    assert service.voices()
    assert len(harness.loads) == 2  # retried after the re-download
    assert harness.downloads == [{"force": True}]


async def test_corrupt_bundle_never_mode_still_raises(harness: Harness, tmp_path: Path) -> None:
    settings = Settings(model_dir=tmp_path, device="cpu", download="never", warmup=False)
    make_bundle(tmp_path, settings)
    harness.fail_first_n = 1
    with pytest.raises(FileNotFoundError):
        await SynthesisService.create(settings)  # no self-healing in never mode
    assert harness.downloads == []
