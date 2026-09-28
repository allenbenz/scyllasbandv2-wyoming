"""Loading, downloading, and streaming synthesis for the Scylla's Band v2 bundle."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from scyllasband.delivery import resolve_delivery
from scyllasband.download import download_litert_bundle
from scyllasband.runtime import ScyllasBandRuntime

from .audio import match_model_language
from .config import Settings

_LOGGER = logging.getLogger(__name__)

_SENTINEL = object()


class SynthesisServiceError(RuntimeError):
    """Fatal startup or configuration error (message goes to the logs)."""


class VoiceResolutionError(ValueError):
    """Request referenced an unknown voice or language (message is client-safe)."""


@dataclass(frozen=True)
class AudioPiece:
    """One contiguous piece of synthesized float32 audio."""

    audio: np.ndarray
    sample_rate: int


def _register_windows_cuda_dlls() -> None:
    """Expose pip-installed NVIDIA CUDA/cuDNN DLLs to onnxruntime on Windows.

    onnxruntime-gpu loads cublas64_13.dll, cudnn64_9.dll, ... when the CUDA
    execution provider runs its first kernel. When those libraries come from
    pip, their directories are not on the DLL search path, so register them
    explicitly. Covers both layouts: the CUDA 13 wheels share
    ``nvidia/cu13/bin/<arch>/`` while cuDNN (and the older cu12 wheels) use
    ``nvidia/<name>/bin/``. Harmless when not installed.
    """
    if sys.platform != "win32":
        return
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return
    if spec is None or not spec.submodule_search_locations:
        return
    nvidia_root = Path(next(iter(spec.submodule_search_locations)))
    dll_dirs: set[Path] = set()
    # CUDA 13 shared layout: nvidia/cu13/bin/x86_64
    dll_dirs.update((nvidia_root / "cu13" / "bin").glob("*"))
    # per-library layout: nvidia/cudnn/bin, nvidia/cublas/bin, ...
    for legacy_dir in nvidia_root.glob("*/bin"):
        if legacy_dir.is_dir():
            dll_dirs.add(legacy_dir)
    for dll_dir in sorted(dll_dirs):
        if not dll_dir.is_dir():
            continue
        try:
            os.add_dll_directory(str(dll_dir))
            # onnxruntime loads its provider DLLs with plain LoadLibrary,
            # which ignores add_dll_directory and only searches PATH
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
        except OSError:  # pragma: no cover - defensive
            pass


def resolve_onnx_providers(device: str) -> list[str]:
    """Choose onnxruntime execution providers for the requested device."""
    import onnxruntime as ort

    _register_windows_cuda_dlls()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    available = list(ort.get_available_providers())
    has_cuda = "CUDAExecutionProvider" in available
    if device == "cuda" and not has_cuda:
        raise SynthesisServiceError(
            "SCYLLASBAND_DEVICE=cuda but onnxruntime was built without the CUDA "
            "execution provider (is the CPU-only onnxruntime package installed?)"
        )
    if has_cuda:
        _LOGGER.info(
            "CUDA execution provider is available; preferring GPU "
            "(onnxruntime falls back to CPU if the driver/libraries are missing)"
        )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    _LOGGER.info("CUDA execution provider unavailable; using CPU")
    return ["CPUExecutionProvider"]


def bundle_path(settings: Settings) -> Path:
    """Local bundle directory for the configured model dir and variant."""
    return (Path(settings.model_dir) / "v2" / settings.bundle).resolve()


def _download_bundle_sync(settings: Settings, *, force: bool) -> Path:
    return download_litert_bundle(
        models_dir=Path(settings.model_dir),
        bundle_subdir=settings.bundle,
        revision=settings.revision,
        force=force,
    )


def _load_runtime_sync(
    bundle_dir: Path, providers: list[str], settings: Settings
) -> ScyllasBandRuntime:
    return ScyllasBandRuntime.from_bundle(
        bundle_dir,
        backends=["onnx"],
        onnx_providers=providers,
        onnx_intra_op_num_threads=settings.intra_op_threads,
        onnx_inter_op_num_threads=settings.inter_op_threads,
    )


class SynthesisService:
    """Wraps a ScyllasBandRuntime with download, validation, and async streaming."""

    def __init__(
        self,
        runtime: Any,
        settings: Settings,
        *,
        providers: list[str],
        bundle_dir: Path,
    ) -> None:
        self._runtime = runtime
        self._settings = settings
        self._providers = list(providers)
        self._bundle_dir = Path(bundle_dir)

    # ------------------------------------------------------------------
    # Introspection (used for the wyoming info message and logging)
    # ------------------------------------------------------------------
    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def providers(self) -> list[str]:
        return list(self._providers)

    @property
    def bundle_dir(self) -> Path:
        return self._bundle_dir

    @property
    def sample_rate(self) -> int:
        return int(self._runtime.manifest.audio.sample_rate)

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(self._runtime.manifest.languages)

    def voices(self) -> tuple[Any, ...]:
        return tuple(self._runtime.manifest.voices)

    # ------------------------------------------------------------------
    # Voice/language resolution
    # ------------------------------------------------------------------
    def default_voice(self) -> Any:
        voices = self.voices()
        if not voices:
            raise SynthesisServiceError("model bundle exposes no voices")
        if self._settings.default_voice:
            try:
                return self._runtime.voice_for_id(self._settings.default_voice)
            except ValueError as exc:
                raise SynthesisServiceError(f"invalid SCYLLASBAND_VOICE: {exc}") from exc
        return voices[0]

    def resolve_voice(self, name: str | None) -> Any:
        if name is None or not str(name).strip():
            return self.default_voice()
        try:
            return self._runtime.voice_for_id(str(name).strip())
        except ValueError:
            options = ", ".join(voice.id for voice in self.voices())
            raise VoiceResolutionError(
                f"Unknown voice {name!r}. Available voices: {options}"
            ) from None

    def resolve_language(self, voice: Any, language: str | None) -> str:
        """Resolve the language for a request: explicit, then server default."""
        available = list(self.languages)
        candidates: list[str] = []
        if language is not None and str(language).strip():
            candidates.append(match_model_language(available, str(language).strip()))
        if self._settings.default_language:
            candidates.append(match_model_language(available, self._settings.default_language))
        for candidate in candidates:
            try:
                return str(self._runtime.resolve_language_for_voice(voice.id, candidate))
            except ValueError as exc:
                raise VoiceResolutionError(str(exc)) from None
        return str(voice.default_language)

    def resolve_voice_for_request(self, name: str | None, language: str | None) -> Any:
        """Resolve the voice for a request: explicit name, then any voice that
        speaks the requested language, then the server default."""
        if name is not None and str(name).strip():
            return self.resolve_voice(name)
        if language is not None and str(language).strip():
            target = match_model_language(
                (lang for voice in self.voices() for lang in voice.languages),
                str(language).strip(),
            )
            for voice in self.voices():
                if target in voice.languages:
                    return voice
        return self.default_voice()

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    async def stream(self, *, text: str, voice_id: str, language: str) -> AsyncIterator[AudioPiece]:
        """Stream synthesized audio pieces for ``text`` without blocking the loop.

        The scyllasband streaming iterator is sync (it runs ONNX inference);
        each ``next()`` is dispatched to the default thread-pool executor.
        """
        settings = self._settings
        overrides: dict[str, Any] = {
            "steps": settings.steps,
            "sampler": settings.sampler,
            "speed": settings.speed,
            "max_chunk_seconds": settings.max_chunk_seconds,
        }
        if settings.seed is not None:
            overrides["seed"] = settings.seed

        iterator = self._runtime.synthesize_stream(
            text=text,
            voice_id=voice_id,
            language=language,
            delivery=settings.delivery,
            **overrides,
        )
        loop = asyncio.get_running_loop()
        stream_iter = iter(iterator)
        while True:
            event = await loop.run_in_executor(None, partial(next, stream_iter, _SENTINEL))
            if event is _SENTINEL:
                break
            event_type = getattr(event, "type", None)
            if event_type == "audio_chunk":
                audio = getattr(event, "audio", None)
                if audio is None:
                    continue
                array = np.asarray(audio, dtype=np.float32).reshape(-1)
                if array.size == 0:
                    continue
                rate = int(getattr(event, "sample_rate", 0) or self.sample_rate)
                yield AudioPiece(audio=array, sample_rate=rate)
            elif event_type == "warning":
                message = getattr(event, "message", None) or ""
                _LOGGER.warning("scyllasband streaming warning: %s", message)

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------
    @classmethod
    async def create(cls, settings: Settings) -> SynthesisService:
        """Validate settings, ensure the bundle exists (downloading if allowed),
        load the runtime, and warm it up."""
        if settings.delivery:
            try:
                resolve_delivery(settings.delivery)
            except ValueError as exc:
                raise SynthesisServiceError(
                    f"invalid SCYLLASBAND_DELIVERY {settings.delivery!r}: {exc}"
                ) from exc

        bundle_dir = bundle_path(settings)
        have_bundle = (bundle_dir / "manifest.json").is_file()
        force_download = settings.download == "always"
        if force_download or (not have_bundle and settings.download == "auto"):
            reason = (
                " (forced re-download)"
                if force_download and have_bundle
                else " (one-time, ~500 MB)"
            )
            _LOGGER.info(
                "Downloading Scylla's Band v2 %s bundle to %s%s",
                settings.bundle,
                settings.model_dir,
                reason,
            )
            await asyncio.to_thread(_download_bundle_sync, settings, force=force_download)
        elif not have_bundle:
            raise SynthesisServiceError(
                f"Model bundle not found at {bundle_dir} and downloading is disabled "
                "(SCYLLASBAND_DOWNLOAD=never). Use SCYLLASBAND_DOWNLOAD=auto or "
                "provide the bundle directory."
            )

        providers = resolve_onnx_providers(settings.device)
        _LOGGER.info(
            "Loading Scylla's Band v2 bundle from %s (device=%s, providers=%s)",
            bundle_dir,
            settings.device,
            providers,
        )
        try:
            runtime = await asyncio.to_thread(_load_runtime_sync, bundle_dir, providers, settings)
        except Exception:
            if not (settings.download == "auto" and have_bundle and not force_download):
                raise
            # The bundle looked present but fails to load (e.g. copied
            # incompletely or corrupted on disk). Self-heal once via a forced
            # re-download while in auto mode.
            _LOGGER.warning(
                "Bundle at %s failed to load; re-downloading once",
                bundle_dir,
                exc_info=True,
            )
            await asyncio.to_thread(_download_bundle_sync, settings, force=True)
            runtime = await asyncio.to_thread(_load_runtime_sync, bundle_dir, providers, settings)

        if settings.warmup:
            _LOGGER.info("Warming up the runtime (discarded one-step render)")
            try:
                await asyncio.to_thread(runtime.warmup)
            except Exception as exc:
                if settings.device == "auto" and providers[0] == "CUDAExecutionProvider":
                    # The CUDA EP can fail at first run (e.g. cuDNN missing on
                    # Windows) even though session creation succeeded. Honor
                    # "auto" by degrading to CPU-only providers.
                    _LOGGER.warning(
                        "warmup failed with the CUDA execution provider (%s); "
                        "falling back to CPU-only providers",
                        exc,
                    )
                    providers = ["CPUExecutionProvider"]
                    runtime = await asyncio.to_thread(
                        _load_runtime_sync, bundle_dir, providers, settings
                    )
                    await asyncio.to_thread(runtime.warmup)
                elif settings.device == "cuda":
                    raise SynthesisServiceError(
                        f"warmup failed with the CUDA execution provider: {exc}"
                    ) from exc
                raise

        service = cls(runtime, settings, providers=providers, bundle_dir=bundle_dir)
        service.default_voice()  # fail fast on an unknown SCYLLASBAND_VOICE

        _LOGGER.info(
            "Model ready: %d voices, languages: %s (providers: %s)",
            len(service.voices()),
            ", ".join(service.languages),
            ", ".join(providers),
        )
        return service
