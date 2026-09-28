"""Test doubles for the scyllasband runtime surface used by SynthesisService."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
from scyllasband.streaming import StreamingEvent

FAKE_SAMPLE_RATE = 24000
FAKE_TONE_HZ = 440.0
FAKE_AMPLITUDE = 0.5
FAKE_PIECE_MS = 100


def make_voice_spec(voice_id: str, languages: list[str], name: str | None = None) -> Any:
    return SimpleNamespace(
        id=voice_id,
        name=name or voice_id.title(),
        languages=tuple(languages),
        default_language=languages[0],
    )


def fake_audio_piece(
    sample_rate: int = FAKE_SAMPLE_RATE, piece_ms: int = FAKE_PIECE_MS
) -> np.ndarray:
    """Deterministic sine wave, mirroring float_to_int16_bytes's math order."""
    samples = sample_rate * piece_ms // 1000
    t = np.arange(samples, dtype=np.float32) / np.float32(sample_rate)
    return (np.float32(FAKE_AMPLITUDE) * np.sin(2 * np.pi * np.float32(FAKE_TONE_HZ) * t)).astype(
        np.float32
    )


class FakeRuntime:
    """Minimal stand-in for ScyllasBandRuntime covering the surface we call.

    - voices/manifest structure for info + resolution
    - resolve_language_for_voice with the same error shape as the real runtime
    - synthesize_stream emitting deterministic audio_chunk StreamingEvents
      (two pieces when the text is longer than 40 chars, one otherwise)
    """

    def __init__(
        self,
        voices: list[Any] | None = None,
        sample_rate: int = FAKE_SAMPLE_RATE,
    ) -> None:
        if voices is None:
            voices = [
                make_voice_spec("ariadne", ["en_us", "es"]),
                make_voice_spec("bella", ["en_us", "fr"]),
                make_voice_spec("vesper", ["it"]),
            ]
        self.manifest = SimpleNamespace(
            voices=tuple(voices),
            languages=tuple(sorted({lang for voice in voices for lang in voice.languages})),
            audio=SimpleNamespace(sample_rate=sample_rate),
        )
        self.requests: list[dict[str, Any]] = []
        self._sample_rate = sample_rate

    # -- runtime API ---------------------------------------------------
    def available_voices(self) -> list[str]:
        return [voice.id for voice in self.manifest.voices]

    def voice_for_id(self, voice_id: str) -> Any:
        for voice in self.manifest.voices:
            if voice.id == voice_id:
                return voice
        raise ValueError(f"Unknown voice_id: {voice_id}")

    def resolve_language_for_voice(self, voice_id: str, language: str | None) -> str:
        voice = self.voice_for_id(voice_id)
        requested = None if language is None else str(language).strip().lower().replace("-", "_")
        resolved = voice.default_language if requested in (None, "en") else requested
        if resolved not in self.manifest.languages:
            options = ", ".join(self.manifest.languages)
            raise ValueError(f"Unsupported language {resolved!r}; expected one of: {options}")
        if resolved not in voice.languages:
            options = ", ".join(voice.languages)
            raise ValueError(
                f"Unsupported language {resolved!r} for voice {voice_id!r}; "
                f"expected one of: {options}"
            )
        return resolved

    def warmup(self, **_: Any) -> dict[str, Any]:
        return {"backend": "onnx"}

    # -- synthesis -----------------------------------------------------
    def synthesize_stream(
        self,
        text: str | None = None,
        voice_id: str | None = None,
        language: str | None = None,
        delivery: Any = None,
        **overrides: Any,
    ) -> Iterator[StreamingEvent]:
        self.requests.append(
            {
                "text": text,
                "voice_id": voice_id,
                "language": language,
                "delivery": delivery,
                **overrides,
            }
        )
        piece = fake_audio_piece(self._sample_rate)
        piece_count = 2 if text and len(text) > 40 else 1
        for index in range(piece_count):
            yield StreamingEvent(
                type="audio_chunk",
                audio=piece,
                sample_rate=self._sample_rate,
                start_sample=index * len(piece),
                end_sample=(index + 1) * len(piece),
                metadata={"index": index},
            )
        yield StreamingEvent(type="done", sample_rate=self._sample_rate)
