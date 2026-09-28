"""Audio helpers: PCM conversion, chunking, and language-code mapping."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

SAMPLE_WIDTH = 2  # bytes per sample (16-bit PCM)
CHANNELS = 1  # mono


def float_to_int16_bytes(audio: np.ndarray, gain: float = 1.0) -> bytes:
    """Convert float32 [-1, 1] samples to little-endian 16-bit PCM bytes.

    An optional linear ``gain`` is applied before clipping (this mirrors the
    runtime's waveform-gain control, which the streaming path does not expose).
    """
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if gain != 1.0:
        samples = samples * np.float32(gain)
    clipped = np.clip(samples, np.float32(-1.0), np.float32(1.0))
    return (clipped * np.float32(32767.0)).astype("<i2").tobytes()


def split_bytes(data: bytes, max_bytes: int) -> Iterator[bytes]:
    """Yield ``data`` in pieces of at most ``max_bytes`` bytes (never empty)."""
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be > 0, got {max_bytes}")
    for offset in range(0, len(data), max_bytes):
        yield data[offset : offset + max_bytes]


def chunk_bytes_for_ms(
    sample_rate: int,
    chunk_ms: int,
    *,
    width: int = SAMPLE_WIDTH,
    channels: int = CHANNELS,
) -> int:
    """Byte size of one output chunk holding at most ``chunk_ms`` of audio."""
    samples = max(1, round(sample_rate * chunk_ms / 1000))
    return samples * width * channels


def model_language_to_bcp47(code: str) -> str:
    """Map a model language code (``en_us``) to BCP-47 style (``en-US``).

    Single-part codes (``es``, ``it``, ``fr``, ``de``, ``vi``) pass through
    lowercased; two-part codes become ``language-REGION``.
    """
    code = code.strip()
    if not code:
        return code
    parts = [part for part in code.replace("-", "_").split("_") if part]
    if not parts:
        return code
    if len(parts) == 1:
        return parts[0].lower()
    return f"{parts[0].lower()}-{'-'.join(part.upper() for part in parts[1:])}"


def bcp47_to_model_language(code: str) -> str:
    """Map a BCP-47 style code (``en-US``) to the model's form (``en_us``)."""
    normalized = code.strip().replace("-", "_").lower()
    return normalized or code


def match_model_language(available, requested: str) -> str:
    """Best-match a requested BCP-47 code against available model codes.

    Tries the full mapping first (``en-US`` -> ``en_us``), then the primary
    subtag (``de-DE`` -> ``de`` when only ``de`` is available). Returns the
    mapped form unchanged when nothing matches so callers can report it.
    """
    available = tuple(available)
    mapped = bcp47_to_model_language(requested)
    if mapped in available:
        return mapped
    primary = mapped.split("_", 1)[0]
    if primary and primary in available:
        return primary
    return mapped
