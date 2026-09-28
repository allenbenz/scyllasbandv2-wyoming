"""Unit tests for audio conversion, chunking, and language mapping."""

from __future__ import annotations

import numpy as np
import pytest

from scyllasbandv2_wyoming.audio import (
    bcp47_to_model_language,
    chunk_bytes_for_ms,
    float_to_int16_bytes,
    model_language_to_bcp47,
    split_bytes,
)


def test_float_to_int16_bytes_known_values() -> None:
    audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5], dtype=np.float32)
    data = float_to_int16_bytes(audio)
    values = np.frombuffer(data, dtype="<i2")
    assert values[0] == 0
    assert values[1] == int(0.5 * 32767)
    assert values[2] == -int(0.5 * 32767)
    assert values[3] == 32767  # clipped
    assert values[4] == -32767
    assert values[5] == 32767  # 1.5 clips to 1
    assert values[6] == -32767


def test_float_to_int16_bytes_gain() -> None:
    audio = np.array([0.25, -0.25, 0.75], dtype=np.float32)
    values = np.frombuffer(float_to_int16_bytes(audio, gain=2.0), dtype="<i2")
    assert values[0] == int(0.5 * 32767)
    assert values[1] == -int(0.5 * 32767)
    assert values[2] == 32767  # 1.5 clips


def test_float_to_int16_bytes_shape_handling() -> None:
    audio = np.zeros((2, 3), dtype=np.float32)
    assert len(float_to_int16_bytes(audio)) == 6 * 2


def test_split_bytes_sizes() -> None:
    data = b"x" * 250
    parts = list(split_bytes(data, 100))
    assert [len(part) for part in parts] == [100, 100, 50]
    assert b"".join(parts) == data


def test_split_bytes_empty() -> None:
    assert list(split_bytes(b"", 100)) == []


def test_split_bytes_invalid() -> None:
    with pytest.raises(ValueError):
        list(split_bytes(b"abc", 0))


def test_chunk_bytes_for_ms() -> None:
    # 24000 Hz * 200 ms = 4800 samples * 2 bytes = 9600 bytes
    assert chunk_bytes_for_ms(24000, 200) == 9600
    # 24000 Hz * 50 ms = 1200 samples
    assert chunk_bytes_for_ms(24000, 50) == 2400


@pytest.mark.parametrize(
    ("model", "bcp47"),
    [
        ("en_us", "en-US"),
        ("en_gb", "en-GB"),
        ("es", "es"),
        ("it", "it"),
        ("fr", "fr"),
        ("de", "de"),
        ("vi", "vi"),
    ],
)
def test_model_language_to_bcp47(model: str, bcp47: str) -> None:
    assert model_language_to_bcp47(model) == bcp47
    assert bcp47_to_model_language(bcp47) == model


def test_bcp47_to_model_language_passthrough() -> None:
    assert bcp47_to_model_language("en-US") == "en_us"
    assert bcp47_to_model_language("en") == "en"
