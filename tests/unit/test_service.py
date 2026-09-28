"""Unit tests for SynthesisService voice/language resolution and streaming,
plus the wyoming Info message construction (no server, fake runtime)."""

from __future__ import annotations

import numpy as np
import pytest

from scyllasbandv2_wyoming.config import Settings
from scyllasbandv2_wyoming.runtime import SynthesisService, VoiceResolutionError
from scyllasbandv2_wyoming.server import PROGRAM_NAME, build_info

from ..fakes import FAKE_SAMPLE_RATE, FakeRuntime, fake_audio_piece, make_voice_spec


def test_resolve_voice_by_name(fake_service: SynthesisService) -> None:
    voice = fake_service.resolve_voice("bella")
    assert voice.id == "bella"


def test_resolve_voice_default(fake_service: SynthesisService) -> None:
    assert fake_service.resolve_voice(None).id == "ariadne"
    assert fake_service.resolve_voice("").id == "ariadne"


def test_resolve_voice_unknown(fake_service: SynthesisService) -> None:
    with pytest.raises(VoiceResolutionError, match="ariadne, bella, vesper"):
        fake_service.resolve_voice("nope")


def test_resolve_voice_for_request_prefers_name(fake_service: SynthesisService) -> None:
    assert fake_service.resolve_voice_for_request("vesper", None).id == "vesper"


def test_resolve_voice_for_request_matches_language(fake_service: SynthesisService) -> None:
    # fr is only spoken by bella; en-US maps to en_us (ariadne first)
    assert fake_service.resolve_voice_for_request(None, "fr").id == "bella"
    assert fake_service.resolve_voice_for_request(None, "en-US").id == "ariadne"


def test_resolve_voice_for_request_falls_back_to_default(
    fake_service: SynthesisService,
) -> None:
    assert fake_service.resolve_voice_for_request(None, None).id == "ariadne"
    # unsupported language falls back to the default voice; language
    # validation then reports the error with options
    assert fake_service.resolve_voice_for_request(None, "de-DE").id == "ariadne"


def test_default_voice_setting(fake_runtime) -> None:
    settings = Settings(default_voice="vesper")
    service = SynthesisService(
        fake_runtime, settings, providers=["CPUExecutionProvider"], bundle_dir="."
    )
    assert service.default_voice().id == "vesper"
    assert service.resolve_voice(None).id == "vesper"


def test_resolve_language_explicit_maps_bcp47(fake_service: SynthesisService) -> None:
    voice = fake_service.resolve_voice("ariadne")
    assert fake_service.resolve_language(voice, "en-US") == "en_us"
    assert fake_service.resolve_language(voice, "es") == "es"


def test_resolve_language_en_alias_uses_voice_default(
    fake_service: SynthesisService,
) -> None:
    voice = fake_service.resolve_voice("ariadne")  # default en_us
    assert fake_service.resolve_language(voice, "en") == "en_us"


def test_resolve_language_falls_back_to_voice_default(
    fake_service: SynthesisService,
) -> None:
    voice = fake_service.resolve_voice("vesper")  # it only
    assert fake_service.resolve_language(voice, None) == "it"


def test_resolve_language_setting_default(fake_runtime) -> None:
    settings = Settings(default_language="fr")
    service = SynthesisService(
        fake_runtime, settings, providers=["CPUExecutionProvider"], bundle_dir="."
    )
    voice = service.resolve_voice("bella")  # en_us + fr
    assert service.resolve_language(voice, None) == "fr"
    # explicit request wins over the server default
    assert service.resolve_language(voice, "en-US") == "en_us"


def test_resolve_language_rejected_for_voice(fake_service: SynthesisService) -> None:
    voice = fake_service.resolve_voice("vesper")  # it only
    with pytest.raises(VoiceResolutionError, match="for voice 'vesper'"):
        fake_service.resolve_language(voice, "fr")  # in the bundle, not for vesper


def test_resolve_language_unknown_in_bundle(fake_service: SynthesisService) -> None:
    voice = fake_service.resolve_voice("ariadne")
    with pytest.raises(VoiceResolutionError, match="Unsupported language"):
        fake_service.resolve_language(voice, "zh-CN")


def test_resolve_language_primary_subtag_fallback(fake_runtime) -> None:
    """HA sends 'de-DE'; a bundle language of plain 'de' must match."""
    runtime = FakeRuntime(voices=[make_voice_spec("klaus", ["de"])])
    service = SynthesisService(
        runtime, Settings(), providers=["CPUExecutionProvider"], bundle_dir="."
    )
    voice = service.resolve_voice("klaus")
    assert service.resolve_language(voice, "de-DE") == "de"


async def test_stream_yields_audio_pieces(fake_service: SynthesisService) -> None:
    pieces = [
        piece
        async for piece in fake_service.stream(text="Hello", voice_id="ariadne", language="en_us")
    ]
    assert len(pieces) == 1  # short text -> one piece
    assert pieces[0].sample_rate == FAKE_SAMPLE_RATE
    np.testing.assert_array_equal(pieces[0].audio, fake_audio_piece())


async def test_stream_passes_synthesis_settings(fake_runtime) -> None:
    settings = Settings(steps=4, sampler="euler", seed=7, speed=1.25, delivery="energy=2.5")
    service = SynthesisService(
        fake_runtime, settings, providers=["CPUExecutionProvider"], bundle_dir="."
    )
    async for _ in service.stream(text="Hi", voice_id="ariadne", language="en_us"):
        pass
    request = fake_runtime.requests[-1]
    assert request["steps"] == 4
    assert request["sampler"] == "euler"
    assert request["seed"] == 7
    assert request["speed"] == 1.25
    assert request["delivery"] == "energy=2.5"


def test_build_info(fake_service: SynthesisService) -> None:
    info = build_info(fake_service)
    assert len(info.tts) == 1
    program = info.tts[0]
    assert program.name == PROGRAM_NAME
    assert program.installed is True
    assert program.supports_synthesize_streaming is True
    assert program.version is not None
    by_name = {voice.name: voice for voice in program.voices}
    assert set(by_name) == {"ariadne", "bella", "vesper"}
    assert by_name["ariadne"].languages == ["en-US", "es"]
    assert by_name["vesper"].languages == ["it"]
    assert by_name["ariadne"].installed is True
    assert by_name["ariadne"].description == "Ariadne"
