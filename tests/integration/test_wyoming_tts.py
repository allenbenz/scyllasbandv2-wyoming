"""Integration tests: a real Wyoming TCP server backed by the fake runtime.

These exercise the full protocol path -- describe/ping, synthesize (buffered
and streaming), error handling, and audio byte-exactness -- without the model.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from wyoming.audio import AudioChunk, AudioStart
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info, SelectProgram
from wyoming.ping import Ping, Pong
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeVoice,
)

from scyllasbandv2_wyoming.config import Settings
from scyllasbandv2_wyoming.runtime import SynthesisService
from scyllasbandv2_wyoming.server import PROGRAM_NAME

from ..fakes import FakeRuntime, fake_audio_piece
from ..utils import connect, read_until, start_test_server


@pytest.fixture
async def server_and_client(fake_runtime: FakeRuntime, fake_settings: Settings):
    service = SynthesisService(
        fake_runtime,
        fake_settings,
        providers=["CPUExecutionProvider"],
        bundle_dir=fake_settings.model_dir / "v2" / fake_settings.bundle,
    )
    server, port = await start_test_server(service, fake_settings)
    client = await connect(port)
    try:
        yield client, fake_runtime, fake_settings
    finally:
        await client.disconnect()
        await server.stop()


def expected_pcm(gain: float = 1.0) -> bytes:
    audio = fake_audio_piece()
    scaled = audio * np.float32(gain) if gain != 1.0 else audio
    clipped = np.clip(scaled, np.float32(-1.0), np.float32(1.0))
    return (clipped * np.float32(32767.0)).astype("<i2").tobytes()


def collect_audio(events: list) -> bytes:
    audio = bytearray()
    for event in events:
        if AudioChunk.is_type(event.type):
            audio.extend(AudioChunk.from_event(event).audio)
    return bytes(audio)


async def test_ping_pong(server_and_client) -> None:
    client, _, _ = server_and_client
    await client.write_event(Ping(text="hello").event())
    event = await client.read_event()
    assert event is not None and Pong.is_type(event.type)
    assert Pong.from_event(event).text == "hello"


async def test_describe_lists_voices(server_and_client) -> None:
    client, _, _ = server_and_client
    await client.write_event(Describe().event())
    event = await client.read_event()
    assert event is not None and Info.is_type(event.type)
    info = Info.from_event(event)
    assert len(info.tts) == 1
    program = info.tts[0]
    assert program.name == PROGRAM_NAME
    assert program.supports_synthesize_streaming is True
    by_name = {voice.name: voice for voice in program.voices}
    assert set(by_name) == {"ariadne", "bella", "vesper"}
    assert by_name["ariadne"].languages == ["en-US", "es"]


async def test_synthesize_returns_exact_pcm(server_and_client) -> None:
    client, runtime, _ = server_and_client
    await client.write_event(
        Synthesize(text="Hello", voice=SynthesizeVoice(name="ariadne")).event()
    )
    events = await read_until(client, "audio-stop")
    start_event = next(event for event in events if AudioStart.is_type(event.type))
    start = AudioStart.from_event(start_event)
    assert (start.rate, start.width, start.channels) == (24000, 2, 1)

    assert collect_audio(events) == expected_pcm()
    assert runtime.requests[-1]["voice_id"] == "ariadne"
    assert runtime.requests[-1]["language"] == "en_us"


async def test_synthesize_splits_long_text_and_chunks_output(
    server_and_client,
) -> None:
    client, _, _ = server_and_client
    text = "This sentence is deliberately longer than forty characters."
    assert len(text) > 40
    await client.write_event(Synthesize(text=text).event())
    events = await read_until(client, "audio-stop")
    chunks = [event for event in events if AudioChunk.is_type(event.type)]
    # fake runtime yields two 100 ms pieces; output_chunk_ms=50 -> 2 chunks each
    assert len(chunks) == 4
    assert collect_audio(events) == expected_pcm() + expected_pcm()


async def test_synthesize_default_voice_and_language(server_and_client) -> None:
    client, runtime, _ = server_and_client
    await client.write_event(Synthesize(text="Hi").event())
    await read_until(client, "audio-stop")
    assert runtime.requests[-1]["voice_id"] == "ariadne"
    assert runtime.requests[-1]["language"] == "en_us"


async def test_synthesize_voice_language_mapping(server_and_client) -> None:
    """The wire drops `language` when `name` is set (name wins), so HA sends
    a language alone to pick the voice: es -> ariadne."""
    client, runtime, _ = server_and_client
    await client.write_event(Synthesize(text="Hola", voice=SynthesizeVoice(language="es")).event())
    await read_until(client, "audio-stop")
    assert runtime.requests[-1]["voice_id"] == "ariadne"
    assert runtime.requests[-1]["language"] == "es"


async def test_synthesize_language_only_picks_matching_voice(
    server_and_client,
) -> None:
    """HA can send a language without a voice; pick a voice that speaks it."""
    client, runtime, _ = server_and_client
    await client.write_event(
        Synthesize(text="Bonjour", voice=SynthesizeVoice(language="fr")).event()
    )
    await read_until(client, "audio-stop")
    assert runtime.requests[-1]["voice_id"] == "bella"
    assert runtime.requests[-1]["language"] == "fr"


async def test_synthesize_unknown_voice_error(server_and_client) -> None:
    client, _, _ = server_and_client
    await client.write_event(Synthesize(text="Hello", voice=SynthesizeVoice(name="nope")).event())
    event = await client.read_event()
    assert event is not None and Error.is_type(event.type)
    error = Error.from_event(event)
    assert error.code == "invalid-voice"
    assert "ariadne" in error.text
    # connection stays usable
    await client.write_event(Ping(text="still-alive").event())
    event = await client.read_event()
    assert event is not None and Pong.is_type(event.type)


async def test_synthesize_unsupported_language_error(server_and_client) -> None:
    client, _, _ = server_and_client
    # pt-BR is spoken by no voice: falls back to the default voice, whose
    # language check then fails with invalid-language
    await client.write_event(
        Synthesize(text="Ola", voice=SynthesizeVoice(language="pt-BR")).event()
    )
    event = await client.read_event()
    assert event is not None and Error.is_type(event.type)
    assert Error.from_event(event).code == "invalid-language"


async def test_synthesize_empty_text_error(server_and_client) -> None:
    client, _, _ = server_and_client
    await client.write_event(Synthesize(text="   ").event())
    event = await client.read_event()
    assert event is not None and Error.is_type(event.type)
    assert Error.from_event(event).code == "empty-text"


async def test_synthesize_ssml_rejected(server_and_client) -> None:
    client, _, _ = server_and_client
    await client.write_event(Synthesize(text="<speak>hi</speak>", text_format="ssml").event())
    event = await client.read_event()
    assert event is not None and Error.is_type(event.type)
    assert Error.from_event(event).code == "text-format-not-supported"


async def test_streaming_synthesize_flow(server_and_client) -> None:
    client, runtime, _ = server_and_client
    await client.write_event(SynthesizeStart(voice=SynthesizeVoice(name="bella")).event())
    await client.write_event(SynthesizeChunk(text="It is a real ").event())
    await client.write_event(SynthesizeChunk(text="pleasure to meet you.").event())
    # backwards-compat echo of the full text inside the streaming flow
    await client.write_event(Synthesize(text="It is a real pleasure to meet you.").event())
    await client.write_event(SynthesizeStop().event())

    events = await read_until(client, "synthesize-stopped")
    assert SynthesizeStopped.is_type(events[-1].type)
    assert collect_audio(events) == expected_pcm()  # 34 chars -> single piece
    assert runtime.requests[-1]["voice_id"] == "bella"
    assert runtime.requests[-1]["text"] == "It is a real pleasure to meet you."
    # exactly one synthesis for the whole streaming exchange
    assert len(runtime.requests) == 1


async def test_settings_passthrough_to_runtime(fake_runtime: FakeRuntime, tmp_path) -> None:
    settings = Settings(
        model_dir=tmp_path,
        output_chunk_ms=100,
        steps=3,
        sampler="euler",
        seed=11,
        speed=1.5,
        delivery="whisper=on",
    )
    service = SynthesisService(
        fake_runtime,
        settings,
        providers=["CPUExecutionProvider"],
        bundle_dir=tmp_path / "v2" / "onnx",
    )
    server, port = await start_test_server(service, settings)
    client = await connect(port)
    try:
        await client.write_event(Synthesize(text="Passthrough").event())
        await read_until(client, "audio-stop")
    finally:
        await client.disconnect()
        await server.stop()
    request = fake_runtime.requests[-1]
    assert request["steps"] == 3
    assert request["sampler"] == "euler"
    assert request["seed"] == 11
    assert request["speed"] == 1.5
    assert request["delivery"] == "whisper=on"


async def test_energy_gain_scales_output(fake_runtime: FakeRuntime, tmp_path) -> None:
    settings = Settings(model_dir=tmp_path, energy_gain=0.5, output_chunk_ms=100)
    service = SynthesisService(
        fake_runtime,
        settings,
        providers=["CPUExecutionProvider"],
        bundle_dir=tmp_path / "v2" / "onnx",
    )
    server, port = await start_test_server(service, settings)
    client = await connect(port)
    try:
        await client.write_event(Synthesize(text="Quiet").event())
        events = await read_until(client, "audio-stop")
        audio = collect_audio(events)
        assert audio == expected_pcm(gain=0.5)
        peak = np.abs(np.frombuffer(audio, dtype="<i2")).max()
        # 0.5 amplitude * 0.5 gain -> ~8191, well below the 16383 of gain=1
        assert 7000 < peak < 9000
    finally:
        await client.disconnect()
        await server.stop()


async def test_select_program_ignored(server_and_client) -> None:
    client, _, _ = server_and_client
    await client.write_event(SelectProgram(name="something-else").event())
    await client.write_event(Ping(text="ok").event())
    event = await client.read_event()
    assert event is not None and Pong.is_type(event.type)


async def test_unknown_event_ignored(server_and_client) -> None:
    client, _, _ = server_and_client
    await client.write_event(Event(type="transcribe", data={}))
    await client.write_event(Ping(text="after").event())
    event = await client.read_event()
    assert event is not None and Pong.is_type(event.type)


async def test_synthesis_serialized_across_connections(fake_runtime: FakeRuntime, tmp_path) -> None:
    """Two concurrent clients must not interleave inside the model runtime."""
    settings = Settings(model_dir=tmp_path, output_chunk_ms=100)
    service = SynthesisService(
        fake_runtime,
        settings,
        providers=["CPUExecutionProvider"],
        bundle_dir=tmp_path / "v2" / "onnx",
    )
    server, port = await start_test_server(service, settings)
    clients = [await connect(port) for _ in range(2)]
    try:

        async def synthesize(client) -> bytes:
            await client.write_event(Synthesize(text="Concurrent").event())
            events = await read_until(client, "audio-stop")
            return collect_audio(events)

        results = await asyncio.gather(*(synthesize(client) for client in clients))
        assert all(result == expected_pcm() for result in results)
    finally:
        for client in clients:
            await client.disconnect()
        await server.stop()
