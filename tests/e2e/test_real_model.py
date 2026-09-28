"""End-to-end tests against the real Scylla's Band v2 bundle.

Disabled by default because they download ~500 MB and run real inference.
Enable with:

    SCYLLASBAND_E2E=1 uv run pytest -m e2e

Additional SCYLLASBAND_* environment variables (MODEL_DIR, DEVICE, BUNDLE,
STEPS, ...) are honoured so you can reuse a cached bundle or force CPU/GPU.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from wyoming.audio import AudioChunk, AudioStart
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize

from scyllasbandv2_wyoming.config import settings_from_env
from scyllasbandv2_wyoming.runtime import SynthesisService

from ..utils import connect, read_until, start_test_server

pytestmark = [pytest.mark.e2e]

if os.environ.get("SCYLLASBAND_E2E") != "1":
    pytest.skip("set SCYLLASBAND_E2E=1 to run e2e tests", allow_module_level=True)

EXPECTED_MIN_VOICES = 8
MIN_SPEECH_SECONDS = 0.5


async def test_real_model_serves_wyoming_tts() -> None:
    settings = settings_from_env()
    service = await SynthesisService.create(settings)
    assert service.sample_rate == 24000
    if settings.device == "cuda":
        assert any("CUDA" in provider for provider in service.providers)

    server, port = await start_test_server(service, settings)
    client = await connect(port, read_timeout=300.0)
    try:
        # describe
        await client.write_event(Describe().event())
        event = await client.read_event()
        assert event is not None and Info.is_type(event.type)
        program = Info.from_event(event).tts[0]
        assert len(program.voices) >= EXPECTED_MIN_VOICES

        # synthesize
        await client.write_event(Synthesize(text="It is a real pleasure to meet you.").event())
        events = await read_until(client, "audio-stop")
        start_event = next(event for event in events if AudioStart.is_type(event.type))
        start = AudioStart.from_event(start_event)
        assert (start.rate, start.width, start.channels) == (24000, 2, 1)

        audio = bytearray()
        for item in events:
            if AudioChunk.is_type(item.type):
                audio.extend(AudioChunk.from_event(item).audio)
        duration_seconds = len(audio) / 2 / 24000
        assert duration_seconds > MIN_SPEECH_SECONDS
        peak = np.abs(np.frombuffer(bytes(audio), dtype="<i2")).max()
        assert peak > 1000, "synthesized audio looks silent"
    finally:
        await client.disconnect()
        await server.stop()
