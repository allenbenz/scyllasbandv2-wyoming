"""Server bootstrap: build the wyoming Info message and run the TCP server."""

from __future__ import annotations

import asyncio
import logging
from importlib.metadata import PackageNotFoundError, version

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from . import __version__
from .audio import model_language_to_bcp47
from .config import Settings
from .handler import ScyllasbandEventHandler
from .runtime import SynthesisService

_LOGGER = logging.getLogger(__name__)

PROGRAM_NAME = "scyllasbandv2-wyoming"
PROGRAM_URL = "https://github.com/allenbenz/scyllasbandv2-wyoming"
MODEL_ATTRIBUTION = Attribution(
    name="Scylla's Band v2",
    url="https://huggingface.co/spybyscript/scyllasbandv2",
)


def program_version() -> str:
    try:
        return version("scyllasbandv2-wyoming")
    except PackageNotFoundError:
        return __version__


def build_info(service: SynthesisService) -> Info:
    """Describe the service for Home Assistant's Wyoming integration."""
    using_cuda = any("CUDA" in provider for provider in service.providers)
    voices = [
        TtsVoice(
            name=str(voice.id),
            attribution=MODEL_ATTRIBUTION,
            installed=True,
            description=str(voice.name),
            version=None,
            languages=sorted({model_language_to_bcp47(lang) for lang in voice.languages}),
        )
        for voice in service.voices()
    ]
    program = TtsProgram(
        name=PROGRAM_NAME,
        attribution=Attribution(name="scyllasbandv2-wyoming", url=PROGRAM_URL),
        installed=True,
        description=(
            f"Scylla's Band v2 text to speech (onnxruntime, {'CUDA' if using_cuda else 'CPU'})"
        ),
        version=program_version(),
        voices=voices,
        supports_synthesize_streaming=True,
    )
    return Info(tts=[program])


def make_handler_factory(
    service: SynthesisService,
    info: Info,
    settings: Settings,
    synth_lock: asyncio.Lock,
):
    """HandlerFactory closure wiring shared state into every connection."""

    def factory(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> ScyllasbandEventHandler:
        return ScyllasbandEventHandler(
            reader,
            writer,
            service=service,
            info=info,
            settings=settings,
            synth_lock=synth_lock,
            program_name=PROGRAM_NAME,
        )

    return factory


async def run_server(settings: Settings) -> None:
    """Load the model, then serve Wyoming events until shutdown."""
    service = await SynthesisService.create(settings)
    _LOGGER.info("voices: %s", ", ".join(str(voice.id) for voice in service.voices()))
    info = build_info(service)
    server = AsyncServer.from_uri(settings.uri)
    _LOGGER.info("listening on %s", settings.uri)
    await server.run(make_handler_factory(service, info, settings, asyncio.Lock()))
