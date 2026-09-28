"""Helpers for starting a real Wyoming TCP server backed by a fake runtime."""

from __future__ import annotations

import asyncio
import socket

from wyoming.client import AsyncClient
from wyoming.event import Event
from wyoming.server import AsyncTcpServer

from scyllasbandv2_wyoming.config import Settings
from scyllasbandv2_wyoming.runtime import SynthesisService
from scyllasbandv2_wyoming.server import make_handler_factory


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def start_test_server(
    service: SynthesisService, settings: Settings
) -> tuple[AsyncTcpServer, int]:
    """Start a Wyoming TCP server on an ephemeral port. Caller must stop()."""
    from scyllasbandv2_wyoming.server import build_info

    info = build_info(service)
    port = free_port()
    server = AsyncTcpServer("127.0.0.1", port)
    factory = make_handler_factory(service, info, settings, asyncio.Lock())
    await server.start(factory)
    return server, port


async def connect(port: int, read_timeout: float = 10.0) -> AsyncClient:
    client = AsyncClient.from_uri(
        f"tcp://127.0.0.1:{port}", connect_timeout=5.0, read_timeout=read_timeout
    )
    await client.connect()
    return client


async def read_until(client: AsyncClient, *stop_types: str) -> list[Event]:
    """Read events until one of ``stop_types`` arrives; returns everything read."""
    events: list[Event] = []
    while True:
        event = await client.read_event()
        assert event is not None, "connection closed before expected event"
        events.append(event)
        if event.type in stop_types:
            return events
