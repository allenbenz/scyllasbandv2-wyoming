"""Docker/Podman e2e: build the image and drive it over the Wyoming protocol.

Disabled by default. Enable with:

    SCYLLASBAND_DOCKER_E2E=1 uv run pytest -m docker

Works with docker or podman (whichever is found on PATH first). Set
SCYLLASBAND_DOCKER_V2_MOUNT to a host directory that directly contains the
`onnx/` (and optionally `onnx-int8/`) bundle subdirectories to mount it at
/data/models/v2 and skip the ~500 MB in-container download; alternatively
SCYLLASBAND_MODEL_DIR mounts a full models dir at /data/models.
Set SCYLLASBAND_DOCKER_IMAGE to reuse a prebuilt tag.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.docker]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_IMAGE = os.environ.get("SCYLLASBAND_DOCKER_IMAGE", "scyllasbandv2-wyoming:e2e")
_START_TIMEOUT_SECONDS = float(os.environ.get("SCYLLASBAND_DOCKER_START_TIMEOUT", "900"))


def _container_cli() -> str:
    for cli in ("docker", "podman"):
        if shutil.which(cli):
            return cli
    raise RuntimeError("neither docker nor podman found on PATH")


if os.environ.get("SCYLLASBAND_DOCKER_E2E") != "1":
    pytest.skip(
        "set SCYLLASBAND_DOCKER_E2E=1 to run docker e2e tests",
        allow_module_level=True,
    )

try:
    _CLI = _container_cli()
except RuntimeError:
    pytest.skip("neither docker nor podman found on PATH", allow_module_level=True)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_for_wyoming(port: int, timeout: float) -> None:
    """Poll until the container answers ping + describe.

    Retries everything: the port proxy may accept connections before the
    server is listening (EOF on read) and the model load takes a while.
    """
    from wyoming.info import Describe, Info
    from wyoming.ping import Ping, Pong

    from tests.utils import connect

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        client = None
        try:
            client = await connect(port, read_timeout=5.0)
            await client.write_event(Ping(text="health").event())
            event = await client.read_event()
            if event is None or not Pong.is_type(event.type):
                last_error = EOFError("no pong")
            else:
                await client.write_event(Describe().event())
                event = await client.read_event()
                if event is not None and Info.is_type(event.type):
                    return
                last_error = EOFError("no info")
        except Exception as exc:  # refused / timeout / reset: retry until deadline
            last_error = exc
        finally:
            if client is not None:
                await client.disconnect()
        await asyncio.sleep(3)
    raise TimeoutError(f"container did not answer wyoming ping in {timeout}s: {last_error}")


async def test_container_serves_wyoming_tts() -> None:
    from wyoming.audio import AudioChunk, AudioStart
    from wyoming.tts import Synthesize

    from tests.utils import connect, read_until

    build_command = [_CLI, "build", "-t", _IMAGE]
    if _CLI == "podman":
        build_command += ["--format", "docker"]  # keep HEALTHCHECK (dropped in OCI)
    build_command.append(str(_PROJECT_ROOT))
    subprocess.run(build_command, check=True, cwd=str(_PROJECT_ROOT))

    port = _free_port()
    name = f"scyllasbandv2-wyoming-e2e-{port}"
    run_command = [
        _CLI,
        "run",
        "-d",
        "--rm",
        "--name",
        name,
        "-p",
        f"127.0.0.1:{port}:10200",
        "-e",
        "SCYLLASBAND_DEVICE=cpu",
    ]
    v2_mount = os.environ.get("SCYLLASBAND_DOCKER_V2_MOUNT")
    model_dir = os.environ.get("SCYLLASBAND_MODEL_DIR")
    if v2_mount:
        run_command += ["-v", f"{v2_mount}:/data/models/v2"]
    elif model_dir:
        run_command += ["-v", f"{model_dir}:/data/models"]
    run_command.append(_IMAGE)

    container = subprocess.check_output(run_command).decode().strip()
    try:
        await _wait_for_wyoming(port, _START_TIMEOUT_SECONDS)

        client = await connect(port, read_timeout=600.0)
        try:
            await client.write_event(Synthesize(text="Hello from the container.").event())
            events = await read_until(client, "audio-stop")
            start_event = next(event for event in events if AudioStart.is_type(event.type))
            start = AudioStart.from_event(start_event)
            assert (start.rate, start.width, start.channels) == (24000, 2, 1)
            audio = bytearray()
            for event in events:
                if AudioChunk.is_type(event.type):
                    audio.extend(AudioChunk.from_event(event).audio)
            assert len(audio) > 24000  # > 0.5 s of 16-bit mono audio
        finally:
            await client.disconnect()
    finally:
        subprocess.run([_CLI, "logs", "--tail", "50", container], check=False)
        subprocess.run([_CLI, "rm", "-f", container], check=False)
