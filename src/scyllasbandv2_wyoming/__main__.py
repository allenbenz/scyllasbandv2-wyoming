"""CLI entry point: ``scyllasbandv2-wyoming`` or ``python -m scyllasbandv2_wyoming``."""

from __future__ import annotations

import asyncio
import logging
import sys

from .config import build_arg_parser, settings_from_args
from .server import run_server


def run(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = settings_from_args(args)
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # onnxruntime is chatty at INFO; keep our own INFO signal readable
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)
    try:
        asyncio.run(run_server(settings))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
