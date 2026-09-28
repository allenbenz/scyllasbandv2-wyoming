"""Server configuration: environment variables, CLI flags, and validation.

Every knob can be set through a ``SCYLLASBAND_*`` environment variable (the
Docker interface) or a CLI flag (flags win over the environment). Values the
Wyoming protocol controls per request -- text, voice name, voice language --
are deliberately *not* configurable here beyond server-side defaults.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

ENV_PREFIX = "SCYLLASBAND_"

VALID_BUNDLES = ("onnx", "onnx-int8")
VALID_DOWNLOAD_MODES = ("auto", "always", "never")
VALID_DEVICES = ("auto", "cuda", "cpu")
VALID_SAMPLERS = ("euler", "heun")
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

MAX_STEPS = 64
MIN_OUTPUT_CHUNK_MS = 20


@dataclass(frozen=True)
class Settings:
    """Validated server settings."""

    uri: str = "tcp://0.0.0.0:10200"
    model_dir: Path = Path("data/models")
    bundle: str = "onnx"
    download: str = "auto"
    revision: str | None = None
    device: str = "auto"
    intra_op_threads: int = 0
    inter_op_threads: int = 0
    default_voice: str | None = None
    default_language: str | None = None
    sampler: str = "heun"
    steps: int = 8
    seed: int | None = None
    speed: float = 1.0
    energy_gain: float = 1.0
    delivery: str | None = None
    output_chunk_ms: int = 200
    warmup: bool = True
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise ValueError("uri must not be empty")
        if self.bundle not in VALID_BUNDLES:
            raise ValueError(f"bundle must be one of {VALID_BUNDLES}, got {self.bundle!r}")
        if self.download not in VALID_DOWNLOAD_MODES:
            raise ValueError(
                f"download must be one of {VALID_DOWNLOAD_MODES}, got {self.download!r}"
            )
        if self.device not in VALID_DEVICES:
            raise ValueError(f"device must be one of {VALID_DEVICES}, got {self.device!r}")
        if self.sampler not in VALID_SAMPLERS:
            raise ValueError(f"sampler must be one of {VALID_SAMPLERS}, got {self.sampler!r}")
        if not 1 <= self.steps <= MAX_STEPS:
            raise ValueError(f"steps must be between 1 and {MAX_STEPS}, got {self.steps}")
        if self.speed <= 0:
            raise ValueError(f"speed must be > 0, got {self.speed}")
        if self.energy_gain < 0:
            raise ValueError(f"energy_gain must be >= 0, got {self.energy_gain}")
        if self.output_chunk_ms < MIN_OUTPUT_CHUNK_MS:
            raise ValueError(
                f"output_chunk_ms must be >= {MIN_OUTPUT_CHUNK_MS}, got {self.output_chunk_ms}"
            )
        if self.intra_op_threads < 0:
            raise ValueError(f"intra_op_threads must be >= 0, got {self.intra_op_threads}")
        if self.inter_op_threads < 0:
            raise ValueError(f"inter_op_threads must be >= 0, got {self.inter_op_threads}")
        if self.log_level not in VALID_LOG_LEVELS:
            raise ValueError(f"log_level must be one of {VALID_LOG_LEVELS}, got {self.log_level!r}")


def _raw(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(ENV_PREFIX + name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_int(env: Mapping[str, str], name: str) -> int | None:
    value = _raw(env, name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{ENV_PREFIX}{name} must be an integer, got {value!r}") from None


def _env_float(env: Mapping[str, str], name: str) -> float | None:
    value = _raw(env, name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{ENV_PREFIX}{name} must be a number, got {value!r}") from None


def _env_bool(env: Mapping[str, str], name: str) -> bool | None:
    value = _raw(env, name)
    if value is None:
        return None
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{ENV_PREFIX}{name} must be a boolean, got {value!r}")


def settings_from_env(env: Mapping[str, str] | None = None) -> Settings:
    """Build settings from environment variables (defaults for the rest)."""
    source = os.environ if env is None else env
    kwargs: dict[str, Any] = {}

    if text := _raw(source, "URI"):
        kwargs["uri"] = text
    if text := _raw(source, "MODEL_DIR"):
        kwargs["model_dir"] = Path(text)
    if text := _raw(source, "BUNDLE"):
        kwargs["bundle"] = text.lower()
    if text := _raw(source, "DOWNLOAD"):
        kwargs["download"] = text.lower()
    kwargs["revision"] = _raw(source, "REVISION")
    if text := _raw(source, "DEVICE"):
        kwargs["device"] = text.lower()
    if (count := _env_int(source, "INTRA_OP_THREADS")) is not None:
        kwargs["intra_op_threads"] = count
    if (count := _env_int(source, "INTER_OP_THREADS")) is not None:
        kwargs["inter_op_threads"] = count
    kwargs["default_voice"] = _raw(source, "VOICE")
    kwargs["default_language"] = _raw(source, "LANGUAGE")
    if text := _raw(source, "SAMPLER"):
        kwargs["sampler"] = text.lower()
    if (count := _env_int(source, "STEPS")) is not None:
        kwargs["steps"] = count
    kwargs["seed"] = _env_int(source, "SEED")
    if (factor := _env_float(source, "SPEED")) is not None:
        kwargs["speed"] = factor
    if (factor := _env_float(source, "ENERGY_GAIN")) is not None:
        kwargs["energy_gain"] = factor
    kwargs["delivery"] = _raw(source, "DELIVERY")
    if (count := _env_int(source, "OUTPUT_CHUNK_MS")) is not None:
        kwargs["output_chunk_ms"] = count
    if (flag := _env_bool(source, "WARMUP")) is not None:
        kwargs["warmup"] = flag
    if text := _raw(source, "LOG_LEVEL"):
        kwargs["log_level"] = text.upper()

    return Settings(**kwargs)


def build_arg_parser(env: Mapping[str, str] | None = None) -> argparse.ArgumentParser:
    """Argument parser whose defaults come from the environment."""
    base = settings_from_env(env)
    parser = argparse.ArgumentParser(
        prog="scyllasbandv2-wyoming",
        description="Wyoming-protocol TTS server for the Scylla's Band v2 model",
    )
    parser.add_argument(
        "--uri",
        default=base.uri,
        metavar="URI",
        help="Wyoming listen URI: tcp://HOST:PORT, unix://PATH or stdio:// (env SCYLLASBAND_URI)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=base.model_dir,
        metavar="DIR",
        help="directory holding (or receiving) the downloaded bundles (env SCYLLASBAND_MODEL_DIR)",
    )
    parser.add_argument(
        "--bundle",
        choices=VALID_BUNDLES,
        default=base.bundle,
        help="model bundle variant (env SCYLLASBAND_BUNDLE)",
    )
    parser.add_argument(
        "--download",
        choices=VALID_DOWNLOAD_MODES,
        default=base.download,
        help=(
            "download behaviour: auto when missing, always at startup, never "
            "(env SCYLLASBAND_DOWNLOAD)"
        ),
    )
    parser.add_argument(
        "--revision",
        default=base.revision,
        metavar="REV",
        help="Hugging Face revision to download (env SCYLLASBAND_REVISION)",
    )
    parser.add_argument(
        "--device",
        choices=VALID_DEVICES,
        default=base.device,
        help="execution device: auto prefers CUDA and falls back to CPU (env SCYLLASBAND_DEVICE)",
    )
    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=base.intra_op_threads,
        metavar="N",
        help="onnxruntime intra-op threads, 0 = runtime default (env SCYLLASBAND_INTRA_OP_THREADS)",
    )
    parser.add_argument(
        "--inter-op-threads",
        type=int,
        default=base.inter_op_threads,
        metavar="N",
        help="onnxruntime inter-op threads, 0 = runtime default (env SCYLLASBAND_INTER_OP_THREADS)",
    )
    parser.add_argument(
        "--voice",
        dest="default_voice",
        default=base.default_voice,
        metavar="ID",
        help="default voice id for requests without one (env SCYLLASBAND_VOICE)",
    )
    parser.add_argument(
        "--language",
        dest="default_language",
        default=base.default_language,
        metavar="LANG",
        help="default language for requests without one (env SCYLLASBAND_LANGUAGE)",
    )
    parser.add_argument(
        "--sampler",
        choices=VALID_SAMPLERS,
        default=base.sampler,
        help="rectified-flow sampler (env SCYLLASBAND_SAMPLER)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=base.steps,
        metavar="N",
        help=f"rectified-flow steps, 1-{MAX_STEPS} (env SCYLLASBAND_STEPS)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=base.seed,
        metavar="N",
        help="fixed sampling seed for reproducible output (env SCYLLASBAND_SEED)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=base.speed,
        metavar="X",
        help="speaking-rate scale from the model (env SCYLLASBAND_SPEED)",
    )
    parser.add_argument(
        "--energy-gain",
        type=float,
        default=base.energy_gain,
        metavar="X",
        help="post-synthesis waveform gain, linear (env SCYLLASBAND_ENERGY_GAIN)",
    )
    parser.add_argument(
        "--delivery",
        default=base.delivery,
        metavar="SPEC",
        help=(
            'default measured-delivery string, e.g. "energy=2.3,whisper=off" '
            "(env SCYLLASBAND_DELIVERY)"
        ),
    )
    parser.add_argument(
        "--output-chunk-ms",
        type=int,
        default=base.output_chunk_ms,
        metavar="MS",
        help="max milliseconds of audio per audio-chunk event (env SCYLLASBAND_OUTPUT_CHUNK_MS)",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=base.warmup,
        help=(
            "run a warmup synthesis at startup so the first request is fast "
            "(env SCYLLASBAND_WARMUP)"
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=VALID_LOG_LEVELS,
        default=base.log_level,
        help="logging verbosity (env SCYLLASBAND_LOG_LEVEL)",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Build settings from a parsed namespace (whose defaults came from env)."""
    names = {field.name for field in fields(Settings)}
    kwargs = {name: getattr(args, name) for name in names if hasattr(args, name)}
    return Settings(**kwargs)
