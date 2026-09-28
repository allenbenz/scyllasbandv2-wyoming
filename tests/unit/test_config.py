"""Unit tests for Settings: env parsing, CLI defaults/overrides, validation."""

from __future__ import annotations

import pytest

from scyllasbandv2_wyoming.config import (
    ENV_PREFIX,
    Settings,
    build_arg_parser,
    settings_from_args,
    settings_from_env,
)


def test_defaults() -> None:
    settings = Settings()
    assert settings.uri == "tcp://0.0.0.0:10200"
    assert settings.model_version == "v2"
    assert settings.bundle == "onnx"
    assert settings.download == "auto"
    assert settings.device == "auto"
    assert settings.sampler == "heun"
    assert settings.steps == 8
    assert settings.seed is None
    assert settings.speed == 1.0
    assert settings.energy_gain == 1.0
    assert settings.delivery is None
    assert settings.output_chunk_ms == 200
    assert settings.warmup is True
    assert settings.log_level == "INFO"


def test_env_roundtrip() -> None:
    env = {
        f"{ENV_PREFIX}URI": "tcp://127.0.0.1:10500",
        f"{ENV_PREFIX}MODEL_DIR": "/tmp/models",
        f"{ENV_PREFIX}BUNDLE": "onnx-int8",
        f"{ENV_PREFIX}REVISION": "",
        f"{ENV_PREFIX}DEVICE": "cuda",
        f"{ENV_PREFIX}INTRA_OP_THREADS": "4",
        f"{ENV_PREFIX}VOICE": "ariadne",
        f"{ENV_PREFIX}LANGUAGE": "en-US",
        f"{ENV_PREFIX}SAMPLER": "euler",
        f"{ENV_PREFIX}STEPS": "12",
        f"{ENV_PREFIX}SEED": "7",
        f"{ENV_PREFIX}SPEED": "1.2",
        f"{ENV_PREFIX}MAX_CHUNK_SECONDS": "16",
        f"{ENV_PREFIX}ENERGY_GAIN": "0.5",
        f"{ENV_PREFIX}DELIVERY": "energy=2.3,whisper=off",
        f"{ENV_PREFIX}OUTPUT_CHUNK_MS": "100",
        f"{ENV_PREFIX}WARMUP": "false",
        f"{ENV_PREFIX}LOG_LEVEL": "debug",
    }
    settings = settings_from_env(env)
    assert settings.uri == "tcp://127.0.0.1:10500"
    assert settings.model_dir.as_posix() == "/tmp/models"
    assert settings.bundle == "onnx-int8"
    assert settings.revision is None  # empty means unset
    assert settings.device == "cuda"
    assert settings.intra_op_threads == 4
    assert settings.default_voice == "ariadne"
    assert settings.default_language == "en-US"
    assert settings.sampler == "euler"
    assert settings.steps == 12
    assert settings.seed == 7
    assert settings.speed == 1.2
    assert settings.max_chunk_seconds == 16
    assert settings.energy_gain == 0.5
    assert settings.delivery == "energy=2.3,whisper=off"
    assert settings.v1_emotion is None
    assert settings.v1_emotion_scale == 1.0
    assert settings.output_chunk_ms == 100
    assert settings.warmup is False
    assert settings.log_level == "DEBUG"


def test_env_v1_roundtrip() -> None:
    env = {
        f"{ENV_PREFIX}MODEL_VERSION": "v1",
        f"{ENV_PREFIX}V1_EMOTION": "joy=0.7,sarcasm=0.3",
        f"{ENV_PREFIX}V1_EMOTION_SCALE": "1.5",
    }
    settings = settings_from_env(env)
    assert settings.model_version == "v1"
    assert settings.v1_emotion == "joy=0.7,sarcasm=0.3"
    assert settings.v1_emotion_scale == 1.5
    assert settings.delivery is None


def test_env_rejects_delivery_with_v1() -> None:
    env = {
        f"{ENV_PREFIX}MODEL_VERSION": "v1",
        f"{ENV_PREFIX}DELIVERY": "energy=2.3",
    }
    with pytest.raises(ValueError, match="delivery requires model_version=v2"):
        settings_from_env(env)


def test_env_rejects_v1_emotion_with_v2() -> None:
    env = {
        f"{ENV_PREFIX}MODEL_VERSION": "v2",
        f"{ENV_PREFIX}V1_EMOTION": "joy=0.7",
    }
    with pytest.raises(ValueError, match="v1_emotion requires model_version=v1"):
        settings_from_env(env)


def test_v1_v2_cross_validation() -> None:
    # delivery is v2-only
    with pytest.raises(ValueError, match="delivery requires model_version=v2"):
        Settings(model_version="v1", delivery="energy=2.3")
    # v1 emotion knobs are v1-only
    with pytest.raises(ValueError, match="v1_emotion requires model_version=v1"):
        Settings(model_version="v2", v1_emotion="joy=0.7")
    with pytest.raises(ValueError, match="v1_emotion_scale requires model_version=v1"):
        Settings(model_version="v2", v1_emotion_scale=1.5)
    # v1 accepts its own knobs together
    Settings(model_version="v1", v1_emotion="joy=0.7", v1_emotion_scale=1.5)
    # v2 accepts its own
    Settings(model_version="v2", delivery="energy=2.3")


def test_v1_emotion_scale_negative_rejected() -> None:
    with pytest.raises(ValueError, match="v1_emotion_scale"):
        Settings(model_version="v1", v1_emotion_scale=-0.5)


def test_env_bad_int() -> None:
    with pytest.raises(ValueError, match="STEPS"):
        settings_from_env({f"{ENV_PREFIX}STEPS": "lots"})


def test_env_bad_bool() -> None:
    with pytest.raises(ValueError, match="WARMUP"):
        settings_from_env({f"{ENV_PREFIX}WARMUP": "maybe"})


def test_env_max_chunk_seconds_zero_is_allowed() -> None:
    settings = settings_from_env({f"{ENV_PREFIX}MAX_CHUNK_SECONDS": "0"})
    assert settings.max_chunk_seconds == 0  # 0 = cap disabled


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_version", "v3"),
        ("bundle", "onnx-fp16"),
        ("download", "sometimes"),
        ("device", "tpu"),
        ("sampler", "dopri"),
        ("steps", 0),
        ("steps", 65),
        ("speed", 0.0),
        ("max_chunk_seconds", -1.0),
        ("max_chunk_seconds", 61.0),
        ("energy_gain", -1.0),
        ("output_chunk_ms", 1),
        ("intra_op_threads", -2),
        ("log_level", "LOUD"),
    ],
)
def test_validation_rejects_bad_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        Settings(**{field: value})


def test_cli_parser_defaults_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"{ENV_PREFIX}STEPS", "9")
    monkeypatch.setenv(f"{ENV_PREFIX}VOICE", "bella")
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.steps == 9
    assert args.default_voice == "bella"
    settings = settings_from_args(args)
    assert settings.steps == 9
    assert settings.default_voice == "bella"


def test_cli_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"{ENV_PREFIX}STEPS", "9")
    monkeypatch.setenv(f"{ENV_PREFIX}DEVICE", "cpu")
    parser = build_arg_parser()
    args = parser.parse_args(["--steps", "4", "--device", "cuda"])
    settings = settings_from_args(args)
    assert settings.steps == 4
    assert settings.device == "cuda"


def test_cli_model_version_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"{ENV_PREFIX}MODEL_VERSION", "v1")
    args = build_arg_parser().parse_args([])
    assert settings_from_args(args).model_version == "v1"


def test_cli_no_warmup_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"{ENV_PREFIX}WARMUP", "true")
    args = build_arg_parser().parse_args(["--no-warmup"])
    assert settings_from_args(args).warmup is False
