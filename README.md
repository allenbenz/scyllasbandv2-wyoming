# scyllasbandv2-wyoming

A [Wyoming](https://github.com/rhasspy/wyoming)-protocol text-to-speech server for **Scylla's Band v2**, a duration-flow TTS model: a phone-duration predictor feeds a rectified-flow latent generator, followed by a Vocos vocoder. Audio is emitted as 16-bit PCM at 24 kHz mono. The model ships 10 voices across 6 languages (`en_us`, `en_gb`, `es`, `it`, `fr`, `de`, `vi`; advertised to clients as `en-US`, `en-GB`, `es`, `it`, `fr`, `de`, `vi`) and supports *measured delivery* style controls — `energy`, `tension`, `valence`, `assertiveness` on a 0–4 scale (2 = neutral) plus a `whisper` on/off toggle.

Inference runs on CPU or NVIDIA GPU via **onnxruntime** with the CUDA Execution Provider. Note: inference is ONNX only — the PyTorch checkpoints in the model repository are upstream training artifacts and are not used by this server.

## How it works

- Answers Wyoming `describe` and `ping` requests so Home Assistant can discover the server and its voices.
- Handles both one-shot `synthesize` and streaming `synthesize-start` / `synthesize-chunk` / `synthesize-stop` requests.
- Emits `audio-start`, `audio-chunk`, and `audio-stop` events carrying 16-bit PCM at 24 kHz mono.
- Serializes synthesis across connections: one request is processed at a time on the model.

## Requirements

- Python 3.11+ (3.14 recommended)
- ~1 GB disk space for the downloaded model bundle
- Optional: NVIDIA GPU with drivers (CUDA 13.0 + cuDNN 9 for `onnxruntime-gpu` ≥ 1.30, i.e. driver ≥ 580)

## Quickstart (local)

Requires [mise](https://mise.jdx.dev/) (which provides Python 3.14 and uv):

```sh
mise run install                      # or: mise exec -- uv sync
mise exec -- uv run scyllasbandv2-wyoming
```

The first run downloads the model bundle from Hugging Face (`spybyscript/scyllasbandv2`, pinned to revision `v2-measured-20260922`, ~498 MB for the FP32 `onnx` bundle or ~316 MB for `onnx-int8`) into the model directory. The server then listens on `tcp://0.0.0.0:10200`.

## Docker

```sh
docker run -p 10200:10200 -v scyllasbandv2-models:/data ghcr.io/allenbenz/scyllasbandv2-wyoming:latest
```

GPU variant (requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

```sh
docker run --gpus all -p 10200:10200 -v scyllasbandv2-models:/data ghcr.io/allenbenz/scyllasbandv2-wyoming:latest
```

The image is **linux/amd64 only** — `onnxruntime-gpu` publishes no arm64 wheels.

## Model versions

The server can serve either generation of Scylla's Band; bundles live side by side under `<model-dir>/v1/` and `<model-dir>/v2/`, so you can run one container per version if you want both.

| | `v2` (default) | `v1` |
| --- | --- | --- |
| Style conditioning | Measured delivery axes: `energy`, `tension`, `valence`, `assertiveness` (0–4) + `whisper` (`SCYLLASBAND_DELIVERY`) | Continuous affect axes `calm, joy, anger, sadness, sarcasm, questioning` (0–1) or named presets (`SCYLLASBAND_V1_EMOTION`) + CFG scale (`SCYLLASBAND_V1_EMOTION_SCALE`) |
| Cross-chunk continuity | No prefix carryover (the released v2 graphs do not accept prefix latents) | **Prefix-latent carryover enabled** — chunk N is conditioned on the last ~24 latent frames of chunk N−1, keeping pitch/register steadier across chunk seams |
| Languages | `en-US`, `en-GB`, `es`, `it`, `fr`, `de`, `vi` | `en-US`, `en-GB`, `es`, `it` |
| Largest chunk (latent bucket) | 1024 frames ≈ 21.3 s | 512 frames ≈ 10.9 s (`SCYLLASBAND_MAX_CHUNK_SECONDS` above this is preflight-split, with a startup warning) |
| Training | Newer "measured delivery" training run | Legacy run |

Voice names (ariadne, felix, gwen, …) and the audio contract (24 kHz mono) are identical across versions.

```sh
# v1 with continuous affect axes and stronger CFG
docker run --gpus all -p 10200:10200 -v scyllasbandv2-models:/data \
  -e SCYLLASBAND_MODEL_VERSION=v1 -e SCYLLASBAND_V1_EMOTION="joy=0.75,sarcasm=0.25" \
  ghcr.io/allenbenz/scyllasbandv2-wyoming:latest
```

The v1 emotion spec is validated at startup (a discarded one-step probe render) — an invalid axis name or out-of-range value fails fast with the valid options listed.

## Home Assistant

1. Go to **Settings → Devices & Services → Add Integration → Wyoming Protocol**.
2. Enter the server's host and port `10200`.
3. Voices appear under the TTS entity dropdown.

Voice and language are chosen **per request** from Home Assistant — they are not environment variables. The `SCYLLASBAND_VOICE` / `SCYLLASBAND_LANGUAGE` settings below are only server-side defaults used when a request omits them.

## Configuration

Every setting is available as an environment variable and as a CLI flag of the same meaning (`--uri`, `--model-dir`, `--model-version`, `--bundle`, `--download`, `--revision`, `--device`, `--intra-op-threads`, `--inter-op-threads`, `--voice`, `--language`, `--sampler`, `--steps`, `--seed`, `--speed`, `--max-chunk-seconds`, `--energy-gain`, `--delivery`, `--v1-emotion`, `--v1-emotion-scale`, `--output-chunk-ms`, `--no-warmup`, `--log-level`). CLI flags override environment variables.

| Variable | Default | Description |
| --- | --- | --- |
| `SCYLLASBAND_URI` | `tcp://0.0.0.0:10200` | Wyoming listen URI (`tcp://`, `unix://`, or `stdio://`) |
| `SCYLLASBAND_MODEL_DIR` | `data/models` (`/data/models` in docker) | Where model bundles are downloaded and looked up |
| `SCYLLASBAND_MODEL_VERSION` | `v2` | Model generation to serve: `v2` (measured delivery) or `v1` (emotion presets + prefix-latent carryover). See [Model versions](#model-versions) |
| `SCYLLASBAND_BUNDLE` | `onnx` | Bundle variant: `onnx` (~498 MB, FP32) or `onnx-int8` (~316 MB) |
| `SCYLLASBAND_DOWNLOAD` | `auto` | `auto` downloads when missing, `always` re-downloads at startup, `never` fails if the bundle is missing |
| `SCYLLASBAND_REVISION` | *(empty = runtime default `v2-measured-20260922`)* | Hugging Face revision to pin |
| `HF_TOKEN` | *(empty)* | Passed through to `huggingface_hub` for gated or rate-limited downloads |
| `SCYLLASBAND_DEVICE` | `auto` | `auto` prefers CUDA and falls back to CPU; `cuda` fails fast if the CUDA EP is unavailable; `cpu` forces CPU |
| `SCYLLASBAND_INTRA_OP_THREADS` | `0` (onnxruntime default) | onnxruntime intra-op threads |
| `SCYLLASBAND_INTER_OP_THREADS` | `0` (onnxruntime default) | onnxruntime inter-op threads |
| `SCYLLASBAND_VOICE` | *(empty = first bundle voice)* | Default voice id for requests that don't specify one (e.g. `ariadne`) |
| `SCYLLASBAND_LANGUAGE` | *(empty = voice default)* | Default language for requests without one |
| `SCYLLASBAND_SAMPLER` | `heun` | Rectified-flow sampler: `euler` or `heun` |
| `SCYLLASBAND_STEPS` | `8` | Rectified-flow steps (quality/latency tradeoff, 1–64) |
| `SCYLLASBAND_SEED` | *(empty = random per request)* | Deterministic seeding |
| `SCYLLASBAND_SPEED` | `1.0` | Speaking-rate scale from the model |
| `SCYLLASBAND_MAX_CHUNK_SECONDS` | `8` | Max predicted seconds of audio per planner chunk (the upstream default). Fewer/longer chunks = fewer seams = steadier intonation across sentences, at the cost of first-audio latency; `0` disables the seconds cap (the character limit still applies). The v2 graph ceiling is ~21 s, v1 ~10.9 s; overlong chunks are split automatically. |
| `SCYLLASBAND_ENERGY_GAIN` | `1.0` | Post-synthesis waveform gain (linear; the delivery `energy` axis is separate and learned) |
| `SCYLLASBAND_DELIVERY` | *(empty = neutral)* | **v2 only.** Default measured-delivery string, e.g. `"energy=2.3,tension=1.7,valence=2.6,whisper=off"`; axes `energy`/`tension`/`valence`/`assertiveness` are 0–4 (2 = neutral), `whisper` is `on`/`off` |
| `SCYLLASBAND_V1_EMOTION` | *(empty = neutral)* | **v1 only.** Emotion spec, either continuous axes `"joy=0.75,sarcasm=0.25"` (axes: `calm, joy, anger, sadness, sarcasm, questioning`, values 0–1) or a named preset (`neutral, friendly, happy, excited, angry, frustrated, sad, sarcastic, unsure`) |
| `SCYLLASBAND_V1_EMOTION_SCALE` | `1.0` | **v1 only.** Emotion CFG strength: `1` = direct, `0` = null |
| `SCYLLASBAND_OUTPUT_CHUNK_MS` | `200` | Max milliseconds of audio per `audio-chunk` event |
| `SCYLLASBAND_WARMUP` | `true` | Run a warmup synthesis at startup so the first request is fast |
| `SCYLLASBAND_LOG_LEVEL` | `INFO` | Log level |

The input **text**, **voice name**, and **language** always arrive per request via the Wyoming protocol from Home Assistant — they are intentionally *not* environment variables.

## Development

Useful mise tasks:

| Task | What it does |
| --- | --- |
| `mise run install` | `uv sync` — install dependencies |
| `mise run lock` | Regenerate `uv.lock` (commit the result) |
| `mise run test` | `uv run pytest -m "not e2e and not docker"` — unit + integration |
| `mise run test-unit` | Unit tests only (`tests/unit`) |
| `mise run test-integration` | Integration tests only (`tests/integration`) |
| `mise run test-e2e` | E2E tests (see below) |
| `mise run test-docker` | Docker image tests (see below) |
| `mise run lint` | `ruff check` + format check |
| `mise run format` | Format and autofix |
| `mise run typecheck` | `mypy src` |
| `mise run run` | Start the server |

Test markers:

- Default suite skips heavy tests: `pytest -m "not e2e and not docker"`.
- `SCYLLASBAND_E2E=1 mise run test-e2e` runs e2e tests against the **real model** — it downloads ~500 MB from Hugging Face (or reuses a bundle you point `SCYLLASBAND_MODEL_DIR` at; the layout is `<model-dir>/v2/onnx`).
- `SCYLLASBAND_DOCKER_E2E=1 mise run test-docker` builds and runs the Docker container end-to-end.

### CUDA on Windows

`onnxruntime-gpu` needs the CUDA/cuDNN DLLs at runtime. The Docker image gets them from its `nvidia/cuda` base image; on a Windows dev box install them from pip instead — the server registers the DLL directories automatically at startup:

```sh
uv sync --group cuda-windows
```

With the libraries missing, `SCYLLASBAND_DEVICE=auto` (the default) logs a warning and falls back to CPU; `SCYLLASBAND_DEVICE=cuda` fails fast with a clear error.

`uv.lock` must be committed; regenerate it with `mise run lock` when dependencies change.

## Troubleshooting

- **GPU not used** — check the startup logs with `SCYLLASBAND_DEVICE=auto`; verify NVIDIA drivers are installed and that CUDA 13 + cuDNN 9 are available for `onnxruntime-gpu` ≥ 1.30 (driver ≥ 580).
- **First start is slow** — the model bundle is downloaded (~316–498 MB) and a warmup synthesis runs at startup. Both happen only once per model directory.
- **Home Assistant shows no voices** — enable debug logging and confirm the server answers the Wyoming `describe` request with the expected voice list.

## License

- This project: [Apache-2.0](LICENSE).
- Model weights: Apache-2.0, at [huggingface.co/spybyscript/scyllasbandv2](https://huggingface.co/spybyscript/scyllasbandv2).
- The [Wyoming](https://github.com/rhasspy/wyoming) library is MIT/Apache dual-licensed.
