# Codex Workspace Notes

## Project

This repository is `FasterLivePortrait`. The current working focus is making
Gradio text-driven animation work in Docker with ONNX mode:

```bash
docker run -it --gpus=all --rm \
  -v "$PWD":/root/FasterLivePortrait \
  -w /root/FasterLivePortrait \
  -p 9870:9870 \
  flp python3 webui.py --mode onnx --host_ip 0.0.0.0 --port 9870
```

The container bind-mounts this repo, so Python source edits apply without
rebuilding the image. Docker is not available in this Codex workspace, so full
GPU/container verification must be done by the user on the host.

## Last Commit

Latest commit inspected: `50d7e4d fixed build and voice transcription`
(`50d7e4dc7f4d3cd2aa4348bb3336400e3f0d4b78`).

Main changes in that commit:

- Added `.dockerignore` to keep checkpoints, results, caches, and build output
  out of Docker build context.
- Updated `Dockerfile.ada` and `Dockerfile.blackwell` for CUDA 12.8 runtime
  compatibility, custom ONNX Runtime wheel handling, cuDNN 8 runtime install,
  eSpeak packages/data env vars, and build parallelism controls.
- Added optional `FLP_PROFILE=1` frame timing in
  `src/pipelines/faster_live_portrait_pipeline.py` and
  `src/pipelines/gradio_live_portrait_pipeline.py`.
- Added `mux_audio()` helper and replaced duplicated ffmpeg calls in
  `src/pipelines/gradio_live_portrait_pipeline.py`.
- Added Kokoro/eSpeak compatibility patching in
  `src/pipelines/gradio_live_portrait_pipeline.py`.
- Updated JoyVASA audio path compatibility in
  `src/pipelines/joyvasa_audio_to_motion_pipeline.py`.
- Set JoyVASA Transformers audio encoders to `attn_implementation="eager"` in
  `src/models/JoyVASA/dit_talking_head.py`.

Current uncommitted local change seen after that commit: `.gitignore` includes
`.hf-download-venv/`.

## Text-To-Animation Fixes Applied

The text path is:

1. Kokoro text to WAV in `run_text_driving()`.
2. JoyVASA audio to motion pickle in `run_audio_driving()`.
3. LivePortrait animation from generated motion.

Issues fixed while debugging:

- eSpeak `phontab` missing:
  `patch_espeak_wrapper()` now installs a compatible `set_data_path()` shim and
  initializes phonemizer/eSpeak with a data path containing `phontab`.
- Missing JoyVASA checkpoints:
  Downloaded required models to:
  - `checkpoints/JoyVASA/motion_generator/motion_generator_hubert_chinese.pt`
  - `checkpoints/JoyVASA/motion_template/motion_template.pkl`
  - `checkpoints/chinese-hubert-base/pytorch_model.bin`
  - `checkpoints/chinese-hubert-base/config.json`
- PyTorch 2.6+ `torch.load()` default:
  JoyVASA checkpoint load uses `weights_only=False` because the checkpoint
  includes `argparse.Namespace` metadata.
- New torchaudio TorchCodec requirement:
  JoyVASA audio reading uses `soundfile` via `load_audio()` instead of
  `torchaudio.load()`.
- Transformers SDPA attention incompatibility:
  JoyVASA HuBERT/Wav2Vec2 encoders load with `attn_implementation="eager"`
  because their wrappers request `output_attentions=True`.

## Mona Lisa Text-To-Video Web UI

A dedicated fast-path Gradio entry point was added:

```bash
docker run -it --gpus=all --rm \
  -v "$PWD":/root/FasterLivePortrait \
  -w /root/FasterLivePortrait \
  -p 9871:9871 \
  flp python3 mona_lisa_webui.py --mode onnx --host_ip 0.0.0.0 --port 9871
```

Files involved:

- `mona_lisa_webui.py`
- `assets/examples/source/mona_lisa.jpg`
- `src/pipelines/gradio_live_portrait_pipeline.py`
- `src/pipelines/faster_live_portrait_pipeline.py`

Behavior and speed-oriented changes:

- The site uses a fixed Mona Lisa source image and only accepts text/voice
  settings.
- The source image and Kokoro voice pipeline are prewarmed/cached at startup.
- `GradioLivePortraitPipeline` now caches Kokoro model, voices, and language
  pipelines via `get_kokoro_pipeline()`.
- The Mona app shows detailed timing: Kokoro, JoyVASA, render, muxing, frame
  count, per-frame model time, and ONNX Runtime providers.
- Crop-only output is the default. When `Full Painting` is off, the render path
  skips full original-frame upload/download, paste-back, full-video writing, and
  full-video audio muxing.
- A `Frame Step` slider was added. It renders every Nth motion frame and lowers
  output FPS so audio duration stays aligned. This trades smoothness for lower
  latency.

Measured user run in ONNX mode on an RTX 4070 Laptop GPU at 100 W:

- Total button-to-ready time: about 25 seconds for 112 frames.
- Render loop: about 23.6 seconds.
- Effective render rate: about 4.7 FPS.
- `warping_spade` dominates: about 0.208 seconds per frame.
- Video encode/write, Kokoro, JoyVASA, and muxing are not significant
  bottlenecks.
- ONNX Runtime reported `CUDAExecutionProvider, CPUExecutionProvider` for
  `warping_spade`; GPU utilization was 99% and power was 100/100 W, so this was
  not an idle GPU or obvious CPU fallback case.

Current conclusion:

- In ONNX mode the practical bottleneck is the LivePortrait `warping_spade`
  model, specifically the grid-sample-heavy path.
- `warping_spade.onnx` contains `GridSample` nodes.
- ONNX Runtime documentation now lists CUDA support for `GridSample` for some
  opset/type combinations, so the README claim that grid sample cannot be
  executed by ONNX Runtime is too broad/outdated. For this setup it appears to
  execute on the GPU but slowly.
- The README's TensorRT recommendation still looks correct for real-time
  performance. The repo claims 30+ FPS on an RTX 3090 with TensorRT.
- Local checkpoint tree currently contains ONNX files but no `.trt` engines.

Recommended next performance step:

```bash
docker run -it --gpus=all --rm \
  -v "$PWD":/root/FasterLivePortrait \
  -w /root/FasterLivePortrait \
  flp sh scripts/all_onnx2trt.sh
```

Then run the Mona app with TensorRT:

```bash
docker run -it --gpus=all --rm \
  -v "$PWD":/root/FasterLivePortrait \
  -w /root/FasterLivePortrait \
  -p 9871:9871 \
  flp python3 mona_lisa_webui.py --mode trt --host_ip 0.0.0.0 --port 9871
```

If trying TensorRT on Blackwell/RTX 50-series hardware, prefer the existing
`Dockerfile.blackwell` path if the current image/tooling cannot build/load TRT
engines. The `grid_sample_3d` TensorRT plugin must load successfully, and
`warping_spade-fix.trt` is the key engine for this bottleneck.

## Required Checkpoints

The README commands for audio/text driving are:

```bash
hf download TencentGameMate/chinese-hubert-base --local-dir checkpoints/chinese-hubert-base
hf download jdh-algo/JoyVASA --local-dir checkpoints/JoyVASA
```

Kokoro is expected at `checkpoints/Kokoro-82M` and LivePortrait ONNX models at
`checkpoints/liveportrait_onnx`.

Large checkpoint directories are ignored by git and should not be committed.

## Verification Commands

Use syntax checks with an alternate pycache location because repo pycache dirs
may be owned by another user:

```bash
PYTHONPYCACHEPREFIX=/tmp/flp-pycache python3 -m py_compile \
  mona_lisa_webui.py \
  src/pipelines/gradio_live_portrait_pipeline.py \
  src/pipelines/faster_live_portrait_pipeline.py \
  src/pipelines/joyvasa_audio_to_motion_pipeline.py \
  src/models/JoyVASA/dit_talking_head.py \
  src/models/JoyVASA/hubert.py \
  src/models/JoyVASA/wav2vec2.py
```

Useful checkpoint sanity check:

```bash
test -f checkpoints/JoyVASA/motion_generator/motion_generator_hubert_chinese.pt
test -f checkpoints/JoyVASA/motion_template/motion_template.pkl
test -f checkpoints/chinese-hubert-base/pytorch_model.bin
test -f checkpoints/chinese-hubert-base/config.json
```

## Notes For Future Work

- Do not revert unrelated user changes in this repo.
- Prefer small compatibility patches over broad dependency pinning unless the
  Docker image is being rebuilt intentionally.
- If text-to-animation fails again, read the newest traceback bottom-up. Recent
  failures progressed one stage at a time after each fix, so a new traceback is
  likely the next compatibility issue rather than a regression in earlier fixes.
- The Gradio warnings about Gradio 6 parameter placement and `Tabs()` children
  are noisy but were not blocking text-to-animation.
