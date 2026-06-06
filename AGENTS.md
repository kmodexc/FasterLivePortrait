# Codex Workspace Notes

## Project

This repository is `FasterLivePortrait`. The current working focus is making
Gradio text-driven animation work in Docker with ONNX and TensorRT modes:

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
- Updated the then-current `Dockerfile.ada` and `Dockerfile.blackwell` for
  CUDA 12.8 runtime compatibility, custom ONNX Runtime wheel handling, cuDNN 8
  runtime install, eSpeak packages/data env vars, and build parallelism
  controls.
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

Current uncommitted local changes after that commit include:

- `.gitignore` includes `.hf-download-venv/`.
- ONNX Dockerfiles were renamed to:
  - `Dockerfile.ada.onnx`
  - `Dockerfile.blackwell.onnx`
- New TensorRT Dockerfiles now live at:
  - `Dockerfile.ada`
  - `Dockerfile.blackwell`
- TensorRT conversion/runtime compatibility patches were added in:
  - `scripts/onnx2trt.py`
  - `src/models/predictor.py`
  - `src/models/base_model.py`
  - `src/models/motion_extractor_model.py`
- TensorRT animal v1.1 paths were corrected in:
  - `scripts/all_onnx2trt_animal.sh`
  - `configs/trt_infer.yaml`
  - `configs/trt_mp_infer.yaml`

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

## Text-To-Animation Web UI

A dedicated fast-path Gradio entry point was added:

```bash
docker run -it --gpus=all --rm \
  -v "$PWD":/root/FasterLivePortrait \
  -w /root/FasterLivePortrait \
  -p 9871:9871 \
  flp python3 text_to_animation_webui.py --mode onnx --host_ip 0.0.0.0 --port 9871
```

Files involved:

- `text_to_animation_webui.py`
- `assets/examples/source/*.jpg`
- `src/pipelines/gradio_live_portrait_pipeline.py`
- `src/pipelines/faster_live_portrait_pipeline.py`

Behavior and speed-oriented changes:

- The site accepts a source image from upload, webcam/clipboard, or bundled
  examples, and accepts text/voice settings.
- The voice dropdown is populated from `checkpoints/Kokoro-82M/voices/*.pt`.
  It defaults to the preferred English voice list in `text_to_animation_webui.py`,
  currently starting with `af_heart`, then other English Kokoro voices.
- The source image and selected default Kokoro voice pipeline are
  prewarmed/cached at startup.
- `GradioLivePortraitPipeline` now caches Kokoro model, voices, and language
  pipelines via `get_kokoro_pipeline()`.
- The text-to-animation app shows detailed timing: Kokoro, JoyVASA, render,
  muxing, frame count, per-frame model time, and ONNX Runtime providers.
- Defaults now favor matching the main `webui.py` text-animation quality:
  `Full Image` output, `Frame Step` 1, `relative motion` off, `stitching` on,
  driving multiplier 1.0, `cfg_scale` 4.0, source crop on with scale 2.3/X
  0.0/Y -0.125, animation region `all`, and motion smooth strength `1e-7`.
- Crop-only output remains available as a speed option. When `Full Image` is
  off, the render path skips full original-frame upload/download, paste-back,
  full-video writing, and full-video audio muxing.
- A `Frame Step` slider was added. It renders every Nth motion frame and lowers
  output FPS so audio duration stays aligned. Values above 1 trade smoothness
  for lower latency.
- The text-to-animation app exposes the main still-image text controls from
  `webui.py`: source crop on/off, source crop scale/X/Y, relative motion,
  stitching, driving multiplier, CFG scale, animation region, and motion smooth
  strength.
- It intentionally does not expose unrelated or broader `webui.py` workflows:
  source video, driving video/image/audio/pickle tabs, driving-video crop
  controls, video-to-video head rotation, retargeting, reset buttons, or the
  animal model toggle.

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

ONNX performance conclusion:

- In ONNX mode the practical bottleneck is the LivePortrait `warping_spade`
  model, specifically the grid-sample-heavy path.
- `warping_spade.onnx` contains `GridSample` nodes.
- ONNX Runtime documentation now lists CUDA support for `GridSample` for some
  opset/type combinations, so the README claim that grid sample cannot be
  executed by ONNX Runtime is too broad/outdated. For this setup it appears to
  execute on the GPU but slowly.
- The README's TensorRT recommendation still looks correct for real-time
  performance. The repo claims 30+ FPS on an RTX 3090 with TensorRT.

## TensorRT Docker Status

Docker is still not available inside this Codex workspace, so image build,
engine conversion, and GPU verification must be done by the user on the host.

The user successfully built the Ada TensorRT image as `flp` from
`Dockerfile.ada`. The image is based on NVIDIA TensorRT 24.12 / TensorRT 10.7.

The first build failure was:

- `ImportError: libcuda.so.1` during `docker build` when importing
  `pycuda.driver`.

Fix applied:

- `Dockerfile.ada` and `Dockerfile.blackwell` now import `pycuda` instead of
  `pycuda.driver` during build. `pycuda.driver` can only be imported at
  container runtime with `--gpus=all`, because `libcuda.so.1` is provided by the
  host NVIDIA driver.

The TRT Dockerfiles:

- Build the `grid_sample_3d` TensorRT plugin inside the image.
- Copy it to `/opt/fasterliveportrait/plugins/libgrid_sample_3d_plugin.so`.
- Set `FLP_TRT_PLUGIN_PATH` to that copied plugin.
- Use `python3 -m venv --system-site-packages /opt/venv` so NVIDIA's system
  TensorRT Python bindings are visible inside the venv.
- Default to `webui.py --mode trt`.

`scripts/onnx2trt.py` and `src/models/predictor.py` now search for the plugin
in this order:

1. `$FLP_TRT_PLUGIN_PATH`
2. `./checkpoints/liveportrait_onnx/libgrid_sample_3d_plugin.so`
3. `/opt/fasterliveportrait/plugins/libgrid_sample_3d_plugin.so`

TensorRT 10 compatibility patches:

- `scripts/onnx2trt.py` supports `set_memory_pool_limit()` and
  `build_serialized_network()` while still keeping fallback paths for older TRT.
- `src/models/predictor.py` supports newer tensor APIs such as
  `num_io_tensors`, `get_tensor_name()`, and `get_tensor_mode()`.
- `TensorRTPredictor` now checks for missing `.trt` engines before loading the
  plugin/runtime and raises a clear `FileNotFoundError`.
- `BaseModel.__del__` and `TensorRTPredictor.__del__` tolerate partial
  initialization to avoid cleanup-time traceback noise.

TensorRT engines must be generated once before running `--mode trt`:

```bash
docker run -it --gpus=all --rm \
  -v "$PWD":/root/FasterLivePortrait \
  -w /root/FasterLivePortrait \
  flp sh scripts/all_onnx2trt.sh
```

This creates files such as:

```text
checkpoints/liveportrait_onnx/warping_spade-fix.trt
checkpoints/liveportrait_onnx/motion_extractor.trt
checkpoints/liveportrait_onnx/appearance_feature_extractor.trt
```

Then run the text-to-animation app with TensorRT:

```bash
docker run -it --gpus=all --rm \
  -v "$PWD":/root/FasterLivePortrait \
  -w /root/FasterLivePortrait \
  -p 9871:9871 \
  flp python3 text_to_animation_webui.py --mode trt --host_ip 0.0.0.0 --port 9871
```

The user ran TRT mode after building engines and hit a source-prep crash:

```text
ValueError: cannot reshape array of size 1 into shape (1,newaxis,3)
```

Root cause:

- `MotionExtractorModel.output_process()` assumed legacy TRT output order:
  `kp, pitch, yaw, roll, t, exp, scale`.
- The TensorRT 10 engine exposed ONNX output order:
  `pitch, yaw, roll, t, exp, scale, kp`.

Fix applied:

- `src/models/motion_extractor_model.py` now accepts both orders by checking
  the first output shape. If the first TRT output has shape like `(B, 66)`, it
  treats outputs as ONNX order; otherwise it keeps the legacy TRT order.

No engine rebuild should be needed for that fix because Python source is
bind-mounted into the container.

If trying TensorRT on Blackwell/RTX 50-series hardware, use
`Dockerfile.blackwell` and expect the plugin/engine build to be hardware and
driver sensitive. The `grid_sample_3d` TensorRT plugin must load successfully,
and `warping_spade-fix.trt` is the key engine for this bottleneck.

## Required Checkpoints

The README commands for audio/text driving are:

```bash
hf download TencentGameMate/chinese-hubert-base --local-dir checkpoints/chinese-hubert-base
hf download jdh-algo/JoyVASA --local-dir checkpoints/JoyVASA
```

Kokoro is expected at `checkpoints/Kokoro-82M` and LivePortrait ONNX models at
`checkpoints/liveportrait_onnx`.

Large checkpoint directories are ignored by git and should not be committed.

LivePortrait TRT engines are generated files under checkpoint directories and
should also remain uncommitted.

## Verification Commands

Use syntax checks with an alternate pycache location because repo pycache dirs
may be owned by another user:

```bash
PYTHONPYCACHEPREFIX=/tmp/flp-pycache python3 -m py_compile \
  text_to_animation_webui.py \
  src/pipelines/gradio_live_portrait_pipeline.py \
  src/pipelines/faster_live_portrait_pipeline.py \
  src/pipelines/joyvasa_audio_to_motion_pipeline.py \
  src/models/base_model.py \
  src/models/predictor.py \
  src/models/motion_extractor_model.py \
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
