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
  src/pipelines/gradio_live_portrait_pipeline.py \
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
