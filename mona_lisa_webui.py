# coding: utf-8
import argparse
import os
import time

import gradio as gr
from omegaconf import OmegaConf

from src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline


MONA_LISA_PATH = "assets/examples/source/mona_lisa.jpg"


def build_parser():
    parser = argparse.ArgumentParser(description="Mona Lisa text-to-video web UI")
    parser.add_argument("--mode", required=False, type=str, default="onnx", choices=["onnx", "trt"])
    parser.add_argument("--use_mp", action="store_true", help="use mediapipe face detection")
    parser.add_argument("--host_ip", type=str, default="127.0.0.1", help="host ip")
    parser.add_argument("--port", type=int, default=9871, help="server port")
    parser.add_argument("--source_image", type=str, default=MONA_LISA_PATH, help="fixed source image")
    parser.add_argument("--no_prewarm", action="store_true", help="skip source/voice warmup at launch")
    return parser


args, unknown = build_parser().parse_known_args()

cfg_path = f"configs/{args.mode}_{'mp_' if args.use_mp else ''}infer.yaml"
infer_cfg = OmegaConf.load(cfg_path)
pipeline = GradioLivePortraitPipeline(infer_cfg)


def list_voices():
    voice_dir = "checkpoints/Kokoro-82M/voices"
    if not os.path.isdir(voice_dir):
        return ["af_heart"]
    voices = [
        os.path.splitext(vname)[0]
        for vname in os.listdir(voice_dir)
        if vname.endswith(".pt")
    ]
    return sorted(voices) or ["af_heart"]


def configure_fast_path(paste_back=False, cfg_scale=1.2, driving_multiplier=1.0):
    return pipeline.update_cfg({
        "source": args.source_image,
        "driving": "",
        "flag_relative_motion": True,
        "flag_do_crop": True,
        "flag_pasteback": paste_back,
        "driving_multiplier": driving_multiplier,
        "flag_stitching": True,
        "flag_crop_driving_video": False,
        "flag_video_editing_head_rotation": False,
        "src_scale": 2.3,
        "src_vx_ratio": 0.0,
        "src_vy_ratio": -0.125,
        "driving_smooth_observation_variance": 1e-7,
        "animation_region": "all",
        "cfg_scale": cfg_scale,
    })


def prewarm():
    if args.no_prewarm:
        return
    if not os.path.exists(args.source_image):
        print(f"skip prewarm, source image is missing: {args.source_image}")
        return
    configure_fast_path(paste_back=False)
    pipeline.init_vars()
    if not pipeline.prepare_source(args.source_image):
        print(f"warning: failed to prepare source image: {args.source_image}")
    voices = list_voices()
    if voices:
        try:
            pipeline.get_kokoro_pipeline(voices[0])
        except Exception as exc:
            print(f"warning: failed to prewarm Kokoro: {exc}")


def animate(text, voice_name, paste_back, cfg_scale, driving_multiplier, frame_step):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Enter text first.")
    if not os.path.exists(args.source_image):
        raise gr.Error(f"Missing source image: {args.source_image}")

    started = time.perf_counter()
    update_ret = configure_fast_path(
        paste_back=paste_back,
        cfg_scale=cfg_scale,
        driving_multiplier=driving_multiplier,
    )
    video_path, crop_path, _pipeline_time, timings = pipeline.run_text_driving_profiled(
        text,
        voice_name,
        args.source_image,
        update_ret=update_ret,
        mux_full_audio=paste_back,
        render_full_video=paste_back,
        render_stride=frame_step,
    )
    elapsed = time.perf_counter() - started
    ready_path = video_path if paste_back else crop_path
    timings["button_to_ready"] = elapsed
    timing_text = format_timings(timings)
    runtime_text = "\n".join(pipeline.get_runtime_summary())
    if runtime_text:
        timing_text = f"{timing_text}\n\n{runtime_text}"
    return ready_path, f"{elapsed:.2f} s", timing_text


def format_timings(timings):
    labels = [
        ("button_to_ready", "Button to ready"),
        ("text_to_video_total", "Pipeline total"),
        ("source_prepare", "Source prepare"),
        ("kokoro_synthesize", "Kokoro text to audio"),
        ("joyvasa_init", "JoyVASA load"),
        ("joyvasa_generate", "JoyVASA audio to motion"),
        ("motion_pickle_write", "Motion pickle write"),
        ("render_video", "Render frames"),
        ("render_frames", "Frames rendered"),
        ("render_input_frames", "Input motion frames"),
        ("render_stride", "Frame step"),
        ("render_output_fps", "Output video fps"),
        ("render_actual_fps", "Actual render fps"),
        ("render_seconds_per_frame", "Render seconds per frame"),
        ("render_model_total", "Render model total"),
        ("render_model_seconds_per_frame", "Render model per frame"),
        ("render_crop_write_total", "Crop encode/write total"),
        ("render_crop_write_seconds_per_frame", "Crop encode/write per frame"),
        ("render_full_write_total", "Full encode/write total"),
        ("render_full_write_seconds_per_frame", "Full encode/write per frame"),
        ("render_loop_overhead_total", "Render loop overhead"),
        ("video_info", "Read video info"),
        ("mux_crop_audio", "Mux crop audio"),
        ("mux_full_audio", "Mux full audio"),
    ]
    count_keys = {"render_frames", "render_input_frames", "render_stride"}
    rate_keys = {"render_actual_fps"}
    lines = []
    for key, label in labels:
        if key in timings:
            if key in count_keys:
                lines.append(f"{label}: {timings[key]:.0f}")
            elif key in rate_keys:
                lines.append(f"{label}: {timings[key]:.2f}")
            else:
                lines.append(f"{label}: {timings[key]:.2f} s")
    remaining = sorted(
        (key, value)
        for key, value in timings.items()
        if key not in {item[0] for item in labels}
    )
    for key, value in remaining:
        if key.endswith("_per_frame"):
            lines.append(f"{key}: {value:.4f} s")
        else:
            lines.append(f"{key}: {value:.2f} s")
    return "\n".join(lines)


css = """
.mona-shell {max-width: 1180px; margin: 0 auto;}
.metric input {font-size: 26px !important; font-weight: 700 !important;}
.compact-preview img {object-fit: cover;}
"""

prewarm()
voices = list_voices()
default_voice = "af_heart" if "af_heart" in voices else voices[0]

with gr.Blocks(theme=gr.themes.Soft(font=[gr.themes.GoogleFont("Plus Jakarta Sans")]), css=css) as demo:
    with gr.Column(elem_classes=["mona-shell"]):
        gr.Markdown("# Mona Lisa Text To Video")
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                gr.Image(
                    value=args.source_image if os.path.exists(args.source_image) else None,
                    label="Mona Lisa",
                    interactive=False,
                    height=460,
                    elem_classes=["compact-preview"],
                )
                elapsed_output = gr.Textbox(
                    value="",
                    label="Ready In",
                    interactive=False,
                    elem_classes=["metric"],
                )
                timings_output = gr.Textbox(
                    value="",
                    label="Timing Breakdown",
                    lines=20,
                    interactive=False,
                )
            with gr.Column(scale=2, min_width=420):
                text_input = gr.Textbox(
                    value="Hello. I am Mona Lisa, animated by Faster LivePortrait.",
                    label="Text",
                    lines=5,
                )
                with gr.Row():
                    voice_input = gr.Dropdown(
                        choices=voices,
                        value=default_voice,
                        label="Voice",
                    )
                    cfg_scale_input = gr.Slider(
                        minimum=0.0,
                        maximum=6.0,
                        value=1.2,
                        step=0.1,
                        label="CFG Scale",
                    )
                    driving_multiplier_input = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.05,
                        label="Motion",
                    )
                    frame_step_input = gr.Slider(
                        minimum=1,
                        maximum=4,
                        value=2,
                        step=1,
                        label="Frame Step",
                    )
                paste_back_input = gr.Checkbox(value=False, label="Full Painting")
                run_button = gr.Button("Generate", variant="primary")
                video_output = gr.Video(label="Video", autoplay=True)

    run_button.click(
        fn=animate,
        inputs=[
            text_input,
            voice_input,
            paste_back_input,
            cfg_scale_input,
            driving_multiplier_input,
            frame_step_input,
        ],
        outputs=[video_output, elapsed_output, timings_output],
        show_progress=True,
    )

if __name__ == "__main__":
    demo.launch(
        server_port=args.port,
        share=False,
        server_name=args.host_ip,
    )
