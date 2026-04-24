"""Run LingBot-Map streaming 3D reconstruction on a video and save outputs.

This script is a headless (no viser) wrapper around the core `lingbot_map` model. It:

1. Extracts frames from a video at a target FPS.
2. Preprocesses them with the canonical 518-px crop.
3. Runs streaming inference on GPU (auto keyframe interval for >320 frames).
4. Saves:
   - `frames/NNNNNN.npz` per-frame: depth, depth_conf, world_points, world_points_conf,
     extrinsic (c2w 3x4), intrinsic (3x3), image (H,W,3 uint8 in preprocessed resolution)
   - `meta.npz`: stacked arrays, intrinsics, c2w extrinsics
   - `rgb_depth.mp4`: RGB | colored depth side-by-side video
   - `point_cloud.ply`: world-frame colored point cloud (back-projected depth)
   - `bird_view.png`: static top-down bird's-eye image with trajectory overlay
   - `map.mp4`: animated top-down with current camera position marker
   - `input_and_map.mp4`: input video concatenated side-by-side with `map.mp4`
   - `run_config.json`: run metadata

Example:
    CUDA_VISIBLE_DEVICES=7 python run_video_inference.py \
        --video_path /path/to/video.mp4 \
        --model_path ./checkpoints/lingbot-map-long.pt \
        --output_dir ./outputs/video_name \
        --fps 16
"""

import argparse
import json
import os
import shutil
import tempfile
import time
from typing import Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from lingbot_map.utils.geometry import closed_form_inverse_se3_general
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

# Bird's-eye map helpers live in sibling scripts; imported lazily where needed
# so users without OpenCV-video deps can still run core inference.
from visualize_birdview import (
    collect_world_points,
    crop_xz_range,
    render_topdown,
    draw_trajectory,
    draw_axes_legend,
)
from make_map_video import (
    render_map_video,
    extract_target_frames,
    concat_input_and_map,
)
from video_writer import X265Writer


def extract_video_frames(video_path: str, out_dir: str, target_fps: float) -> list[str]:
    """Extract frames from a video at (approximately) the target FPS, save as JPEGs."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, round(src_fps / target_fps))
    paths: list[str] = []
    idx = 0
    pbar = tqdm(total=total, desc="Extracting frames", unit="frame")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            p = os.path.join(out_dir, f"{len(paths):06d}.jpg")
            cv2.imwrite(p, frame)
            paths.append(p)
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    print(
        f"Extracted {len(paths)} frames from {video_path} "
        f"(src={src_fps:.2f} fps, target={target_fps:.2f} fps, interval={interval})"
    )
    return paths


def build_model(model_path: str, device: torch.device, image_size: int, patch_size: int,
                num_scale_frames: int, kv_cache_sliding_window: int, camera_num_iterations: int,
                max_frame_num: int, use_sdpa: bool) -> torch.nn.Module:
    from lingbot_map.models.gct_stream import GCTStream

    print("Building model...")
    model = GCTStream(
        img_size=image_size,
        patch_size=patch_size,
        enable_3d_rope=True,
        max_frame_num=max_frame_num,
        kv_cache_sliding_window=kv_cache_sliding_window,
        kv_cache_scale_frames=num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=use_sdpa,
        camera_num_iterations=camera_num_iterations,
    )
    print(f"Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    return model.to(device).eval()


def postprocess_predictions(predictions: dict, image_hw: tuple[int, int]) -> dict:
    """Convert pose encoding to c2w extrinsics + intrinsics; strip batch dim; move to CPU."""
    extrinsic_w2c, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], image_hw)

    # Lift 3x4 to 4x4 then invert w2c -> c2w
    e4 = torch.zeros((*extrinsic_w2c.shape[:-2], 4, 4),
                     device=extrinsic_w2c.device, dtype=extrinsic_w2c.dtype)
    e4[..., :3, :4] = extrinsic_w2c
    e4[..., 3, 3] = 1.0
    e4 = closed_form_inverse_se3_general(e4)
    extrinsic_c2w = e4[..., :3, :4]

    predictions["extrinsic"] = extrinsic_c2w
    predictions["intrinsic"] = intrinsic
    predictions.pop("pose_enc_list", None)

    out: dict = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            v = v.to("cpu", non_blocking=True)
            if v.ndim > 1 and v.shape[0] == 1:
                v = v[0]
            out[k] = v
        else:
            out[k] = v
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out


def colorize_depth(depth: np.ndarray, vmin: float, vmax: float,
                   colormap=cv2.COLORMAP_TURBO) -> np.ndarray:
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    valid = d > 0
    d_norm = np.clip((d - vmin) / max(vmax - vmin, 1e-8) * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(d_norm, colormap)
    colored[~valid] = 0
    return colored


def save_rgb_depth_video(images_uint8: np.ndarray, depths: np.ndarray,
                          out_path: str, fps: float,
                          writer_kwargs: Optional[dict] = None) -> None:
    """Write side-by-side (RGB | colored-depth) video.

    images_uint8: (S, H, W, 3) RGB
    depths:       (S, H, W)
    writer_kwargs: forwarded to X265Writer (``codec``, ``crf``, ``preset``, ...).
    """
    s, h, w, _ = images_uint8.shape
    # libx265+yuv420p requires even dims; trim 1 px if needed.
    h_even = (h // 2) * 2
    w_even = (w // 2) * 2
    # Robust colormap range from valid depths across sequence (2/98 percentile)
    valid = depths[(depths > 0) & np.isfinite(depths)]
    if valid.size > 0:
        vmin = float(np.percentile(valid, 2))
        vmax = float(np.percentile(valid, 98))
    else:
        vmin, vmax = 0.0, 1.0
    print(f"Depth colormap range (p2-p98): [{vmin:.3f}, {vmax:.3f}]")

    with X265Writer(out_path, fps=fps, size=(w_even * 2, h_even),
                    **(writer_kwargs or {})) as writer:
        for i in tqdm(range(s), desc="Writing RGB+depth video", unit="frame"):
            rgb_bgr = cv2.cvtColor(images_uint8[i], cv2.COLOR_RGB2BGR)[:h_even, :w_even]
            depth_bgr = colorize_depth(depths[i], vmin, vmax)[:h_even, :w_even]
            writer.write(np.concatenate([rgb_bgr, depth_bgr], axis=1))
    print(f"Saved side-by-side video -> {out_path}")


def save_point_cloud(depth: np.ndarray, depth_conf: np.ndarray,
                     intrinsic: np.ndarray, extrinsic_c2w: np.ndarray,
                     images_uint8: np.ndarray, out_path: str,
                     conf_threshold: float, downsample_factor: int,
                     depth_max: float = 0.0) -> int:
    """Save a colored world-frame PLY by back-projecting per-frame depth.

    The model's raw ``world_points`` tensor is a normalized latent (not in
    world coordinates), so we back-project ``depth`` with the per-frame
    intrinsics and transform by the c2w extrinsic instead. See
    ``visualize_birdview.py::backproject_frame`` for the same logic.

    Args:
        depth:          (S, H, W) metric depth
        depth_conf:     (S, H, W) confidence map
        intrinsic:      (S, 3, 3) per-frame intrinsics
        extrinsic_c2w:  (S, 3, 4) per-frame camera-to-world
        images_uint8:   (S, H, W, 3) RGB
        conf_threshold: keep pixels with ``depth_conf >= threshold``
        downsample_factor: spatial stride (apply BEFORE back-projection)
        depth_max:      drop points farther than this; 0 disables
    """
    S, H, W = depth.shape
    stride = max(1, downsample_factor)

    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []

    ys, xs = np.mgrid[0:H:stride, 0:W:stride].astype(np.float32)
    for i in tqdm(range(S), desc="Back-projecting PLY", unit="frame"):
        d = depth[i, ::stride, ::stride]
        c = depth_conf[i, ::stride, ::stride]
        rgb = images_uint8[i, ::stride, ::stride]

        m = np.isfinite(d) & (d > 0) & (c >= conf_threshold)
        if depth_max > 0:
            m &= d <= depth_max
        if not m.any():
            continue

        u = xs[m]; v = ys[m]; z = d[m]
        Ki = intrinsic[i]
        x_cam = (u - Ki[0, 2]) / Ki[0, 0] * z
        y_cam = (v - Ki[1, 2]) / Ki[1, 1] * z
        pts_cam = np.stack([x_cam, y_cam, z], axis=-1)
        R = extrinsic_c2w[i, :, :3]
        t = extrinsic_c2w[i, :, 3]
        pts_world = pts_cam @ R.T + t
        all_pts.append(pts_world.astype(np.float32))
        all_cols.append(rgb[m].astype(np.uint8))

    if not all_pts:
        print("No valid points for PLY; skipping.")
        return 0
    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)

    _write_ply_ascii(out_path, pts, cols)
    print(f"Saved point cloud ({len(pts):,} pts, depth_conf>={conf_threshold}, "
          f"stride={stride}) -> {out_path}")
    return len(pts)


def _write_ply_ascii(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Minimal binary PLY writer (no extra deps)."""
    n = len(xyz)
    xyz = xyz.astype(np.float32)
    rgb = rgb.astype(np.uint8)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    buf = np.empty(n, dtype=dtype)
    buf["x"], buf["y"], buf["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    buf["r"], buf["g"], buf["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(buf.tobytes())


def save_per_frame_npz(output_dir: str, preds_np: dict, images_uint8: np.ndarray) -> None:
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    s = images_uint8.shape[0]
    for i in tqdm(range(s), desc="Saving per-frame NPZ", unit="frame"):
        np.savez_compressed(
            os.path.join(frames_dir, f"{i:06d}.npz"),
            image=images_uint8[i],
            depth=preds_np["depth"][i],
            depth_conf=preds_np["depth_conf"][i],
            world_points=preds_np["world_points"][i],
            world_points_conf=preds_np["world_points_conf"][i],
            extrinsic_c2w=preds_np["extrinsic"][i],
            intrinsic=preds_np["intrinsic"][i],
        )


def build_map_outputs(preds_np: dict, imgs_np: np.ndarray,
                      args: argparse.Namespace, out_dir: Path,
                      writer_kwargs: Optional[dict] = None) -> None:
    """Produce bird_view.png, map.mp4, and input_and_map.mp4 from in-memory preds.

    Re-uses the same back-projection helpers as ``visualize_birdview.py`` and
    ``make_map_video.py`` so the outputs here are byte-identical to running
    those scripts afterwards on the saved ``meta.npz``.
    """
    print("\n── Building bird's-eye map outputs ──")
    t_map = time.time()

    # Adapter dict with the keys expected by `collect_world_points`.
    depth_arr = (
        preds_np["depth"][..., 0] if preds_np["depth"].ndim == 4
        else preds_np["depth"]
    )
    meta_dict = {
        "depth": depth_arr,
        "depth_conf": preds_np["depth_conf"],
        "intrinsic": preds_np["intrinsic"],
        "extrinsic_c2w": preds_np["extrinsic"],
        "images": imgs_np,
    }
    cams_all = preds_np["extrinsic"][:, :, 3].astype(np.float32)
    cam_R_all = preds_np["extrinsic"][:, :, :3].astype(np.float32)

    # Static map: back-project + splat (uses same filters as the PLY).
    pts, cols, _ = collect_world_points(
        meta_dict,
        conf_threshold=args.conf_threshold,
        depth_max=args.point_cloud_depth_max,
        stride=args.map_pixel_stride,
        first_k=None,
        frame_stride=args.map_frame_stride,
    )
    if pts.shape[0] == 0:
        print("No valid points for bird's-eye map; skipping.")
        return

    # Compute bbox using all camera positions so the full traj fits on-map.
    bbox, pts, cols = crop_xz_range(
        pts, cols, cams_all, args.map_percentile_clip, args.map_pad_frac,
    )
    print(
        f"Static map: {pts.shape[0]:,} pts | "
        f"XZ=[{bbox[0]:.2f},{bbox[1]:.2f}]x[{bbox[2]:.2f},{bbox[3]:.2f}]"
    )

    img_rgb, bbox = render_topdown(
        pts, cols, bbox, args.map_resolution, up_axis=args.map_up_axis,
    )
    base_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # 5a) Static bird_view.png with full trajectory overlay
    if not args.skip_birdview:
        bv_bgr = draw_trajectory(img_rgb, cams_all, bbox)
        bv_bgr = draw_axes_legend(bv_bgr, bbox)
        bv_path = out_dir / "bird_view.png"
        cv2.imwrite(str(bv_path), bv_bgr)
        print(f"Saved bird's-eye image -> {bv_path}  [{bv_bgr.shape[1]}x{bv_bgr.shape[0]}]")

    # 5b) Animated map.mp4 + concat with input video
    if not args.skip_map_video:
        map_path = out_dir / "map.mp4"
        render_map_video(
            base_bgr, cams_all, cam_R_all, bbox,
            out_path=str(map_path), fps=args.fps,
            writer_kwargs=writer_kwargs,
        )
        try:
            input_frames = extract_target_frames(args.video_path, args.fps)
            if input_frames.shape[0] != cams_all.shape[0]:
                print(
                    f"Note: input re-sampled to {input_frames.shape[0]} frames, "
                    f"map has {cams_all.shape[0]}; taking min."
                )
            concat_input_and_map(
                input_frames, str(map_path),
                str(out_dir / "input_and_map.mp4"), fps=args.fps,
                writer_kwargs=writer_kwargs,
            )
        except Exception as e:
            print(f"Skipping input|map concat: {e}")

    print(f"Bird's-eye outputs done in {time.time() - t_map:.1f}s")


def main() -> None:
    p = argparse.ArgumentParser(description="LingBot-Map video inference (headless)")
    p.add_argument("--video_path", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--fps", type=float, default=16.0, help="Target frame extraction FPS")
    p.add_argument("--first_k", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)

    p.add_argument("--image_size", type=int, default=518)
    p.add_argument("--patch_size", type=int, default=14)
    p.add_argument("--num_scale_frames", type=int, default=8)
    p.add_argument("--keyframe_interval", type=int, default=None,
                   help="Streaming only. Auto if None: 1 for <=320 frames else ceil(N/320).")
    p.add_argument("--kv_cache_sliding_window", type=int, default=64)
    p.add_argument("--camera_num_iterations", type=int, default=4)
    p.add_argument("--max_frame_num", type=int, default=1024)
    p.add_argument("--use_sdpa", action="store_true",
                   help="Use PyTorch SDPA instead of FlashInfer")
    p.add_argument("--offload_to_cpu", action=argparse.BooleanOptionalAction, default=True,
                   help="Offload per-frame predictions to CPU during inference (default: on)")

    p.add_argument("--conf_threshold", type=float, default=3.0,
                   help="Point cloud visibility threshold on depth_conf")
    p.add_argument("--downsample_factor", type=int, default=4,
                   help="Spatial stride (pixels) for point cloud back-projection")
    p.add_argument("--point_cloud_depth_max", type=float, default=10.0,
                   help="Drop points with depth > this (meters) from PLY. 0 disables.")
    p.add_argument("--skip_per_frame_npz", action="store_true",
                   help="Skip writing per-frame NPZs (saves disk)")
    p.add_argument("--skip_point_cloud", action="store_true",
                   help="Skip point cloud PLY export")
    p.add_argument("--keep_raw_frames", action="store_true",
                   help="Keep extracted raw video frames in <output_dir>/raw_frames "
                        "(default: extract to a tempdir and delete on completion).")

    # ── Video encoding (libx265 by default) ──────────────────────────────────
    p.add_argument("--video_codec", type=str, default="libx265",
                   choices=["libx265", "libx264"],
                   help="ffmpeg codec for output videos. Default HEVC.")
    p.add_argument("--video_crf", type=int, default=28,
                   help="CRF quality (lower = larger/higher-quality). "
                        "Typical range: libx265 23-30, libx264 18-26.")
    p.add_argument("--video_preset", type=str, default="medium",
                   help="ffmpeg encoder preset (ultrafast..placebo). "
                        "Slower = better compression.")

    # ── Bird's-eye / map video ───────────────────────────────────────────────
    p.add_argument("--skip_birdview", action="store_true",
                   help="Skip static bird_view.png export")
    p.add_argument("--skip_map_video", action="store_true",
                   help="Skip animated map.mp4 and input_and_map.mp4")
    p.add_argument("--map_resolution", type=int, default=1400,
                   help="Longest-side px of the top-down map canvas")
    p.add_argument("--map_up_axis", type=str, default="y", choices=["x", "y", "z"],
                   help="World up axis for the top-down projection")
    p.add_argument("--map_pixel_stride", type=int, default=3,
                   help="Spatial subsample per frame when building the static map")
    p.add_argument("--map_frame_stride", type=int, default=1,
                   help="Temporal subsample of frames used for the static map")
    p.add_argument("--map_percentile_clip", type=float, default=1.0,
                   help="XZ bbox percentile trim for the map (each side)")
    p.add_argument("--map_pad_frac", type=float, default=0.05,
                   help="Extra XZ padding as a fraction of span")

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract + preprocess frames ──────────────────────────────────────────
    # Raw extracted frames are only needed during preprocessing. By default we
    # write them to a tempdir and delete after the run; pass --keep_raw_frames
    # to retain them under <output_dir>/raw_frames for inspection.
    t0 = time.time()
    if args.keep_raw_frames:
        frames_dir = str(out_dir / "raw_frames")
    else:
        frames_dir = tempfile.mkdtemp(prefix="lingbot_map_frames_")
        print(f"Extracting frames to tempdir: {frames_dir}")

    try:
        paths = extract_video_frames(args.video_path, frames_dir, args.fps)
        if args.first_k:
            paths = paths[: args.first_k]
        if args.stride > 1:
            paths = paths[:: args.stride]
        _run_inference_and_save(args, paths, out_dir, t0)
    finally:
        if not args.keep_raw_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)
            print(f"Cleaned up tempdir: {frames_dir}")


def _run_inference_and_save(args: argparse.Namespace,
                             paths: list[str],
                             out_dir: Path,
                             t0: float) -> None:
    """The rest of the pipeline; split out so main() can wrap it in try/finally."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Preprocessing {len(paths)} frames at {args.image_size}-px canonical crop...")
    images = load_and_preprocess_images(
        paths, mode="crop", image_size=args.image_size, patch_size=args.patch_size,
    )
    h, w = images.shape[-2:]
    print(f"Preprocessed to {w}x{h}, tensor {tuple(images.shape)}")

    # ── Build model ──────────────────────────────────────────────────────────
    model = build_model(
        args.model_path, device,
        image_size=args.image_size, patch_size=args.patch_size,
        num_scale_frames=args.num_scale_frames,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        camera_num_iterations=args.camera_num_iterations,
        max_frame_num=args.max_frame_num,
        use_sdpa=args.use_sdpa,
    )
    print(f"Load + preprocess done in {time.time() - t0:.1f}s")

    # Inference dtype
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    if dtype != torch.float32 and model.aggregator is not None:
        print(f"Casting aggregator to {dtype} (heads kept in fp32)")
        model.aggregator = model.aggregator.to(dtype=dtype)

    images = images.to(device)
    num_frames = int(images.shape[0])

    # Auto keyframe interval
    if args.keyframe_interval is None:
        args.keyframe_interval = (num_frames + 319) // 320 if num_frames > 320 else 1
    print(
        f"Frames: {num_frames} | keyframe_interval={args.keyframe_interval} | "
        f"offload_to_cpu={args.offload_to_cpu}"
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"GPU mem after load: alloc={torch.cuda.memory_allocated()/1e9:.2f} GB, "
            f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB"
        )

    # ── Inference ────────────────────────────────────────────────────────────
    output_device = torch.device("cpu") if args.offload_to_cpu else None
    t_inf = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        predictions = model.inference_streaming(
            images,
            num_scale_frames=args.num_scale_frames,
            keyframe_interval=args.keyframe_interval,
            output_device=output_device,
        )
    inf_time = time.time() - t_inf
    fps_inf = num_frames / max(inf_time, 1e-6)
    print(f"Inference done in {inf_time:.1f}s  ({fps_inf:.2f} fps)")
    if torch.cuda.is_available():
        print(
            f"GPU peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB "
            f"(reserved peak {torch.cuda.max_memory_reserved()/1e9:.2f} GB)"
        )

    # ── Post-process ─────────────────────────────────────────────────────────
    preds = postprocess_predictions(predictions, image_hw=(h, w))

    # Build uint8 RGB array (S, H, W, 3) from the preprocessed tensor on GPU side
    if args.offload_to_cpu and "images" in preds:
        imgs_tensor = preds.pop("images")  # (S, 3, H, W) on CPU
    else:
        imgs_tensor = images.detach().cpu()
    imgs_np = (imgs_tensor.float().clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)

    preds_np = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in preds.items()}

    # Free GPU before heavy I/O
    del images, predictions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Save outputs ─────────────────────────────────────────────────────────
    # 1) Meta npz with stacked arrays
    meta_path = out_dir / "meta.npz"
    print(f"Saving meta.npz to {meta_path}")
    np.savez_compressed(
        meta_path,
        depth=preds_np["depth"].astype(np.float32),
        depth_conf=preds_np["depth_conf"].astype(np.float32),
        world_points=preds_np["world_points"].astype(np.float32),
        world_points_conf=preds_np["world_points_conf"].astype(np.float32),
        extrinsic_c2w=preds_np["extrinsic"].astype(np.float32),
        intrinsic=preds_np["intrinsic"].astype(np.float32),
        images=imgs_np,
    )
    print(f"Saved stacked arrays -> {meta_path}")

    # 2) Per-frame NPZ (optional)
    if not args.skip_per_frame_npz:
        print(f"Saving per-frame NPZ to {out_dir}")
        save_per_frame_npz(str(out_dir), preds_np, imgs_np)

    # 3) RGB | depth video
    video_path = out_dir / "rgb_depth.mp4"
    writer_kwargs = {
        "codec": args.video_codec,
        "crf": args.video_crf,
        "preset": args.video_preset,
    }
    save_rgb_depth_video(
        imgs_np, preds_np["depth"].squeeze(-1) if preds_np["depth"].ndim == 4 else preds_np["depth"],
        str(video_path), fps=args.fps, writer_kwargs=writer_kwargs,
    )

    # 4) Point cloud
    if not args.skip_point_cloud:
        print(f"Saving point cloud to {out_dir / 'point_cloud.ply'}")
        save_point_cloud(
            depth=preds_np["depth"][..., 0] if preds_np["depth"].ndim == 4 else preds_np["depth"],
            depth_conf=preds_np["depth_conf"],
            intrinsic=preds_np["intrinsic"],
            extrinsic_c2w=preds_np["extrinsic"],
            images_uint8=imgs_np,
            out_path=str(out_dir / "point_cloud.ply"),
            conf_threshold=args.conf_threshold,
            downsample_factor=args.downsample_factor,
            depth_max=args.point_cloud_depth_max,
        )

    # 5) Bird's-eye map (static PNG + animated MP4 + input|map concat)
    if not (args.skip_birdview and args.skip_map_video):
        build_map_outputs(
            preds_np=preds_np,
            imgs_np=imgs_np,
            args=args,
            out_dir=out_dir,
            writer_kwargs=writer_kwargs,
        )

    # 6) Run config
    cfg = {
        "video_path": os.path.abspath(args.video_path),
        "model_path": os.path.abspath(args.model_path),
        "output_dir": os.path.abspath(str(out_dir)),
        "num_frames": num_frames,
        "preprocessed_hw": [int(h), int(w)],
        "fps_extract": args.fps,
        "keyframe_interval": args.keyframe_interval,
        "num_scale_frames": args.num_scale_frames,
        "kv_cache_sliding_window": args.kv_cache_sliding_window,
        "camera_num_iterations": args.camera_num_iterations,
        "use_sdpa": args.use_sdpa,
        "offload_to_cpu": args.offload_to_cpu,
        "inference_time_s": inf_time,
        "inference_fps": fps_inf,
        "dtype": str(dtype),
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved run config -> {out_dir / 'run_config.json'}")
    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
