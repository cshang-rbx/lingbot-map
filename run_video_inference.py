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
   - `point_cloud.ply`: downsampled colored point cloud (uses world_points_conf to filter)
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
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from lingbot_map.utils.geometry import closed_form_inverse_se3_general
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri


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
                          out_path: str, fps: float) -> None:
    """Write side-by-side (RGB | colored-depth) video.

    images_uint8: (S, H, W, 3) RGB
    depths:       (S, H, W)
    """
    s, h, w, _ = images_uint8.shape
    # Robust colormap range from valid depths across sequence (2/98 percentile)
    valid = depths[(depths > 0) & np.isfinite(depths)]
    if valid.size > 0:
        vmin = float(np.percentile(valid, 2))
        vmax = float(np.percentile(valid, 98))
    else:
        vmin, vmax = 0.0, 1.0
    print(f"Depth colormap range (p2-p98): [{vmin:.3f}, {vmax:.3f}]")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w * 2, h))
    for i in tqdm(range(s), desc="Writing RGB+depth video", unit="frame"):
        rgb_bgr = cv2.cvtColor(images_uint8[i], cv2.COLOR_RGB2BGR)
        depth_bgr = colorize_depth(depths[i], vmin, vmax)
        writer.write(np.concatenate([rgb_bgr, depth_bgr], axis=1))
    writer.release()
    print(f"Saved side-by-side video -> {out_path}")


def save_point_cloud(world_points: np.ndarray, world_points_conf: np.ndarray,
                     images_uint8: np.ndarray, out_path: str,
                     conf_threshold: float, downsample_factor: int) -> int:
    """Save a downsampled, confidence-filtered colored PLY point cloud.

    world_points:      (S, H, W, 3)
    world_points_conf: (S, H, W)
    images_uint8:      (S, H, W, 3) RGB
    """
    pts = world_points.reshape(-1, 3)
    conf = world_points_conf.reshape(-1)
    cols = images_uint8.reshape(-1, 3)

    mask = np.isfinite(pts).all(axis=-1) & (conf >= conf_threshold)
    pts = pts[mask]
    cols = cols[mask]

    if downsample_factor > 1:
        pts = pts[::downsample_factor]
        cols = cols[::downsample_factor]

    _write_ply_ascii(out_path, pts, cols)
    print(f"Saved point cloud ({len(pts):,} pts, conf>={conf_threshold}) -> {out_path}")
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

    p.add_argument("--conf_threshold", type=float, default=1.5,
                   help="Point cloud visibility threshold on world_points_conf")
    p.add_argument("--downsample_factor", type=int, default=10,
                   help="Spatial downsampling for point cloud export")
    p.add_argument("--skip_per_frame_npz", action="store_true",
                   help="Skip writing per-frame NPZs (saves disk)")
    p.add_argument("--skip_point_cloud", action="store_true",
                   help="Skip point cloud PLY export")

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Extract + preprocess frames ──────────────────────────────────────────
    t0 = time.time()
    frames_dir = str(out_dir / "raw_frames")
    paths = extract_video_frames(args.video_path, frames_dir, args.fps)
    if args.first_k:
        paths = paths[: args.first_k]
    if args.stride > 1:
        paths = paths[:: args.stride]

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
        save_per_frame_npz(str(out_dir), preds_np, imgs_np)

    # 3) RGB | depth video
    video_path = out_dir / "rgb_depth.mp4"
    save_rgb_depth_video(
        imgs_np, preds_np["depth"].squeeze(-1) if preds_np["depth"].ndim == 4 else preds_np["depth"],
        str(video_path), fps=args.fps,
    )

    # 4) Point cloud
    if not args.skip_point_cloud:
        save_point_cloud(
            preds_np["world_points"],
            preds_np["world_points_conf"],
            imgs_np,
            str(out_dir / "point_cloud.ply"),
            conf_threshold=args.conf_threshold,
            downsample_factor=args.downsample_factor,
        )

    # 5) Run config
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
