"""Produce an animated bird's-eye map video and concat it with the input video.

Pipeline:
  1. Load ``meta.npz`` (produced by ``run_video_inference.py``).
  2. Back-project all frames into world points and splat them into a static
     top-down image (the "map"). Reuses helpers from ``visualize_birdview``.
  3. For each video frame, draw on the static map:
        * the trajectory polyline up to the current frame (cyan)
        * a filled marker at the current camera XZ position (red)
        * a short forward-direction tick (derived from the c2w rotation)
     and write each composited frame into ``map.mp4``.
  4. Re-encode frames from the input video (time-aligned with the model's
     frame sampling) and concatenate them side-by-side with ``map.mp4`` into
     ``input_and_map.mp4``.

Example:
    python make_map_video.py \
        --meta_path outputs/syn128_overturned_trash/meta.npz \
        --video_path /path/to/input.mp4 \
        --output_dir outputs/syn128_overturned_trash \
        --map_fps 16
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from visualize_birdview import (
    collect_world_points,
    crop_xz_range,
    render_topdown,
)


# =============================================================================
# Drawing helpers
# =============================================================================

def world_xz_to_px(xz: np.ndarray, bbox: tuple[float, float, float, float],
                   HW: tuple[int, int]) -> np.ndarray:
    """Project world (X, Z) -> image (u, v) int coords.

    Row increases downward, so larger Z maps to smaller v (Z increases "up" in
    the image).
    """
    x_min, x_max, z_min, z_max = bbox
    H, W = HW
    sx = (W - 1) / max(x_max - x_min, 1e-6)
    sz = (H - 1) / max(z_max - z_min, 1e-6)
    u = np.clip(((xz[..., 0] - x_min) * sx).astype(np.int32), 0, W - 1)
    v = np.clip(((z_max - xz[..., 1]) * sz).astype(np.int32), 0, H - 1)
    return np.stack([u, v], axis=-1)


def draw_map_legend(bgr: np.ndarray, bbox: tuple[float, float, float, float],
                    frame_idx: int, total_frames: int) -> np.ndarray:
    x_min, x_max, z_min, z_max = bbox
    H, W = bgr.shape[:2]
    # Semi-transparent legend box
    lines = [
        f"Top-down map (XZ, look down -Y)",
        f"X: [{x_min:.2f}, {x_max:.2f}] m",
        f"Z: [{z_min:.2f}, {z_max:.2f}] m",
        f"Frame: {frame_idx + 1:4d} / {total_frames}",
        "cyan=trajectory  red=current",
    ]
    box_w = 330
    box_h = 22 + 18 * len(lines)
    overlay = bgr.copy()
    cv2.rectangle(overlay, (6, 6), (6 + box_w, 6 + box_h), (0, 0, 0), -1)
    bgr = cv2.addWeighted(overlay, 0.55, bgr, 0.45, 0)
    y = 26
    for line in lines:
        cv2.putText(bgr, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 18
    return bgr


def draw_north_arrow(bgr: np.ndarray) -> np.ndarray:
    """Small compass showing +Z direction (up in image) in the corner."""
    H, W = bgr.shape[:2]
    cx, cy = W - 60, H - 60
    overlay = bgr.copy()
    cv2.circle(overlay, (cx, cy), 32, (0, 0, 0), -1)
    bgr = cv2.addWeighted(overlay, 0.45, bgr, 0.55, 0)
    cv2.circle(bgr, (cx, cy), 32, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.arrowedLine(bgr, (cx, cy + 20), (cx, cy - 20), (200, 200, 255), 2,
                    cv2.LINE_AA, tipLength=0.3)
    cv2.putText(bgr, "+Z", (cx - 12, cy - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (200, 200, 255), 1, cv2.LINE_AA)
    cv2.arrowedLine(bgr, (cx - 20, cy), (cx + 20, cy), (200, 255, 200), 2,
                    cv2.LINE_AA, tipLength=0.3)
    cv2.putText(bgr, "+X", (cx + 22, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (200, 255, 200), 1, cv2.LINE_AA)
    return bgr


# =============================================================================
# Map video rendering
# =============================================================================

def render_map_video(
    base_bgr: np.ndarray,
    cams_world: np.ndarray,          # (S, 3) all camera positions in world
    cam_R: np.ndarray,               # (S, 3, 3) camera-to-world rotations
    bbox: tuple[float, float, float, float],
    out_path: str,
    fps: float,
    traj_color: tuple[int, int, int] = (255, 255, 0),    # BGR: cyan-ish
    current_color: tuple[int, int, int] = (0, 0, 255),   # BGR: red
    heading_color: tuple[int, int, int] = (0, 255, 255), # BGR: yellow
    heading_len_m: float = 0.6,
    traj_thickness: int = 2,
    current_radius: int = 7,
) -> None:
    H, W = base_bgr.shape[:2]
    S = cams_world.shape[0]
    # Pre-compute pixel positions for all trajectory points.
    cam_xz = cams_world[:, [0, 2]]
    traj_px = world_xz_to_px(cam_xz, bbox, (H, W))

    # Camera forward in world frame: R_c2w @ [0, 0, 1]  (OpenCV-style +Z forward)
    forward_world = cam_R[:, :, 2]  # (S, 3)

    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
    )
    for i in tqdm(range(S), desc="Rendering map video", unit="frame"):
        frame = base_bgr.copy()

        # Trajectory up to (and including) i
        if i > 0:
            pts = traj_px[: i + 1]
            cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False,
                          traj_color, traj_thickness, cv2.LINE_AA)

        # Heading tick: project +forward_len_m in XZ to pixel space and draw
        fwd_xz = cam_xz[i] + heading_len_m * forward_world[i, [0, 2]]
        p_cur = tuple(traj_px[i].tolist())
        p_fwd = tuple(world_xz_to_px(fwd_xz[None], bbox, (H, W))[0].tolist())
        cv2.arrowedLine(frame, p_cur, p_fwd, heading_color, 2, cv2.LINE_AA,
                        tipLength=0.4)

        # Current position
        cv2.circle(frame, p_cur, current_radius + 3, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, p_cur, current_radius, current_color, -1, cv2.LINE_AA)

        frame = draw_map_legend(frame, bbox, i, S)
        frame = draw_north_arrow(frame)
        writer.write(frame)

    writer.release()
    print(f"Saved map video -> {out_path}")


# =============================================================================
# Input video re-sampling
# =============================================================================

def extract_target_frames(video_path: str, target_fps: float) -> np.ndarray:
    """Re-sample a video at target_fps, return (S, H, W, 3) uint8 BGR frames.

    Sampling matches ``run_video_inference.extract_video_frames``:
    ``interval = max(1, round(src_fps / target_fps))``.
    """
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, round(src_fps / target_fps))
    frames: list[np.ndarray] = []
    idx = 0
    pbar = tqdm(total=total, desc="Reading input video", unit="frame")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % interval == 0:
            frames.append(frame)
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    print(f"Read {len(frames)} frames (src={src_fps:.2f} fps, "
          f"target={target_fps:.2f} fps, interval={interval})")
    return np.stack(frames, axis=0)


# =============================================================================
# Side-by-side concat
# =============================================================================

def concat_input_and_map(
    input_frames_bgr: np.ndarray,
    map_video_path: str,
    out_path: str,
    fps: float,
) -> None:
    """Concat each input frame with its corresponding map frame side-by-side."""
    cap = cv2.VideoCapture(map_video_path)
    map_frames: list[np.ndarray] = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        map_frames.append(f)
    cap.release()

    n = min(len(input_frames_bgr), len(map_frames))
    if n == 0:
        raise RuntimeError("No frames to concat.")
    if len(input_frames_bgr) != len(map_frames):
        print(f"Frame count mismatch: input={len(input_frames_bgr)}, "
              f"map={len(map_frames)}; using first {n}.")

    # Resize input frames to match map-frame height, preserving aspect.
    map_h, map_w = map_frames[0].shape[:2]
    in_h, in_w = input_frames_bgr[0].shape[:2]
    target_h = map_h
    target_w = max(2, int(round(in_w * target_h / in_h)))

    out_w = target_w + map_w
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, target_h)
    )

    for i in tqdm(range(n), desc="Concat input|map", unit="frame"):
        in_frame = cv2.resize(input_frames_bgr[i], (target_w, target_h),
                              interpolation=cv2.INTER_AREA)
        combined = np.concatenate([in_frame, map_frames[i]], axis=1)
        writer.write(combined)
    writer.release()
    print(f"Saved side-by-side video -> {out_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Create a current-position map video and concat with the input video"
    )
    p.add_argument("--meta_path", type=str, required=True)
    p.add_argument("--video_path", type=str, required=True,
                   help="Original input video (for side-by-side concat)")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--map_fps", type=float, default=16.0,
                   help="FPS of the output map video (should match run_video_inference.py --fps)")

    p.add_argument("--conf_threshold", type=float, default=3.0)
    p.add_argument("--depth_max", type=float, default=10.0)
    p.add_argument("--pixel_stride", type=int, default=3)
    p.add_argument("--frame_stride", type=int, default=1,
                   help="Temporal subsample for the STATIC map construction only "
                        "(the overlay animation always runs at every frame)")
    p.add_argument("--first_k_static", type=int, default=None,
                   help="Cap number of frames used to build the static map")
    p.add_argument("--resolution", type=int, default=1400)
    p.add_argument("--up_axis", type=str, default="y", choices=["x", "y", "z"])
    p.add_argument("--percentile_clip", type=float, default=1.0)
    p.add_argument("--pad_frac", type=float, default=0.05)

    p.add_argument("--skip_concat", action="store_true",
                   help="Only produce map.mp4, skip side-by-side concat")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load meta ────────────────────────────────────────────────────────────
    t0 = time.time()
    print(f"Loading {args.meta_path} ...")
    meta = np.load(args.meta_path)
    meta_dict = {k: meta[k] for k in
                 ("depth", "depth_conf", "intrinsic", "extrinsic_c2w", "images")}
    S = int(meta_dict["depth"].shape[0])
    cams_all = meta_dict["extrinsic_c2w"][:, :, 3].astype(np.float32)
    cam_R_all = meta_dict["extrinsic_c2w"][:, :, :3].astype(np.float32)
    print(f"  {S} frames")

    # ── Build static map (back-project + splat) ─────────────────────────────
    pts, cols, _ = collect_world_points(
        meta_dict,
        conf_threshold=args.conf_threshold,
        depth_max=args.depth_max,
        stride=args.pixel_stride,
        first_k=args.first_k_static,
        frame_stride=args.frame_stride,
    )
    print(f"Static map points: {pts.shape[0]:,}")

    # Use ALL camera positions when computing the bbox so the whole traj fits.
    bbox, pts, cols = crop_xz_range(pts, cols, cams_all,
                                    args.percentile_clip, args.pad_frac)
    print(f"XZ bbox: X=[{bbox[0]:.2f},{bbox[1]:.2f}] Z=[{bbox[2]:.2f},{bbox[3]:.2f}]")

    img_rgb, bbox = render_topdown(
        pts, cols, bbox, args.resolution, up_axis=args.up_axis
    )
    base_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # ── Render map video ─────────────────────────────────────────────────────
    map_path = str(out_dir / "map.mp4")
    render_map_video(
        base_bgr, cams_all, cam_R_all, bbox,
        out_path=map_path, fps=args.map_fps,
    )

    if args.skip_concat:
        print(f"\nDone. Map video: {map_path}")
        print(f"Total time: {time.time() - t0:.1f}s")
        return

    # ── Re-sample input video and concat ────────────────────────────────────
    input_frames = extract_target_frames(args.video_path, args.map_fps)
    if input_frames.shape[0] != S:
        print(f"Note: input re-sampled to {input_frames.shape[0]} frames, "
              f"meta has {S}; taking min.")
    concat_path = str(out_dir / "input_and_map.mp4")
    concat_input_and_map(input_frames, map_path, concat_path, fps=args.map_fps)

    print(f"\nDone. Outputs:")
    print(f"  map:        {map_path}")
    print(f"  input|map:  {concat_path}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
