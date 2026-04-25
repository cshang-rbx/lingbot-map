"""Render a bird's-eye (top-down) visualization from a LingBot-Map meta.npz.

Aggregates all per-frame predictions into one 2D top-down (world-frame) image:

- Point cloud is built by back-projecting per-frame depth with the per-frame
  intrinsics and transforming by the per-frame c2w extrinsic (the `world_points`
  tensor saved by the streaming model appears to be a normalized latent and is
  NOT used here).
- Points are filtered by `depth_conf >= conf_threshold`, an optional per-frame
  depth cap (to drop sky / faraway points), and an XZ bounding-box crop.
- Points are then splatted into a top-down image (projection plane defaults to
  XZ, so "up" axis is Y); each pixel holds the median color (uint8) of the
  lowest-Y points falling into it — i.e. the image you would see looking
  straight down from the sky.
- Camera trajectory is drawn as a polyline; optional forward/up frustum lines
  show heading.
- Optionally, `point_cloud.ply` is also splatted into a second image for easy
  comparison.

Example:
    python visualize_birdview.py \
        --meta_path outputs/syn122_wooden_barrel/meta.npz \
        --out_path  outputs/syn122_wooden_barrel/bird_view.png
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm


# =============================================================================
# Geometry helpers
# =============================================================================

def invert_extrinsics(E: np.ndarray) -> np.ndarray:
    """Invert a stack of 3x4 or 4x4 rigid transforms, preserving 3x4 output."""
    E = np.asarray(E)
    R = E[..., :3, :3]
    t = E[..., :3, 3:4]
    R_inv = np.swapaxes(R, -1, -2)
    t_inv = -(R_inv @ t)
    return np.concatenate([R_inv, t_inv], axis=-1).astype(E.dtype, copy=False)


def get_camera_to_world(meta: dict, convention: str = "c2w") -> np.ndarray:
    """Return camera-to-world extrinsics from meta arrays.

    ``run_video_inference.py`` writes ``extrinsic_c2w``. Older experimental
    outputs may have stored a world-to-camera matrix under that key; pass
    ``convention='w2c'`` to render those without camera pan becoming fake
    translation.
    """
    if convention not in {"c2w", "w2c"}:
        raise ValueError(f"convention must be 'c2w' or 'w2c', got {convention!r}")

    if "extrinsic_c2w" in meta:
        E = np.asarray(meta["extrinsic_c2w"])
    elif "extrinsic_w2c" in meta:
        E = np.asarray(meta["extrinsic_w2c"])
        convention = "w2c"
    else:
        raise KeyError("meta must contain 'extrinsic_c2w' or 'extrinsic_w2c'")

    return invert_extrinsics(E) if convention == "w2c" else E


def backproject_frame(depth: np.ndarray, K: np.ndarray, E_c2w: np.ndarray,
                      image_rgb: np.ndarray, conf: np.ndarray,
                      conf_threshold: float, depth_max: float,
                      stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Back-project one frame's depth into world coordinates.

    Args:
        depth:     (H, W) metric depth
        K:         (3, 3) camera intrinsics
        E_c2w:     (3, 4) camera-to-world
        image_rgb: (H, W, 3) uint8
        conf:      (H, W) confidence map (depth_conf or world_points_conf)
        conf_threshold: keep pixels with conf >= threshold
        depth_max: drop depths greater than this (0 = disable)
        stride:    spatial subsample stride

    Returns:
        (pts_world [N,3] float32, cols_rgb [N,3] uint8)
    """
    d = depth[::stride, ::stride]
    c = conf[::stride, ::stride]
    rgb = image_rgb[::stride, ::stride]
    H, W = d.shape
    vs, us = np.mgrid[0:H, 0:W].astype(np.float32)
    us_full = us * stride
    vs_full = vs * stride

    mask = np.isfinite(d) & (d > 0) & (c >= conf_threshold)
    if depth_max > 0:
        mask &= d <= depth_max
    if not mask.any():
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)

    u = us_full[mask]; v = vs_full[mask]; z = d[mask]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (u - cx) / fx * z
    y_cam = (v - cy) / fy * z
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1)          # (N, 3)
    R = E_c2w[:, :3]; t = E_c2w[:, 3]
    pts_world = pts_cam @ R.T + t                            # (N, 3)
    cols = rgb[mask]                                         # (N, 3)
    return pts_world.astype(np.float32), cols.astype(np.uint8)


def collect_world_points(meta: dict, conf_threshold: float, depth_max: float,
                         stride: int, first_k: int | None,
                         frame_stride: int,
                         extrinsic_convention: str = "c2w"
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project every (sub-sampled) frame. Returns (pts, cols, cam_positions)."""
    depth = meta["depth"][..., 0] if meta["depth"].ndim == 4 else meta["depth"]
    K = meta["intrinsic"]
    E = get_camera_to_world(meta, extrinsic_convention)
    conf = meta["depth_conf"]
    images = meta["images"]

    S = depth.shape[0]
    frame_idx = list(range(0, S, frame_stride))
    if first_k:
        frame_idx = frame_idx[:first_k]

    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    cams = np.zeros((len(frame_idx), 3), np.float32)

    for out_i, fi in enumerate(tqdm(frame_idx, desc="Back-projecting", unit="frame")):
        pts, cols = backproject_frame(
            depth[fi], K[fi], E[fi], images[fi], conf[fi],
            conf_threshold, depth_max, stride,
        )
        all_pts.append(pts)
        all_cols.append(cols)
        cams[out_i] = E[fi, :, 3]

    if all_pts:
        pts = np.concatenate(all_pts, axis=0)
        cols = np.concatenate(all_cols, axis=0)
    else:
        pts = np.empty((0, 3), np.float32)
        cols = np.empty((0, 3), np.uint8)
    return pts, cols, cams


# =============================================================================
# Top-down splatting
# =============================================================================

def crop_xz_range(pts: np.ndarray, cols: np.ndarray, cams: np.ndarray,
                  percentile_clip: float, pad_frac: float
                  ) -> tuple[tuple[float, float, float, float], np.ndarray, np.ndarray]:
    """Compute an XZ bounding box using percentiles (robust to outliers)."""
    xz = np.concatenate([pts[:, [0, 2]], cams[:, [0, 2]]], axis=0)
    lo = np.percentile(xz, percentile_clip, axis=0)
    hi = np.percentile(xz, 100.0 - percentile_clip, axis=0)
    span = hi - lo
    lo = lo - pad_frac * span
    hi = hi + pad_frac * span
    x_min, z_min = lo
    x_max, z_max = hi
    mask = (
        (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
        (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    )
    return (float(x_min), float(x_max), float(z_min), float(z_max)), pts[mask], cols[mask]


def render_topdown(pts: np.ndarray, cols: np.ndarray,
                   bbox: tuple[float, float, float, float],
                   resolution_px: int,
                   up_axis: str = "y",
                   bg_color: tuple[int, int, int] = (20, 20, 20)
                   ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Splat (X, Z) with per-pixel top-most (smallest Y) color.

    up_axis='y' means Y is up, so we look down onto the XZ plane.
    The output is a square-ish image whose aspect matches the bbox aspect.
    """
    x_min, x_max, z_min, z_max = bbox
    x_span = x_max - x_min
    z_span = z_max - z_min
    if x_span >= z_span:
        W = resolution_px
        H = max(2, int(round(resolution_px * z_span / max(x_span, 1e-6))))
    else:
        H = resolution_px
        W = max(2, int(round(resolution_px * x_span / max(z_span, 1e-6))))
    scale_x = (W - 1) / max(x_span, 1e-6)
    scale_z = (H - 1) / max(z_span, 1e-6)

    img = np.full((H, W, 3), bg_color, np.uint8)
    zbuf = np.full((H, W), np.inf, np.float32)  # "top" = smallest value on up axis

    if up_axis == "y":
        up_vals = pts[:, 1]
    elif up_axis == "z":
        up_vals = pts[:, 2]
    else:
        up_vals = pts[:, 0]

    xs = pts[:, 0]
    zs = pts[:, 2]
    u = np.clip(((xs - x_min) * scale_x).astype(np.int32), 0, W - 1)
    # Image row grows downward; invert Z so larger-Z appears at the top.
    v = np.clip(((z_max - zs) * scale_z).astype(np.int32), 0, H - 1)

    # Single pass: keep the point with smallest up-axis value per pixel.
    # Sort by up_vals descending so the smallest overwrites last.
    order = np.argsort(-up_vals)
    u_o = u[order]; v_o = v[order]; c_o = cols[order]; up_o = up_vals[order]
    img[v_o, u_o] = c_o
    zbuf[v_o, u_o] = np.minimum(zbuf[v_o, u_o], up_o)

    return img, (x_min, x_max, z_min, z_max)


def draw_trajectory(img: np.ndarray, cams: np.ndarray,
                    bbox: tuple[float, float, float, float],
                    line_color: tuple[int, int, int] = (0, 255, 255),
                    start_color: tuple[int, int, int] = (0, 255, 0),
                    end_color: tuple[int, int, int] = (0, 0, 255),
                    thickness: int = 2, radius: int = 5) -> np.ndarray:
    """Overlay camera XZ trajectory. Returns BGR image suitable for writing."""
    x_min, x_max, z_min, z_max = bbox
    H, W = img.shape[:2]
    sx = (W - 1) / max(x_max - x_min, 1e-6)
    sz = (H - 1) / max(z_max - z_min, 1e-6)

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pts_px = np.stack([
        np.clip(((cams[:, 0] - x_min) * sx).astype(np.int32), 0, W - 1),
        np.clip(((z_max - cams[:, 2]) * sz).astype(np.int32), 0, H - 1),
    ], axis=1)
    for i in range(1, len(pts_px)):
        cv2.line(bgr, tuple(pts_px[i - 1]), tuple(pts_px[i]),
                 line_color[::-1], thickness, cv2.LINE_AA)
    if len(pts_px):
        cv2.circle(bgr, tuple(pts_px[0]), radius, start_color[::-1], -1, cv2.LINE_AA)
        cv2.circle(bgr, tuple(pts_px[-1]), radius, end_color[::-1], -1, cv2.LINE_AA)
    return bgr


def draw_axes_legend(bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    x_min, x_max, z_min, z_max = bbox
    H, W = bgr.shape[:2]
    txt_lines = [
        f"Top-down (XZ plane, looking down -Y)",
        f"X: [{x_min:.2f}, {x_max:.2f}] m",
        f"Z: [{z_min:.2f}, {z_max:.2f}] m",
        f"image: {W}x{H} px",
        "green=start  red=end  yellow=trajectory",
    ]
    y = 22
    for line in txt_lines:
        cv2.putText(bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 18
    return bgr


# =============================================================================
# PLY helpers (optional second panel)
# =============================================================================

def read_ply_xyz_rgb(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Minimal binary_little_endian PLY reader for our writer's layout."""
    with open(path, "rb") as f:
        header_lines: list[bytes] = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        header = b"".join(header_lines).decode("ascii", "ignore")
        n = 0
        for line in header.splitlines():
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
                break
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ])
        buf = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype, count=n)
    xyz = np.stack([buf["x"], buf["y"], buf["z"]], axis=1).astype(np.float32)
    rgb = np.stack([buf["r"], buf["g"], buf["b"]], axis=1).astype(np.uint8)
    return xyz, rgb


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description="Bird's-eye visualization of LingBot-Map outputs")
    p.add_argument("--meta_path", type=str, required=True,
                   help="Path to meta.npz produced by run_video_inference.py")
    p.add_argument("--out_path", type=str, required=True,
                   help="Output bird's-eye PNG path")
    p.add_argument("--ply_path", type=str, default=None,
                   help="Optional point_cloud.ply path to render in a second panel")

    p.add_argument("--resolution", type=int, default=1600, help="Longest-side px")
    p.add_argument("--up_axis", type=str, default="y", choices=["x", "y", "z"],
                   help="World up axis; project onto the remaining two (top-down view)")
    p.add_argument("--conf_threshold", type=float, default=3.0,
                   help="Min depth_conf per pixel")
    p.add_argument("--depth_max", type=float, default=10.0,
                   help="Drop points farther than this (meters). 0 disables.")
    p.add_argument("--pixel_stride", type=int, default=4,
                   help="Spatial subsample of each frame (px)")
    p.add_argument("--frame_stride", type=int, default=2,
                   help="Temporal subsample of frames")
    p.add_argument("--first_k", type=int, default=None,
                   help="Only use the first K sampled frames")
    p.add_argument("--extrinsic_convention", type=str, default="c2w",
                   choices=["c2w", "w2c"],
                   help="Convention of the saved extrinsic matrix. Use w2c for "
                        "older outputs whose extrinsic_c2w key was mislabeled.")
    p.add_argument("--percentile_clip", type=float, default=1.0,
                   help="XZ bbox percentile (1 = trim top/bottom 1%)")
    p.add_argument("--pad_frac", type=float, default=0.05,
                   help="Extra XZ padding as a fraction of span")
    p.add_argument("--no_trajectory", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    print(f"Loading {args.meta_path} ...")
    meta = np.load(args.meta_path)
    meta_dict = {k: meta[k] for k in ("depth", "depth_conf", "intrinsic", "images")}
    if "extrinsic_c2w" in meta:
        meta_dict["extrinsic_c2w"] = meta["extrinsic_c2w"]
    elif "extrinsic_w2c" in meta:
        meta_dict["extrinsic_w2c"] = meta["extrinsic_w2c"]
    else:
        raise KeyError("meta must contain 'extrinsic_c2w' or 'extrinsic_w2c'")
    S = meta_dict["depth"].shape[0]
    print(f"  {S} frames, {meta_dict['depth'].shape[1:]} depth")

    # 1) Back-project all frames to world points
    pts, cols, cams = collect_world_points(
        meta_dict,
        conf_threshold=args.conf_threshold,
        depth_max=args.depth_max,
        stride=args.pixel_stride,
        first_k=args.first_k,
        frame_stride=args.frame_stride,
        extrinsic_convention=args.extrinsic_convention,
    )
    print(f"Collected {pts.shape[0]:,} world points ({cams.shape[0]} cam poses)")

    # 2) Determine XZ bbox
    bbox, pts, cols = crop_xz_range(pts, cols, cams, args.percentile_clip, args.pad_frac)
    print(f"XZ bbox: X=[{bbox[0]:.2f},{bbox[1]:.2f}] Z=[{bbox[2]:.2f},{bbox[3]:.2f}]")

    # 3) Splat top-down
    print("Rendering top-down...")
    img_rgb, bbox = render_topdown(pts, cols, bbox, args.resolution, up_axis=args.up_axis)

    # 4) Overlay trajectory
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) if args.no_trajectory \
        else draw_trajectory(img_rgb, cams, bbox)
    bgr = draw_axes_legend(bgr, bbox)

    # 5) Optional: add PLY panel side-by-side
    if args.ply_path and os.path.exists(args.ply_path):
        print(f"Loading PLY {args.ply_path} ...")
        ply_xyz, ply_rgb = read_ply_xyz_rgb(args.ply_path)
        print(f"  {ply_xyz.shape[0]:,} points")
        # Use a big bbox that contains both (so the scales are directly comparable)
        xz_all = np.concatenate(
            [ply_xyz[:, [0, 2]], pts[:, [0, 2]], cams[:, [0, 2]]], axis=0
        ) if ply_xyz.size else np.concatenate([pts[:, [0, 2]], cams[:, [0, 2]]], axis=0)
        lo = np.percentile(xz_all, args.percentile_clip, axis=0)
        hi = np.percentile(xz_all, 100 - args.percentile_clip, axis=0)
        span = hi - lo
        lo = lo - args.pad_frac * span
        hi = hi + args.pad_frac * span
        big_bbox = (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))

        def _splat(xyz: np.ndarray, rgb: np.ndarray) -> np.ndarray:
            m = (
                (xyz[:, 0] >= big_bbox[0]) & (xyz[:, 0] <= big_bbox[1]) &
                (xyz[:, 2] >= big_bbox[2]) & (xyz[:, 2] <= big_bbox[3])
            )
            img, bb = render_topdown(xyz[m], rgb[m], big_bbox, args.resolution,
                                     up_axis=args.up_axis)
            bgrx = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if not args.no_trajectory:
                bgrx = draw_trajectory(img, cams, bb)
            return bgrx

        bgr_back = _splat(pts, cols)
        bgr_ply = _splat(ply_xyz, ply_rgb)

        def _label(img: np.ndarray, text: str) -> np.ndarray:
            img = img.copy()
            cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
            cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 1, cv2.LINE_AA)
            return img

        # Resize so both panels have same height
        h = max(bgr_back.shape[0], bgr_ply.shape[0])
        def _pad_h(a):
            pad = h - a.shape[0]
            if pad <= 0: return a
            return cv2.copyMakeBorder(a, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        left = _label(_pad_h(bgr_back),
                      f"Back-projected depth ({pts.shape[0]:,} pts)")
        right = _label(_pad_h(bgr_ply), f"point_cloud.ply ({ply_xyz.shape[0]:,} pts)")
        bgr = np.concatenate([left, right], axis=1)
        bgr = draw_axes_legend(bgr, big_bbox)

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out_path, bgr)
    print(f"Saved {args.out_path}  [{bgr.shape[1]}x{bgr.shape[0]}]")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
