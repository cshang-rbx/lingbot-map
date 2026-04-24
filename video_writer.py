"""Lightweight HEVC/H.264 video writer that pipes BGR frames to ffmpeg.

Drop-in replacement for ``cv2.VideoWriter`` when you want ``libx265``:

    writer = X265Writer("out.mp4", fps=16, size=(W, H), crf=28, preset="medium")
    for bgr in frames:
        writer.write(bgr)
    writer.release()

Falls back gracefully to ``cv2.VideoWriter`` (mp4v) if the ``ffmpeg`` binary is
not on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional

import cv2
import numpy as np


class X265Writer:
    """Pipe raw BGR frames to an ffmpeg subprocess that encodes with libx265."""

    def __init__(
        self,
        path: str,
        fps: float,
        size: tuple[int, int],
        crf: int = 28,
        preset: str = "medium",
        codec: str = "libx265",
        tag: str = "hvc1",
        pix_fmt_out: str = "yuv420p",
        loglevel: str = "error",
    ) -> None:
        """
        Args:
            path: output path (should end in .mp4)
            fps: frame rate
            size: (width, height)
            crf: constant-rate-factor; lower = higher quality/size. 28 is a
                 good default for dense top-down / screen-captured content.
            preset: libx265 preset (ultrafast, fast, medium, slow, ...).
            codec: "libx265" (HEVC, default) or "libx264" (H.264).
            tag: container tag ("hvc1" makes HEVC playable in QuickTime/Safari).
            pix_fmt_out: output pixel format; yuv420p is the widely-compatible choice.
            loglevel: ffmpeg loglevel (error, warning, info, ...).
        """
        self.path = path
        self.size = size
        self.fps = float(fps)
        self.codec = codec
        self._closed = False
        w, h = size

        if shutil.which("ffmpeg") is None:
            # Fallback — caller API is the same.
            self._fallback = True
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._vw: Optional[cv2.VideoWriter] = cv2.VideoWriter(
                path, fourcc, self.fps, (w, h)
            )
            self._proc: Optional[subprocess.Popen] = None
            return

        self._fallback = False
        self._vw = None

        args = [
            "ffmpeg", "-hide_banner", "-loglevel", loglevel, "-y",
            # Raw input: BGR frames at fixed rate.
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{self.fps}",
            "-i", "-",
            # Output: HEVC/H.264 CRF, broadly compatible pixel format.
            "-c:v", codec, "-crf", str(crf), "-preset", preset,
            "-pix_fmt", pix_fmt_out,
        ]
        if codec == "libx265":
            args += ["-tag:v", tag]
        args += [path]

        self._proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("X265Writer already released")
        if self._fallback:
            assert self._vw is not None
            self._vw.write(frame_bgr)
            return
        assert self._proc is not None and self._proc.stdin is not None
        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8)
        if not frame_bgr.flags["C_CONTIGUOUS"]:
            frame_bgr = np.ascontiguousarray(frame_bgr)
        try:
            self._proc.stdin.write(frame_bgr.tobytes())
        except BrokenPipeError as e:
            err = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
            raise RuntimeError(f"ffmpeg stdin closed early: {err.strip()}") from e

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._fallback:
            if self._vw is not None:
                self._vw.release()
            return
        assert self._proc is not None
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        ret = self._proc.wait()
        if ret != 0:
            err = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
            raise RuntimeError(f"ffmpeg exited with code {ret}: {err.strip()}")

    def __enter__(self) -> "X265Writer":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
