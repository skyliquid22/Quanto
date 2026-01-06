"""Deterministic PNG plotter for equity curves."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import struct
import zlib


def render_equity_curves(
    path: Path,
    curves: Mapping[str, Sequence[float]],
    *,
    width: int = 720,
    height: int = 360,
) -> Path:
    """Render multiple equity curves into a deterministic PNG."""

    path.parent.mkdir(parents=True, exist_ok=True)
    margin_left, margin_right, margin_top, margin_bottom = 60, 20, 20, 40
    plot_width = max(width - margin_left - margin_right, 1)
    plot_height = max(height - margin_top - margin_bottom, 1)
    pixels = bytearray([255] * width * height * 3)

    def set_pixel(x: int, y: int, color: Sequence[int]) -> None:
        if not (0 <= x < width and 0 <= y < height):
            return
        index = (y * width + x) * 3
        pixels[index : index + 3] = bytes(color)

    # Axes
    axis_color = (0, 0, 0)
    x_axis_y = height - margin_bottom
    for x in range(margin_left, width - margin_right):
        set_pixel(x, x_axis_y, axis_color)
    for y in range(margin_top, height - margin_bottom + 1):
        set_pixel(margin_left, y, axis_color)

    all_values = [value for series in curves.values() for value in series]
    if not all_values:
        _write_png(path, pixels, width, height)
        return path

    min_val = min(all_values)
    max_val = max(all_values)
    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5

    def y_coord(value: float) -> int:
        ratio = (value - min_val) / (max_val - min_val)
        ratio = max(0.0, min(1.0, ratio))
        return int(height - margin_bottom - ratio * plot_height)

    total_points = max(len(next(iter(curves.values()))), 1)
    x_step = plot_width / max(total_points - 1, 1)

    palette = [
        (31, 119, 180),  # strategy - blue
        (128, 128, 128),  # benchmark - gray
        (214, 39, 40),
        (44, 160, 44),
    ]

    for index, (label, series) in enumerate(curves.items()):
        color = palette[index % len(palette)]
        previous = None
        for point_idx, value in enumerate(series):
            x = int(margin_left + round(point_idx * x_step))
            y = y_coord(float(value))
            if previous is not None:
                _draw_line(set_pixel, previous, (x, y), color)
            previous = (x, y)
        # Minimal deterministic legend marker
        legend_x = margin_left + 10
        legend_y = margin_top + 14 * index
        for offset in range(12):
            set_pixel(legend_x + offset, legend_y, color)

    _write_png(path, pixels, width, height)
    return path


def _draw_line(set_pixel, start, end, color) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    steps = max(dx, dy, 1)
    for step in range(steps + 1):
        t = step / steps
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        set_pixel(x, y, color)


def _write_png(path: Path, pixels: bytearray, width: int, height: int) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    rows = bytearray()
    row_bytes = width * 3
    for y in range(height):
        start = y * row_bytes
        rows.append(0)  # filter byte
        rows.extend(pixels[start : start + row_bytes])

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(rows), level=9)
    with path.open("wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(chunk(b"IHDR", ihdr))
        handle.write(chunk(b"IDAT", idat))
        handle.write(chunk(b"IEND", b""))


__all__ = ["render_equity_curves"]
