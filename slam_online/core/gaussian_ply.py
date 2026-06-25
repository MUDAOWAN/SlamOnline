"""Shared Gaussian PLY helpers used by voting and object bbox scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


SH_C0 = 0.28209479177387814

PLY_DTYPE_MAP = {
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "uchar": "u1",
    "uint8": "u1",
    "char": "i1",
    "int8": "i1",
    "ushort": "u2",
    "uint16": "u2",
    "short": "i2",
    "int16": "i2",
    "uint": "u4",
    "uint32": "u4",
    "int": "i4",
    "int32": "i4",
}


def slugify(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def rgb_to_sh_dc(rgb: tuple[float, float, float]) -> np.ndarray:
    rgb_np = np.asarray(rgb, dtype=np.float32)
    return (rgb_np - 0.5) / SH_C0


def parse_ply_header(path: Path) -> tuple[list[str], int, list[tuple[str, str]], int]:
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {path}")
            text = line.decode("ascii", errors="replace").strip()
            header_lines.append(text)
            if text == "end_header":
                break
        data_start = f.tell()

    if "format binary_little_endian 1.0" not in header_lines:
        raise ValueError("Only binary_little_endian PLY is supported")

    vertex_count = None
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
            continue
        if in_vertex and len(parts) == 3 and parts[0] == "property":
            properties.append((parts[2], parts[1]))

    if vertex_count is None:
        raise ValueError("PLY has no vertex element")

    return header_lines, vertex_count, properties, data_start


def make_vertex_dtype(properties: list[tuple[str, str]]) -> np.dtype:
    dtype_fields = []
    for name, prop_type in properties:
        if prop_type not in PLY_DTYPE_MAP:
            raise ValueError(f"Unsupported PLY property type: {prop_type} for {name}")
        dtype_fields.append((name, "<" + PLY_DTYPE_MAP[prop_type]))
    return np.dtype(dtype_fields)


def read_binary_little_endian_gaussian_ply(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    header_lines, vertex_count, properties, data_start = parse_ply_header(path)

    prop_names = {name for name, _ in properties}
    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"}
    missing = required - prop_names
    if missing:
        raise ValueError(f"PLY is missing Gaussian fields: {sorted(missing)}")

    dtype = make_vertex_dtype(properties)
    with path.open("rb") as f:
        f.seek(data_start)
        vertices = np.frombuffer(
            f.read(vertex_count * dtype.itemsize),
            dtype=dtype,
            count=vertex_count,
        ).copy()

    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    return points, vertices, header_lines


def write_binary_little_endian_gaussian_ply(path: Path, vertices: np.ndarray, header_lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for line in header_lines:
            f.write((line + "\n").encode("ascii"))
        f.write(vertices.tobytes())
