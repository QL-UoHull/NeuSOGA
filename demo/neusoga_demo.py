#!/usr/bin/env python3
"""Lightweight NeuSOGA reproducibility demo.

This script illustrates the prototype pipeline
Observation -> Topology -> Geometry -> Symbol
using either a deterministic synthetic point cloud or a simple user-provided file.

It is intentionally lightweight and CPU-friendly. It does not claim full
benchmark reproduction for the accompanying research project.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import urllib.request
import zipfile
from collections import deque
from pathlib import Path
from typing import Iterable

MODELNET40_URL = "https://modelnet.cs.princeton.edu/ModelNet40.zip"
MAX_INPUT_POINTS = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight NeuSOGA demo.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional input point cloud (.txt, .csv, .xyz, or .off).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/demo"),
        help="Directory where demo outputs will be written.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data"),
        help="Root directory for external datasets such as ModelNet40.",
    )
    parser.add_argument(
        "--download-modelnet40",
        action="store_true",
        help="Download/extract ModelNet40 into --dataset-root if available.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        help="Optional external SAM checkpoint path recorded for provenance.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device intent recorded in run metadata only; no GPU kernels are invoked in this demo.",
    )
    parser.add_argument(
        "--neighborhood-radius",
        type=float,
        default=0.35,
        help="Radius used to construct the topology graph.",
    )
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=12,
        help="Minimum connected-component size kept during segmentation.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=360,
        help="Number of points to generate for the synthetic scene.",
    )
    return parser.parse_args()


def even_split(total: int, parts: int) -> list[int]:
    base, remainder = divmod(total, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def generate_plane(count: int) -> list[tuple[float, float, float]]:
    rows = max(4, int(math.sqrt(count)))
    cols = max(4, math.ceil(count / rows))
    points: list[tuple[float, float, float]] = []
    for row in range(rows):
        y = -0.8 + (1.6 * row / max(rows - 1, 1))
        for col in range(cols):
            if len(points) >= count:
                break
            x = -1.6 + (1.2 * col / max(cols - 1, 1))
            z = 0.02 * math.sin(row * 0.7) * math.cos(col * 0.5)
            points.append((x, y, z))
    return points


def generate_cylinder(count: int) -> list[tuple[float, float, float]]:
    height_levels = max(6, int(math.sqrt(count / 2)))
    angle_levels = max(10, math.ceil(count / height_levels))
    radius = 0.35
    points: list[tuple[float, float, float]] = []
    for height_index in range(height_levels):
        z = -0.8 + (1.6 * height_index / max(height_levels - 1, 1))
        for angle_index in range(angle_levels):
            if len(points) >= count:
                break
            theta = 2.0 * math.pi * angle_index / angle_levels
            x = 0.8 + radius * math.cos(theta)
            y = radius * math.sin(theta)
            points.append((x, y, z))
    return points


def generate_sphere(count: int) -> list[tuple[float, float, float]]:
    latitude_levels = max(6, int(math.sqrt(count / 2)))
    longitude_levels = max(10, math.ceil(count / latitude_levels))
    radius = 0.45
    points: list[tuple[float, float, float]] = []
    for latitude_index in range(latitude_levels):
        phi = math.pi * (0.2 + 0.6 * latitude_index / max(latitude_levels - 1, 1))
        for longitude_index in range(longitude_levels):
            if len(points) >= count:
                break
            theta = 2.0 * math.pi * longitude_index / longitude_levels
            x = 2.0 + radius * math.sin(phi) * math.cos(theta)
            y = 0.4 + radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)
            points.append((x, y, z))
    return points


def generate_synthetic_scene(count: int) -> tuple[list[tuple[float, float, float]], list[str]]:
    plane_count, cylinder_count, sphere_count = even_split(max(count, 36), 3)
    plane = generate_plane(plane_count)
    cylinder = generate_cylinder(cylinder_count)
    sphere = generate_sphere(sphere_count)
    points = plane + cylinder + sphere
    labels = (
        ["plane_seed"] * len(plane)
        + ["cylinder_seed"] * len(cylinder)
        + ["sphere_seed"] * len(sphere)
    )
    return points, labels


def parse_numeric_tokens(text: str) -> list[float]:
    cleaned = text.replace(",", " ").strip()
    if not cleaned:
        return []
    values: list[float] = []
    for token in cleaned.split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def load_text_point_cloud(path: Path) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = parse_numeric_tokens(line)
        if len(values) >= 3:
            points.append((values[0], values[1], values[2]))
    if not points:
        raise ValueError(f"No xyz points could be read from {path}.")
    return points


def load_off_point_cloud(path: Path) -> list[tuple[float, float, float]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or lines[0] not in {"OFF", "COFF"}:
        raise ValueError(f"{path} is not a valid OFF file.")
    counts = lines[1].split()
    if len(counts) < 1:
        raise ValueError(f"OFF header in {path} is incomplete.")
    vertex_count = int(counts[0])
    points: list[tuple[float, float, float]] = []
    for line in lines[2 : 2 + vertex_count]:
        values = parse_numeric_tokens(line)
        if len(values) >= 3:
            points.append((values[0], values[1], values[2]))
    if not points:
        raise ValueError(f"No vertices could be read from {path}.")
    return points


def load_point_cloud(path: Path) -> list[tuple[float, float, float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".off":
        return load_off_point_cloud(path)
    return load_text_point_cloud(path)


def download_modelnet40(dataset_root: Path) -> dict[str, str]:
    dataset_root.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_root / "ModelNet40.zip"
    partial_archive_path = dataset_root / "ModelNet40.zip.partial"
    extract_root = dataset_root / "ModelNet40"

    if not archive_path.exists():
        try:
            partial_archive_path.unlink(missing_ok=True)
            with urllib.request.urlopen(MODELNET40_URL) as response:
                with partial_archive_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            partial_archive_path.replace(archive_path)
        except Exception as exc:
            partial_archive_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Failed to download ModelNet40. Remove any partial archive and retry when network access is available."
            ) from exc

    if not extract_root.exists():
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(dataset_root)
        nested_root = dataset_root / "ModelNet40"
        if not nested_root.exists():
            extracted_dirs = [path for path in dataset_root.iterdir() if path.is_dir()]
            if extracted_dirs:
                nested_root = extracted_dirs[0]
            if nested_root != extract_root and nested_root.exists():
                if extract_root.exists():
                    shutil.rmtree(extract_root)
                nested_root.rename(extract_root)

    return {
        "archive": str(archive_path.resolve()),
        "extracted_root": str(extract_root.resolve()),
        "source_url": MODELNET40_URL,
    }


def squared_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def build_radius_graph(
    points: list[tuple[float, float, float]], radius: float
) -> list[list[int]]:
    radius_sq = radius * radius
    adjacency = [[] for _ in points]
    for index, point in enumerate(points):
        for neighbor_index in range(index + 1, len(points)):
            if squared_distance(point, points[neighbor_index]) <= radius_sq:
                adjacency[index].append(neighbor_index)
                adjacency[neighbor_index].append(index)
    return adjacency


def connected_components(adjacency: list[list[int]]) -> list[list[int]]:
    components: list[list[int]] = []
    visited = [False] * len(adjacency)
    for start in range(len(adjacency)):
        if visited[start]:
            continue
        queue: deque[int] = deque([start])
        visited[start] = True
        component: list[int] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(component)
    return components


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def std(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def classify_component(component_points: list[tuple[float, float, float]]) -> tuple[str, str, dict[str, float | list[float]]]:
    xs = [point[0] for point in component_points]
    ys = [point[1] for point in component_points]
    zs = [point[2] for point in component_points]
    cx, cy, cz = mean(xs), mean(ys), mean(zs)
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    z_span = max(zs) - min(zs)
    xy_radii = [math.hypot(x - cx, y - cy) for x, y, _ in component_points]
    radial_3d = [math.dist(point, (cx, cy, cz)) for point in component_points]
    xy_mean = mean(xy_radii)
    radial_mean = mean(radial_3d)
    xy_rel_std = (std(xy_radii) / xy_mean) if xy_mean else float("inf")
    radial_rel_std = (std(radial_3d) / radial_mean) if radial_mean else float("inf")
    z_std = std(zs)
    min_span = max(min(x_span, y_span, z_span), 1e-9)
    max_span = max(x_span, y_span, z_span)

    if z_std <= max(0.03, 0.08 * max(x_span, y_span, 1e-9)) and z_span <= max(
        0.12, 0.3 * max(x_span, y_span, 1e-9)
    ):
        primitive = "plane"
        symbolic = (
            f"plane(z≈{cz:.3f}, x∈[{min(xs):.3f}, {max(xs):.3f}], "
            f"y∈[{min(ys):.3f}, {max(ys):.3f}])"
        )
    elif xy_rel_std <= 0.18 and z_span >= 1.1 * max(x_span, y_span):
        primitive = "cylinder"
        symbolic = (
            f"cylinder(center≈({cx:.3f}, {cy:.3f}), radius≈{xy_mean:.3f}, "
            f"z∈[{min(zs):.3f}, {max(zs):.3f}])"
        )
    elif radial_rel_std <= 0.18 and (max_span / min_span) <= 1.8:
        primitive = "sphere"
        symbolic = (
            f"sphere(center≈({cx:.3f}, {cy:.3f}, {cz:.3f}), "
            f"radius≈{radial_mean:.3f})"
        )
    else:
        primitive = "unknown"
        symbolic = (
            f"primitive(points={len(component_points)}, "
            f"bbox=({x_span:.3f}, {y_span:.3f}, {z_span:.3f}))"
        )

    summary = {
        "centroid": [cx, cy, cz],
        "bbox_span": [x_span, y_span, z_span],
        "xy_radius_mean": xy_mean,
        "xy_radius_relative_std": xy_rel_std,
        "radial_mean": radial_mean,
        "radial_relative_std": radial_rel_std,
        "z_std": z_std,
    }
    return primitive, symbolic, summary


def select_components(components: list[list[int]], min_size: int) -> list[list[int]]:
    kept = [component for component in components if len(component) >= min_size]
    if kept:
        return kept
    return [max(components, key=len)] if components else []


def downsample_points(
    points: list[tuple[float, float, float]], labels: list[str], max_points: int
) -> tuple[list[tuple[float, float, float]], list[str], dict[str, int] | None]:
    if len(points) <= max_points:
        return points, labels, None

    indices = [
        min(math.floor(index * len(points) / max_points), len(points) - 1)
        for index in range(max_points)
    ]
    sampled_points = [points[index] for index in indices]
    sampled_labels = [labels[index] for index in indices]
    return (
        sampled_points,
        sampled_labels,
        {
            "original_points": len(points),
            "retained_points": len(sampled_points),
            "max_points": max_points,
        },
    )


def write_outputs(
    output_dir: Path,
    points: list[tuple[float, float, float]],
    labels: list[str],
    components: list[list[int]],
    source: str,
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    component_ids = [-1] * len(points)
    segment_rows: list[dict[str, object]] = []
    symbolic_lines = [
        "# NeuSOGA symbolic abstraction",
        f"# Source: {source}",
        "",
    ]

    for component_id, component in enumerate(components):
        component_points = [points[index] for index in component]
        primitive, symbolic, summary = classify_component(component_points)
        for index in component:
            component_ids[index] = component_id
        segment_rows.append(
            {
                "component_id": component_id,
                "primitive": primitive,
                "symbolic": symbolic,
                "num_points": len(component),
                **summary,
            }
        )
        symbolic_lines.append(f"[segment {component_id}] {symbolic}")

    with (output_dir / "point_cloud.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "z", "component_id", "seed_label"])
        for index, (x, y, z) in enumerate(points):
            writer.writerow([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", component_ids[index], labels[index]])

    (output_dir / "symbolic_representation.txt").write_text(
        "\n".join(symbolic_lines) + "\n", encoding="utf-8"
    )
    (output_dir / "segments.json").write_text(
        json.dumps(segment_rows, indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "source": source,
                "num_points": len(points),
                "num_segments": len(components),
                "metadata": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    metadata: dict[str, object] = {
        "device": args.device,
        "neighborhood_radius": args.neighborhood_radius,
        "min_component_size": args.min_component_size,
        "demo_scope": (
            "Lightweight reproducibility demo; not a full benchmark or full research pipeline."
        ),
    }

    if args.sam_checkpoint:
        if not args.sam_checkpoint.exists():
            raise FileNotFoundError(
                f"SAM checkpoint path does not exist: {args.sam_checkpoint}"
            )
        metadata["sam_checkpoint"] = str(args.sam_checkpoint.resolve())
        metadata["sam_note"] = (
            "Checkpoint recorded for provenance only; the lightweight demo does not invoke SAM directly."
        )
    else:
        metadata["sam_note"] = (
            "No SAM checkpoint supplied. External checkpoints are required only for extended workflows."
        )

    if args.download_modelnet40:
        metadata["modelnet40"] = download_modelnet40(args.dataset_root)
    else:
        metadata["modelnet40"] = {
            "note": "ModelNet40 not downloaded during this run.",
            "expected_root": str((args.dataset_root / "ModelNet40").resolve()),
        }

    if args.input:
        points = load_point_cloud(args.input)
        labels = ["input_point"] * len(points)
        points, labels, downsampling = downsample_points(points, labels, MAX_INPUT_POINTS)
        if downsampling:
            metadata["input_downsampling"] = downsampling
        source = str(args.input.resolve())
    else:
        points, labels = generate_synthetic_scene(args.num_points)
        source = "deterministic_synthetic_scene"
        metadata["synthetic_components"] = ["plane_seed", "cylinder_seed", "sphere_seed"]

    adjacency = build_radius_graph(points, args.neighborhood_radius)
    raw_components = connected_components(adjacency)
    components = select_components(raw_components, args.min_component_size)
    write_outputs(args.output_dir, points, labels, components, source, metadata)

    print("NeuSOGA demo completed.")
    print(f"Source: {source}")
    print(f"Segments kept: {len(components)}")
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
