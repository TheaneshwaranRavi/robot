"""Seeded synthetic defect dataset for the SO-101 sorting cell (phase 3, milestone 2).

Every sample is one image from the close-up ``inspect`` camera of a "plate" part on the
inspection pad, plus ground truth. Parts are nominal (with harmless tolerance variation) or carry
one defect:

  missing_hole   one of the two drilled holes is absent
  scratch        one to three thin dark or bright scratch marks on the top face
  wrong_colour   finish colour far outside the nominal tolerance
  deformed_size  plate length or width off by 12-30 %

Domain randomisation per sample: part position and yaw, light intensity and position, camera
position and field of view, table and pad colour, exposure gain, blur and sensor noise. The
"shift" split uses wider, unseen ranges to measure robustness.

Splits come from disjoint seed streams (SeedSequence([seed, split, index])), so no two samples
share a seed, and the same --seed reproduces the dataset bit for bit. "train" holds good parts
only (one-class training); val/test/shift hold good and defective parts.

Layout (Anomalib Folder-compatible):
  <out>/images/<split>/{good,defect}/<id>.png
  <out>/masks/<split>/defect/<id>.png      binary defect mask (255 = defect)
  <out>/manifest.jsonl                     one JSON record per sample
  <out>/dataset.json                       config, counts, generator version, manifest hash

Run:  python -m phase3.dataset --out outputs/phase3/dataset --seed 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageFilter

from .scene import (HOLE_RADIUS, HOLE_X, INSPECT_CAM, INSPECT_PAD, N_SCRATCH_SLOTS, PART_RGBA, PART_SIZE,
                    SortCell, build_cell)

GENERATOR_VERSION = "phase3-dataset/1"
DEFECT_TYPES = ("missing_hole", "scratch", "wrong_colour", "deformed_size")
SPLITS = ("train", "val", "test", "shift")
SPLIT_INDEX = {name: i for i, name in enumerate(SPLITS)}
IMAGE_SIZE = 256

# Ranges: (low, high). "shift" widens the nuisance ranges; defect ranges never change.
NUISANCE = {
    "nominal": {"pos_jitter_m": 0.006, "light_gain": (0.75, 1.20), "light_pos_m": 0.15,
                "cam_pos_m": 0.004, "fovy_deg": 0.5, "surface_jitter": 0.04, "exposure": (0.85, 1.15),
                "blur_sigma": (0.0, 0.7), "noise_std": (0.0, 3.0)},
    "shift": {"pos_jitter_m": 0.010, "light_gain": (0.50, 1.45), "light_pos_m": 0.30,
              "cam_pos_m": 0.010, "fovy_deg": 1.5, "surface_jitter": 0.12, "exposure": (0.65, 1.40),
              "blur_sigma": (0.3, 1.4), "noise_std": (2.0, 7.0)},
}
# Harmless manufacturing tolerance on good parts.
TOLERANCE = {"colour": 0.025, "size": 0.01, "hole_radius": 0.04}
WRONG_COLOURS = ((0.85, 0.45, 0.15), (0.55, 0.57, 0.60), (0.25, 0.60, 0.30), (0.12, 0.20, 0.45),
                 (0.75, 0.20, 0.25))
MIN_VISIBLE_HOLE_PX = 40          # a drawn hole covers ~140 px at 256 x 256
MIN_SCRATCH_PX = 8


@dataclass
class Sample:
    id: str
    split: str
    index: int
    seed_entropy: list[int]
    label: str                        # "good" | "defect"
    defect_type: str                  # "none" or one of DEFECT_TYPES
    severity: dict[str, Any]
    part_pose: dict[str, float]
    nuisance: dict[str, Any]
    bbox_xyxy: list[int]
    defect_pixels: int
    gt_checks: dict[str, Any]
    image: str
    mask: str | None = None
    tolerance: dict[str, float] = field(default_factory=dict)


class DefectRenderer:
    """Applies one sample's part variant and nuisance settings to the cell, renders, restores."""

    def __init__(self, cell: SortCell) -> None:
        self.cell = cell
        m = cell.model
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.part = gid("part_geom")
        self.holes = [gid(f"part_hole_{i}") for i in range(len(HOLE_X))]
        self.scratches = [gid(f"part_scratch_{i}") for i in range(N_SCRATCH_SLOTS)]
        self.surfaces = [gid("table"), gid("inspect_pad")]
        self.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "inspect")
        self.lights = list(range(m.nlight))
        self._nominal = {
            "geom_size": m.geom_size.copy(), "geom_pos": m.geom_pos.copy(), "geom_rgba": m.geom_rgba.copy(),
            "geom_quat": m.geom_quat.copy(), "cam_pos": m.cam_pos.copy(), "cam_fovy": m.cam_fovy.copy(),
            "light_diffuse": m.light_diffuse.copy(), "light_pos": m.light_pos.copy(),
        }

    def restore(self) -> None:
        m = self.cell.model
        m.geom_size[:] = self._nominal["geom_size"]
        m.geom_pos[:] = self._nominal["geom_pos"]
        m.geom_rgba[:] = self._nominal["geom_rgba"]
        m.geom_quat[:] = self._nominal["geom_quat"]
        m.cam_pos[:] = self._nominal["cam_pos"]
        m.cam_fovy[:] = self._nominal["cam_fovy"]
        m.light_diffuse[:] = self._nominal["light_diffuse"]
        m.light_pos[:] = self._nominal["light_pos"]

    # -- part variants --------------------------------------------------------------------------
    def _set_plate(self, sx: float, sy: float, rgba, hole_r: float) -> None:
        m = self.cell.model
        m.geom_size[self.part, :2] = (sx / 2, sy / 2)
        m.geom_rgba[self.part] = rgba
        scale_x = sx / PART_SIZE[0]
        for g, hx in zip(self.holes, HOLE_X):
            m.geom_pos[g, 0] = hx * scale_x
            m.geom_size[g, 0] = hole_r

    def apply_part(self, rng: np.random.Generator, defect: str) -> dict[str, Any]:
        """Nominal tolerance first, then (maybe) one defect. Returns the defect severity record."""
        m = self.cell.model
        tol_c = rng.uniform(-TOLERANCE["colour"], TOLERANCE["colour"], 3)
        rgba = np.clip(np.array(PART_RGBA) + np.append(tol_c, 0.0), 0, 1)
        sx = PART_SIZE[0] * (1 + rng.uniform(-TOLERANCE["size"], TOLERANCE["size"]))
        sy = PART_SIZE[1] * (1 + rng.uniform(-TOLERANCE["size"], TOLERANCE["size"]))
        hole_r = HOLE_RADIUS * (1 + rng.uniform(-TOLERANCE["hole_radius"], TOLERANCE["hole_radius"]))
        severity: dict[str, Any] = {}
        if defect == "wrong_colour":
            base = np.array(WRONG_COLOURS[rng.integers(len(WRONG_COLOURS))])
            rgba = np.append(np.clip(base + rng.uniform(-0.04, 0.04, 3), 0, 1), 1.0)
            severity = {"rgb": [round(float(v), 4) for v in rgba[:3]],
                        "delta_from_nominal": round(float(np.linalg.norm(rgba[:3] - np.array(PART_RGBA[:3]))), 4)}
        elif defect == "deformed_size":
            axis = int(rng.integers(2))
            factor = float(rng.choice([rng.uniform(0.70, 0.88), rng.uniform(1.12, 1.30)]))
            if axis == 0:
                sx = PART_SIZE[0] * factor
            else:
                sy = PART_SIZE[1] * factor
            severity = {"axis": "xy"[axis], "scale": round(factor, 4)}
        self._set_plate(sx, sy, rgba, hole_r)
        if defect == "missing_hole":
            which = int(rng.integers(len(self.holes)))
            m.geom_rgba[self.holes[which], 3] = 0.0
            severity = {"hole": which}
        elif defect == "scratch":
            n = int(rng.integers(1, N_SCRATCH_SLOTS + 1))
            marks = []
            for g in self.scratches[:n]:
                length = rng.uniform(0.006, 0.018)
                width = rng.uniform(0.0008, 0.0016)
                angle = rng.uniform(0, np.pi)
                cx = rng.uniform(-sx / 2 + 0.004, sx / 2 - 0.004)
                cy = rng.uniform(-sy / 2 + 0.004, sy / 2 - 0.004)
                bright = bool(rng.random() < 0.4)
                m.geom_size[g, :2] = (length / 2, width / 2)
                m.geom_pos[g, :2] = (cx, cy)
                m.geom_quat[g] = (np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2))
                m.geom_rgba[g] = (0.85, 0.88, 0.92, 1.0) if bright else (0.08, 0.08, 0.10, 1.0)
                marks.append({"length_m": round(length, 4), "width_m": round(width, 5),
                              "angle_rad": round(angle, 3), "bright": bright})
            severity = {"count": n, "marks": marks}
        return severity

    # -- nuisance -------------------------------------------------------------------------------
    def apply_nuisance(self, rng: np.random.Generator, ranges: dict[str, Any]) -> dict[str, Any]:
        m = self.cell.model
        gain = rng.uniform(*ranges["light_gain"])
        m.light_diffuse[:] = self._nominal["light_diffuse"] * gain
        dl = rng.uniform(-ranges["light_pos_m"], ranges["light_pos_m"], (m.nlight, 2))
        m.light_pos[:, :2] = self._nominal["light_pos"][:, :2] + dl
        dc = rng.uniform(-ranges["cam_pos_m"], ranges["cam_pos_m"], 3)
        m.cam_pos[self.cam] = self._nominal["cam_pos"][self.cam] + dc
        m.cam_fovy[self.cam] = self._nominal["cam_fovy"][self.cam] + rng.uniform(-ranges["fovy_deg"], ranges["fovy_deg"])
        surf = []
        for g in self.surfaces:
            jitter = rng.uniform(-ranges["surface_jitter"], ranges["surface_jitter"], 3)
            m.geom_rgba[g, :3] = np.clip(self._nominal["geom_rgba"][g, :3] + jitter, 0, 1)
            surf.append([round(float(v), 4) for v in m.geom_rgba[g, :3]])
        return {"light_gain": round(float(gain), 4), "cam_offset_m": [round(float(v), 5) for v in dc],
                "fovy_deg": round(float(m.cam_fovy[self.cam]), 3), "surface_rgb": surf,
                "exposure": round(float(rng.uniform(*ranges["exposure"])), 4),
                "blur_sigma": round(float(rng.uniform(*ranges["blur_sigma"])), 3),
                "noise_std": round(float(rng.uniform(*ranges["noise_std"])), 3)}

    # -- render ---------------------------------------------------------------------------------
    def render(self, size: int = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
        return (self.cell.render("inspect", size, size), self.cell.render_geom_ids("inspect", size, size))

    def ghost_pixels(self, geom: int, size: int = IMAGE_SIZE) -> np.ndarray:
        """Pixels a hidden geom would cover (used for the mask of a missing hole)."""
        m = self.cell.model
        alpha = m.geom_rgba[geom, 3]
        m.geom_rgba[geom, 3] = 1.0
        ids = self.cell.render_geom_ids("inspect", size, size)
        m.geom_rgba[geom, 3] = alpha
        return ids == geom


def _post_process(img: np.ndarray, nuisance: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32) * nuisance["exposure"]
    if nuisance["blur_sigma"] > 0.05:
        pil = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
        out = np.asarray(pil.filter(ImageFilter.GaussianBlur(nuisance["blur_sigma"])), dtype=np.float32)
    out += rng.normal(0.0, nuisance["noise_std"], out.shape)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def plan_split(split: str, count: int, defect_fraction: float) -> list[str]:
    """Deterministic defect assignment: train is all good; others alternate defect types evenly."""
    if split == "train":
        return ["none"] * count
    n_defect = int(round(count * defect_fraction))
    kinds = [DEFECT_TYPES[i % len(DEFECT_TYPES)] for i in range(n_defect)] + ["none"] * (count - n_defect)
    order = np.random.default_rng([SPLIT_INDEX[split], count, 977]).permutation(len(kinds))
    return [kinds[i] for i in order]


def make_sample(renderer: DefectRenderer, split: str, index: int, defect: str, seed: int,
                out: Path | None) -> tuple[Sample, np.ndarray, np.ndarray]:
    entropy = [seed, SPLIT_INDEX[split], index]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    renderer.restore()
    cell = renderer.cell
    ranges = NUISANCE["shift" if split == "shift" else "nominal"]
    severity = renderer.apply_part(rng, defect)
    nuisance = renderer.apply_nuisance(rng, ranges)
    j = rng.uniform(-ranges["pos_jitter_m"], ranges["pos_jitter_m"], 2)
    yaw = float(rng.uniform(-np.pi, np.pi))
    xy = (INSPECT_PAD[0] + float(j[0]), INSPECT_PAD[1] + float(j[1]))
    cell.set_part_pose(xy, yaw)
    raw, ids = renderer.render()
    image = _post_process(raw, nuisance, rng)

    part_px = ids == renderer.part
    feature_px = part_px | np.isin(ids, renderer.holes + renderer.scratches)
    ys, xs = np.nonzero(feature_px)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if xs.size else [0, 0, 0, 0]
    hole_px = [int((ids == g).sum()) for g in renderer.holes]
    scratch_px = int(np.isin(ids, renderer.scratches).sum())
    if defect == "none":
        mask = np.zeros(ids.shape, bool)
    elif defect == "missing_hole":
        mask = renderer.ghost_pixels(renderer.holes[severity["hole"]])
    elif defect == "scratch":
        mask = np.isin(ids, renderer.scratches)
    else:  # wrong_colour, deformed_size: the whole part is out of spec
        mask = feature_px
    checks = {"hole_pixels": hole_px, "scratch_pixels": scratch_px, "part_pixels": int(feature_px.sum()),
              "part_in_view": bool(xs.size and bbox[0] > 0 and bbox[1] > 0
                                   and bbox[2] < ids.shape[1] - 1 and bbox[3] < ids.shape[0] - 1)}
    renderer.restore()

    label = "good" if defect == "none" else "defect"
    sid = f"{split}_{index:05d}"
    image_rel = f"images/{split}/{label}/{sid}.png"
    mask_rel = f"masks/{split}/defect/{sid}.png" if label == "defect" else None
    if out is not None:
        (out / image_rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(out / image_rel)
        if mask_rel:
            (out / mask_rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((mask * 255).astype(np.uint8)).save(out / mask_rel)
    sample = Sample(id=sid, split=split, index=index, seed_entropy=entropy, label=label, defect_type=defect,
                    severity=severity, part_pose={"x": round(xy[0], 5), "y": round(xy[1], 5), "yaw": round(yaw, 4)},
                    nuisance=nuisance, bbox_xyxy=bbox, defect_pixels=int(mask.sum()), gt_checks=checks,
                    image=image_rel, mask=mask_rel, tolerance=dict(TOLERANCE))
    return sample, image, mask


def check_ground_truth(sample: Sample) -> list[str]:
    """Rendered evidence must agree with the label. Returns a list of problems (empty = consistent)."""
    c, problems = sample.gt_checks, []
    if not c["part_in_view"]:
        problems.append("part not fully in view")
    holes = c["hole_pixels"]
    if sample.defect_type == "missing_hole":
        missing = sample.severity["hole"]
        if holes[missing] != 0:
            problems.append("missing hole is still drawn")
        if any(h < MIN_VISIBLE_HOLE_PX for i, h in enumerate(holes) if i != missing):
            problems.append("the remaining hole is not visible")
    elif any(h < MIN_VISIBLE_HOLE_PX for h in holes):
        problems.append("a nominal hole is not visible")
    if sample.defect_type == "scratch":
        if c["scratch_pixels"] < MIN_SCRATCH_PX:
            problems.append("scratch too small to see")
    elif c["scratch_pixels"]:
        problems.append("scratch pixels on a part without a scratch defect")
    if (sample.label == "defect") != (sample.defect_pixels > 0):
        problems.append("mask does not match the label")
    return problems


def generate(out: Path, seed: int, counts: dict[str, int], defect_fraction: float = 0.5) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    cell = build_cell()
    renderer = DefectRenderer(cell)
    summary: dict[str, Any] = {"generator": GENERATOR_VERSION, "seed": seed, "image_size": IMAGE_SIZE,
                               "camera": "inspect", "camera_nominal": INSPECT_CAM, "counts": {},
                               "defect_fraction": defect_fraction, "nuisance_ranges": NUISANCE,
                               "tolerance": TOLERANCE, "defect_types": DEFECT_TYPES}
    problems: list[str] = []
    try:
        with open(out / "manifest.jsonl", "w") as manifest:
            for split in SPLITS:
                plan = plan_split(split, counts.get(split, 0), defect_fraction)
                tally: dict[str, int] = {}
                for index, defect in enumerate(plan):
                    sample, _, _ = make_sample(renderer, split, index, defect, seed, out)
                    issues = check_ground_truth(sample)
                    problems += [f"{sample.id}: {p}" for p in issues]
                    manifest.write(json.dumps(asdict(sample)) + "\n")
                    tally[defect] = tally.get(defect, 0) + 1
                summary["counts"][split] = tally
    finally:
        cell.close()
    summary["manifest_sha256"] = hashlib.sha256((out / "manifest.jsonl").read_bytes()).hexdigest()
    summary["config_sha256"] = hashlib.sha256(json.dumps(
        {k: summary[k] for k in ("generator", "seed", "image_size", "defect_fraction", "nuisance_ranges",
                                 "tolerance", "counts")}, sort_keys=True).encode()).hexdigest()
    summary["ground_truth_problems"] = problems
    (out / "dataset.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the phase-3 synthetic defect dataset")
    parser.add_argument("--out", type=Path, default=Path("outputs/phase3/dataset"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", type=int, default=300, help="good images for one-class training")
    parser.add_argument("--val", type=int, default=120)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--shift", type=int, default=120)
    parser.add_argument("--defect-fraction", type=float, default=0.5)
    args = parser.parse_args(argv)
    summary = generate(args.out, args.seed, {"train": args.train, "val": args.val, "test": args.test,
                                             "shift": args.shift}, args.defect_fraction)
    print(json.dumps({k: summary[k] for k in ("counts", "manifest_sha256", "ground_truth_problems")}, indent=1))
    return 1 if summary["ground_truth_problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
