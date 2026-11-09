"""Milestone-1 smoke test: Anomalib PatchCore trained on good images only, exported to OpenVINO.

It answers one question from the workplan: does the train -> export -> CPU inference path run on
this machine (Apple Silicon, macOS arm64)? It is NOT a benchmark: the dataset is tiny and the
defects are deliberately obvious. Numbers it prints are a sanity check, not a result.

Steps
  1. Render overhead crops of the part on the inspection pad from the MuJoCo cell:
     good parts (pose jitter) and three defect types (wrong colour, oversize, extra hole).
  2. Train PatchCore (ResNet-18 backbone, CPU) on good images only.
  3. Export to OpenVINO IR and score every test image with the OpenVINO inferencer.
  4. Write outputs/phase3/anomalib_smoke/report.json.

Run with the vision environment (anomalib, openvino, torch):
    .venv-vision/bin/python -m phase3.tools.anomalib_smoke
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from phase3.scene import INSPECT_PAD, PART_SIZE, build_cell

OUT = Path("outputs/phase3/anomalib_smoke")
CROP = 128
DEFECTS = ("wrong_colour", "oversize", "extra_hole")


def render_dataset(root: Path, n_train: int, n_test: int, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    cell = build_cell()
    model = cell.model
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "part_geom")
    hole = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "part_hole")
    base_rgba = model.geom_rgba[gid].copy()
    base_size = model.geom_size[gid].copy()
    counts: dict[str, int] = {}

    def shot(path: Path, defect: str | None) -> None:
        model.geom_rgba[gid] = base_rgba
        model.geom_size[gid] = base_size
        model.geom_rgba[hole, 3] = 0.0
        if defect == "wrong_colour":
            model.geom_rgba[gid, :3] = (0.80, 0.45, 0.20)
        elif defect == "oversize":
            model.geom_size[gid, :2] = base_size[:2] * 1.35
        elif defect == "extra_hole":
            model.geom_rgba[hole, 3] = 1.0
        jitter = rng.normal(0.0, 0.004, size=2)
        cell.set_part_pose((INSPECT_PAD[0] + jitter[0], INSPECT_PAD[1] + jitter[1]),
                           yaw=float(rng.uniform(-0.3, 0.3)))
        img = cell.render("overhead", 320, 240)
        cy, cx = img.shape[0] // 2, img.shape[1] // 2
        crop = img[cy - CROP // 2:cy + CROP // 2, cx - CROP // 2:cx + CROP // 2]
        # mild sensor noise so the model does not key on perfectly identical pixels
        crop = np.clip(crop.astype(np.int16) + rng.normal(0, 2.0, crop.shape).astype(np.int16), 0, 255)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop.astype(np.uint8)).save(path)
        counts[str(path.parent.relative_to(root))] = counts.get(str(path.parent.relative_to(root)), 0) + 1

    for i in range(n_train):
        shot(root / "train_good" / f"{i:03d}.png", None)
    for i in range(n_test):
        shot(root / "test_good" / f"{i:03d}.png", None)
    for d in DEFECTS:
        for i in range(max(1, n_test // len(DEFECTS))):
            shot(root / "test_defect" / f"{d}_{i:03d}.png", d)
    cell.close()
    _ = PART_SIZE
    return counts


def auroc(scores_good: list[float], scores_bad: list[float]) -> float:
    """Probability that a random defective image scores higher than a random good one."""
    wins = sum((b > g) + 0.5 * (b == g) for b in scores_bad for g in scores_good)
    return wins / (len(scores_good) * len(scores_bad))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Anomalib + OpenVINO CPU smoke test")
    parser.add_argument("--train", type=int, default=40)
    parser.add_argument("--test", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    import torch
    from anomalib.data import Folder
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    import anomalib
    import openvino

    report: dict = {"ok": False, "platform": f"{platform.system()} {platform.machine()}",
                    "python": platform.python_version(), "anomalib": anomalib.__version__,
                    "openvino": openvino.__version__, "torch": torch.__version__, "steps": {}}
    data_root = OUT / "data"
    t0 = time.perf_counter()
    report["counts"] = render_dataset(data_root, args.train, args.test, args.seed)
    report["steps"]["render_s"] = round(time.perf_counter() - t0, 1)

    datamodule = Folder(name="so101_parts", root=data_root, normal_dir="train_good",
                        normal_test_dir="test_good", abnormal_dir="test_defect",
                        train_batch_size=8, eval_batch_size=8, num_workers=0, seed=args.seed)
    model = Patchcore(backbone="resnet18", layers=("layer2", "layer3"), coreset_sampling_ratio=0.1)
    engine = Engine(default_root_dir=str(OUT / "results"), accelerator="cpu", devices=1,
                    logger=False, enable_progress_bar=False)
    t0 = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    report["steps"]["train_s"] = round(time.perf_counter() - t0, 1)
    t0 = time.perf_counter()
    test_metrics = engine.test(model=model, datamodule=datamodule, verbose=False)
    report["steps"]["torch_test_s"] = round(time.perf_counter() - t0, 1)
    report["torch_test_metrics"] = {k: round(float(v), 4) for k, v in (test_metrics[0] if test_metrics else {}).items()}

    t0 = time.perf_counter()
    ir = engine.export(model=model, export_type=ExportType.OPENVINO, export_root=str(OUT / "export"))
    report["steps"]["export_s"] = round(time.perf_counter() - t0, 1)
    xml = next(Path(ir).rglob("*.xml")) if ir and Path(ir).is_dir() else Path(ir)
    report["openvino_ir"] = str(xml)

    # Anomalib 2.7's dynamo export freezes the batch dimension at the tracing batch size (seen
    # here: input [2,3,?,?]); reshaping the IR to batch 1 fails in an Interpolate node. So we call
    # OpenVINO directly and pad a single image up to the exported batch. The IR contains the
    # pre-processing (resize + normalise): input is float32 RGB in [0, 1], NCHW.
    core = openvino.Core()
    compiled = core.compile_model(core.read_model(xml), "CPU")
    in_shape = compiled.input(0).get_partial_shape()
    batch = in_shape[0].get_length() if in_shape[0].is_static else 1
    report["exported_input_shape"] = str(in_shape)
    report["outputs"] = [o.get_any_name() for o in compiled.outputs]

    def ov_score(rgb: np.ndarray) -> float:
        x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = compiled(np.repeat(x, batch, axis=0))
        return float(np.asarray(out[compiled.output("pred_score")]).reshape(-1)[0])

    scores: dict[str, list[float]] = {"good": [], "defect": []}
    per_defect: dict[str, list[float]] = {d: [] for d in DEFECTS}
    latencies = []
    images = sorted((data_root / "test_good").glob("*.png")) + sorted((data_root / "test_defect").glob("*.png"))
    ov_score(np.asarray(Image.open(images[0]).convert("RGB")))  # warm-up, not timed
    for path in images:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        t0 = time.perf_counter()
        score = ov_score(rgb)
        latencies.append((time.perf_counter() - t0) * 1000)
        if path.parent.name == "test_good":
            scores["good"].append(score)
        else:
            scores["defect"].append(score)
            per_defect[path.stem.rsplit("_", 1)[0]].append(score)
    report["openvino_scores"] = {
        "good_mean": round(float(np.mean(scores["good"])), 4),
        "defect_mean": round(float(np.mean(scores["defect"])), 4),
        "per_defect_mean": {d: round(float(np.mean(v)), 4) for d, v in per_defect.items() if v},
        "image_auroc_openvino": round(auroc(scores["good"], scores["defect"]), 4),
    }
    lat = np.array(latencies)
    report["openvino_cpu_latency_ms"] = {"n": int(lat.size), "median": round(float(np.median(lat)), 2),
                                         "p95": round(float(np.percentile(lat, 95)), 2)}
    report["ok"] = True
    report["note"] = ("Smoke test on a tiny synthetic set with obvious defects; proves the pipeline runs, "
                      "not detection quality.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
