"""Milestone-1 smoke test: Anomalib PatchCore trained on good images only, exported to OpenVINO.

It answers one question from the workplan: does the train -> export -> CPU inference path run on
this machine (Apple Silicon, macOS arm64)? It is NOT a benchmark: the dataset is tiny and the
defects are deliberately obvious. Numbers it prints are a sanity check, not a result.

Steps
  1. Render inspection-camera images with the milestone-2 generator (phase3.dataset):
     good parts and three defect types (wrong colour, deformed size, missing hole).
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

import numpy as np
from PIL import Image

from phase3.dataset import DefectRenderer, make_sample
from phase3.scene import build_cell

OUT = Path("outputs/phase3/anomalib_smoke")
DEFECTS = ("wrong_colour", "deformed_size", "missing_hole")


def render_dataset(root: Path, n_train: int, n_test: int, seed: int) -> dict[str, int]:
    """Render with the milestone-2 generator (inspect camera, nominal randomisation)."""
    cell = build_cell()
    renderer = DefectRenderer(cell)
    counts: dict[str, int] = {}

    def shot(folder: str, name: str, split: str, index: int, defect: str) -> None:
        _, image, _ = make_sample(renderer, split, index, defect, seed, None)
        path = root / folder / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(path)
        counts[folder] = counts.get(folder, 0) + 1

    for i in range(n_train):
        shot("train_good", f"{i:03d}", "train", i, "none")
    for i in range(n_test):
        shot("test_good", f"{i:03d}", "test", i, "none")
    for k, d in enumerate(DEFECTS):
        for i in range(max(1, n_test // len(DEFECTS))):
            shot("test_defect", f"{d}_{i:03d}", "test", 1000 + 100 * k + i, d)
    cell.close()
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
