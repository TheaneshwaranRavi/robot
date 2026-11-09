"""Phase 3 milestone 2: synthetic defect dataset - labels match renders, seeds reproduce, splits isolate."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "phase3" / "assets" / "so101" / "so101.xml"
HAVE = importlib.util.find_spec("mujoco") is not None and MODEL.exists()


@unittest.skipUnless(HAVE, "needs mujoco and the SO-101 model (python -m phase3.tools.fetch_so101)")
class DefectDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from phase3 import dataset, scene
        cls.np, cls.ds, cls.scene = np, dataset, scene
        cls.cell = scene.build_cell()
        cls.renderer = dataset.DefectRenderer(cls.cell)

    @classmethod
    def tearDownClass(cls):
        cls.cell.close()

    def sample(self, defect, split="test", index=0, seed=0):
        return self.ds.make_sample(self.renderer, split, index, defect, seed, None)

    def test_every_defect_type_matches_its_render(self):
        for split in ("test", "shift"):
            for k, defect in enumerate(("none",) + self.ds.DEFECT_TYPES):
                for index in range(4):
                    s, image, mask = self.sample(defect, split, 10 * k + index)
                    self.assertEqual(self.ds.check_ground_truth(s), [], f"{split} {defect} {index}")
                    self.assertEqual(image.shape, (256, 256, 3))
                    self.assertEqual(s.label == "defect", bool(mask.any()))

    def test_defect_masks_are_localised(self):
        s, _, mask = self.sample("missing_hole", index=3)
        area = self.np.pi * (self.scene.HOLE_RADIUS / 0.0006) ** 2  # ~0.6 mm per pixel
        self.assertGreater(mask.sum(), 0.4 * area)
        self.assertLess(mask.sum(), 2.5 * area)
        s, _, mask = self.sample("scratch", index=5)
        self.assertEqual(int(mask.sum()), s.gt_checks["scratch_pixels"])
        s, _, mask = self.sample("wrong_colour", index=7)
        self.assertEqual(int(mask.sum()), s.gt_checks["part_pixels"])

    def test_defects_change_the_measured_part(self):
        good, gimg, _ = self.sample("none", index=11)
        part = self.cell.model.geom_rgba[self.renderer.part, :3]
        self.assertTrue(self.np.allclose(part, self.scene.PART_RGBA[:3]), "renderer must restore the model")
        big = [self.sample("deformed_size", index=i)[0] for i in range(6)]
        ratios = [s.gt_checks["part_pixels"] / good.gt_checks["part_pixels"] for s in big]
        self.assertTrue(all(abs(r - 1) > 0.08 for r in ratios), ratios)
        colour = self.sample("wrong_colour", index=2)[0]
        self.assertGreater(colour.severity["delta_from_nominal"], 0.15)

    def test_same_seed_reproduces_and_different_index_differs(self):
        a, ia, ma = self.sample("scratch", index=4, seed=3)
        b, ib, mb = self.sample("scratch", index=4, seed=3)
        c, ic, _ = self.sample("scratch", index=5, seed=3)
        self.assertEqual(a, b)
        self.assertTrue((ia == ib).all() and (ma == mb).all())
        self.assertFalse((ia == ic).all())
        d, _, _ = self.sample("scratch", index=4, seed=4)
        self.assertNotEqual(a.part_pose, d.part_pose)

    def test_split_plans(self):
        self.assertEqual(set(self.ds.plan_split("train", 50, 0.5)), {"none"})
        test = self.ds.plan_split("test", 40, 0.5)
        self.assertEqual(test.count("none"), 20)
        for d in self.ds.DEFECT_TYPES:
            self.assertEqual(test.count(d), 5)
        self.assertEqual(test, self.ds.plan_split("test", 40, 0.5))

    def test_generate_is_reproducible_and_splits_are_isolated(self):
        counts = {"train": 6, "val": 8, "test": 8, "shift": 4}
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            s1 = self.ds.generate(Path(t1), 5, counts)
            s2 = self.ds.generate(Path(t2), 5, counts)
            self.assertEqual(s1["ground_truth_problems"], [])
            self.assertEqual(s1["manifest_sha256"], s2["manifest_sha256"])
            for rel in ("images/test/defect", "images/train/good"):
                for f in sorted((Path(t1) / rel).glob("*.png")):
                    self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),
                                     hashlib.sha256((Path(t2) / rel / f.name).read_bytes()).hexdigest())
            records = [json.loads(line) for line in (Path(t1) / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), sum(counts.values()))
            self.assertTrue(all(r["label"] == "good" for r in records if r["split"] == "train"))
            seeds = [tuple(r["seed_entropy"]) for r in records]
            self.assertEqual(len(seeds), len(set(seeds)), "no two samples share a seed")
            for r in records:
                self.assertTrue((Path(t1) / r["image"]).exists())
                self.assertEqual(r["label"] == "defect", r["mask"] is not None)
                self.assertNotIn(r["defect_type"], r["image"].replace("defect/", ""), "no defect type in names")


if __name__ == "__main__":
    unittest.main()
