"""End-to-end: all co-simulation processes complete one mission cycle together."""
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPS = all(importlib.util.find_spec(m) for m in ("mujoco", "numpy", "zmq", "msgpack"))
MODULES = all((ROOT / "cosim" / f"{m}.py").exists() for m in ("world_sim", "navigator", "cell"))


@unittest.skipUnless(DEPS and MODULES, "needs mujoco/numpy/zmq and all cosim modules")
class CosimEndToEndTest(unittest.TestCase):
    def test_one_cycle_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-m", "cosim.launch", "--cycles", "1", "--rtf", "0",
                 "--no-dashboard", "--record", tmp, "--timeout", "240"],
                cwd=ROOT, capture_output=True, text=True, timeout=300,
            )
            logs = {p.name: p.read_text(errors="replace")[-2000:] for p in Path(tmp).glob("*.log")}
            self.assertEqual(result.returncode, 0, f"{result.stdout}\n{result.stderr}\n{json.dumps(logs, indent=1)}")
            self.assertTrue((Path(tmp) / "overview.mp4").exists(), "world_sim did not record")
            self.assertRegex(logs.get("cell.log", ""), re.compile(r"cycle", re.I))


if __name__ == "__main__":
    unittest.main()
