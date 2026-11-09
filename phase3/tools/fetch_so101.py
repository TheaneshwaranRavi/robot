"""Fetch the SO-101 MJCF model from MuJoCo Menagerie at a pinned commit.

The model (robotstudio_so101, Apache-2.0) is ~18 MB of STL meshes, so it is not committed.
It is downloaded into phase3/assets/so101/ (gitignored) together with its LICENSE.

Run:  python -m phase3.tools.fetch_so101 [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
COMMIT = "98a70c569c4c1b6da467a20012f766ee1f8dc521"
SUBDIR = "robotstudio_so101"
DEST = Path(__file__).resolve().parents[1] / "assets" / "so101"
MANIFEST = "fetch_manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(dest: Path = DEST, force: bool = False) -> Path:
    if (dest / "so101.xml").exists() and (dest / MANIFEST).exists() and not force:
        return dest
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "menagerie"
        run = lambda *args, cwd=None: subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)
        run("git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", REPO, str(clone))
        run("git", "sparse-checkout", "set", SUBDIR, cwd=clone)
        run("git", "checkout", "--quiet", COMMIT, cwd=clone)
        src = clone / SUBDIR
        if not (src / "so101.xml").exists():
            raise RuntimeError(f"{SUBDIR}/so101.xml missing at {COMMIT}")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    files = {str(p.relative_to(dest)): _sha256(p) for p in sorted(dest.rglob("*")) if p.is_file()}
    (dest / MANIFEST).write_text(json.dumps(
        {"source": REPO, "commit": COMMIT, "subdir": SUBDIR, "license": "Apache-2.0", "files": files},
        indent=1))
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)
    path = fetch(force=args.force)
    print(f"SO-101 model ready at {path} (menagerie {COMMIT[:10]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
