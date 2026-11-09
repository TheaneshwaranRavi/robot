# Phase 3 licence manifest

Third-party software and assets keep their upstream licences. The repository itself has no LICENSE file yet; add one before redistributing.

| Item | Source and pinned revision | Licence | Repository policy |
| --- | --- | --- | --- |
| SO-101 MJCF and STL model | MuJoCo Menagerie `robotstudio_so101`, commit `98a70c569c4c1b6da467a20012f766ee1f8dc521` | Apache-2.0 | Fetched by `python -m phase3.tools.fetch_so101` into `phase3/assets/so101/` (gitignored) with its LICENSE and a SHA-256 manifest. Not committed. |
| Anomalib | `anomalib[openvino]==2.7.0` (PyPI) | Apache-2.0 | Installed from `requirements-phase3.txt` into `.venv-vision`. |
| OpenVINO | `openvino==2026.4.1` (PyPI, pulled in by the extra) | Apache-2.0 | Installed dependency. |
| ResNet-18 ImageNet weights | Downloaded by Anomalib/timm on first PatchCore training | See the timm / torchvision model card | Cached outside the repository; not committed. |
