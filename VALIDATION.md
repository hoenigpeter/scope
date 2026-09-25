# Release validation

Tested on 2026-09-25 on Linux x86_64 with an NVIDIA GeForce RTX 3090.

- Created a new project-local conda environment with `setup.sh`, installed
  dependencies and built TEASER++ from the pinned upstream commit. The tested
  compiler compatibility fix (`std::vector`) is included in the build script.
- Dependency isolation was tested with ROS Jazzy paths present in the calling
  shell. The scripts remove inherited Python paths and disable user-site imports.
- Finalized full conda and pip locks from this environment; reran the locked
  installer successfully. `pip check` reported no broken requirements.
- All ten tests passed, covering geometry, registration with outliers, RLE
  decoding, asset checksums, preservation of existing files on validation failure,
  archive path rejection, offline reuse and reference mismatch detection.
- Offline GPU inference with 10 diffusion steps and base seed 0 processed all
  **18 prepared frames / 99 detections**, with **zero registration failures**.
- The 99 object inference/registration calls took 40.7 seconds total (median
  0.40 s/object), excluding model loading, checksums and PLY export. This is a
  local smoke-run measurement, not a controlled paper runtime benchmark.
- Repeated the first frame: all five NOCS PNGs and all R/t/scale values were
  exactly identical on this environment and GPU.
- Open3D created a window and rendered saved scene, overlay and axes geometries
  under Xvfb in the fresh environment. The resulting image was visually inspected
  and is included as `docs/preview.png`.
- Extracted the source-only archive outside the research repository, installed
  both assets through the checksum-verifying fetch script, and ran the first
  frame successfully from `/tmp` with the fresh environment: five poses, no failures.
- A full repeat exposed a 0.055-degree difference in one pose caused by parallel
  maximum-clique tie-breaking in TEASER++. Registration now uses one OpenMP
  thread. Two full runs with this setting produced exactly identical R/t/scale
  values for all 99 detections. The released reference uses this setting.
- The source-only public-download workflow installed both hosted archives and
  processed all 18 frames. Reusing those installed assets required no downloads;
  `--verify-reference` passed for all 99 poses in the subsequent run.
- Python compilation and Bash syntax checks passed.

The demo does not measure ground-truth pose error or paper mAP. Registration
success means a finite transform was fitted, not that every pose is correct.
Different library versions or hardware can change floating-point results.

The conda bootstrap branch (installing Miniforge on a machine without conda)
and CPU inference were not exercised in this validation. The bootstrap uses a
versioned installer and verifies its upstream SHA-256. The public ownCloud model and data archives were downloaded and matched
the expected archive SHA-256 checksums; automatic installation was also tested
from a source-only copy outside the research repository. Interactive mouse/key usability should be tried on the
publication author's desktop; automated rendering does not exercise those inputs.

README visuals: Figures 1 and 2 were cropped from the published PDF and
visually checked for complete labels. Three Open3D GIFs (scenes 1, 3 and 6)
were rendered from saved predictions at 640×480 with 48 frames each. The
animations, looping metadata, pure white backgrounds and README paths were
checked. The interactive viewer and refreshed static preview use white too.

Repository relocation check: rebuilt the locked environment in the cloned
repository, passed dependency checks and all 10 tests, and matched all 99
reference poses across 18 frames. The launcher was tested from another working
directory. Open3D rendering, white backgrounds, README links, GIFs, asset hashes
and public download URLs were checked. Git-visible files match the source
package; environments, datasets, weights and generated outputs remain excluded.
