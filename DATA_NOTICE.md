# REAL275 sample data

RGB and depth images originate from the REAL test split of the NOCS dataset,
by He Wang, Srinath Sridhar, Jingwei Huang, Julien Valentin, Shuran Song and
Leonidas J. Guibas (CVPR 2019).

Source and dataset conditions: https://github.com/hughw19/NOCS_CVPR2019#datasets
The authors specify non-commercial use and require citation of their paper.
These data are not covered by this repository's Apache-2.0 code license.

The sample archive contains three evenly spaced available frames per scene
from all six scenes. The manifest records exact frame IDs and per-file SHA-256
checksums. Object masks and boxes are cached detector predictions from the
SCOPE research evaluation JSON, not ground-truth annotations. The manifest
also records the SHA-256 of that source JSON. No ground-truth pose, NOCS, or
CAD models are used as model inputs or to construct the predicted overlay.

Citation:

Wang et al., “Normalized Object Coordinate Space for Category-Level 6D Object
Pose and Size Estimation,” IEEE/CVF CVPR, 2019.

The Open3D previews and GIFs under `docs/` show these REAL275 samples with
predicted overlays. The paper overview figures (`docs/paper-conditioning.png`
and `docs/paper-architecture.png`) are cropped from Figures 1 and 2 of the
published SCOPE article, © 2026 Hönig et al., licensed under CC BY 4.0.
Article: https://doi.org/10.1016/j.imavis.2026.106145
License: https://creativecommons.org/licenses/by/4.0/
