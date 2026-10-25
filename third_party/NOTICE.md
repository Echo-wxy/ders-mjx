# Third-party code included in this repository

| Directory | Upstream | Commit | License |
|---|---|---|---|
| `mujoco_mjx_diffmjx/` | MuJoCo XLA (MJX) from the `diffmjx` branch of https://github.com/martius-lab/mujoco (a fork of https://github.com/google-deepmind/mujoco); provides Contacts From Distance (Paulus et al., ICLR 2026) | ced2cffdc4a8c62056dce596b6df53e5fdc59ac7 | Apache-2.0 (`mujoco_mjx_diffmjx/LICENSE`) |
| `softjax/` | https://github.com/a-paulus/softjax (imported by the fork) | 82fa3c908bc5f7378ee044f0171cbe8560d52f61 | Apache-2.0 (`softjax/LICENSE`) |

Both are included unmodified, except that test data, integration tests and unit
tests were removed from the MJX copy to keep the archive small. All changes made
for this work are applied at run time by `ders/soft_contact.py` (documented
there) and are not edits of these files.
