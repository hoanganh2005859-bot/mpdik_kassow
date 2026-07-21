# Asset Conversion Report — Stage 2

## Source

- URDF: `kr2_robot_a810_clean.urdf` (plain-text, expanded from xacro), external local asset, copied unmodified to `assets/kr810.urdf`.
- Meshes: 8 STL files referenced by the URDF, copied unmodified from the external mesh source to `assets/meshes/a810/`. The 9th mesh present at the source (`a810_Flange.stl`/`.obj`) is not referenced by any URDF link and was not copied. No OBJ files were copied since neither the URDF nor the MJCF reference them.
- SHA256 of source and destination URDF match exactly. SHA256 of each source and destination STL match exactly (see `model_metadata.json`).
- The legacy `kr810/` folder in this repository was not read, modified, or used as a source for this stage.

## Conversion method

MuJoCo's built-in URDF importer (Path A/B in the stage plan):

1. `assets/kr810.urdf` loaded directly via `mujoco.MjModel.from_xml_path` — succeeded with nq=7, nv=7, nu=0.
2. Canonical MJCF exported via `mujoco.mj_saveLastXML`.
3. Minimal manual edits applied to the exported file:
   - `compiler meshdir` set to `meshes/a810`; per-mesh `file` attributes shortened to basenames (all paths relative, no absolute path).
   - `ee_site` added to the `end_effector` body at local origin.
4. Saved as `assets/kr810.xml`, verified to load and run `mj_forward` independently of the URDF.

No external conversion package was used; no actuators were added.

## Joint mapping

Movable joints, in order: `joint_1` … `joint_7` (all revolute), matching the URDF joint names and order exactly. `nq = nv = 7`, `nu = 0`.

## Fixed-joint mapping

URDF fixed joints `dummy`, `jointPL2`, `jointPL4`, `jointPL6` connect the `world`/`base`/`link2`/`link4`/`link6` links rigidly to their neighbors. MuJoCo's built-in compiler fuses these jointless child bodies into their nearest jointed parent body during URDF-to-MJCF compilation (a standard MuJoCo optimization), composing geometry offsets and inertial properties via rigid transform. Consequently these fixed links/joints are not present as separate named bodies in `kr810.xml`; the 7-DOF kinematic chain, joint order, and joint transforms are unaffected. This is recorded as a known limitation in `model_metadata.json`.

## Mesh mapping

All 8 meshes are referenced from `assets/kr810.xml` via `compiler meshdir="meshes/a810"` plus basename `file` attributes, resolving to `assets/meshes/a810/*.stl`.

## Operational-limit policy

Joint limits are read directly from the source URDF `<limit>` elements. `joint_2` and `joint_4` have asymmetric revolute limits, used as-is. `joint_1`, `joint_3`, `joint_5`, `joint_6`, `joint_7` encode a full ±2π revolute range in the URDF (a continuous-joint convention expressed as a revolute joint). These ±2π values are **not** claimed as verified mechanical hard stops; they are recorded as `operational_lower_rad` / `operational_upper_rad` for use in sampling and DLS, per project plan. Velocity and effort limits are likewise taken directly from the URDF and recorded in metadata/config since MJCF has no native per-joint velocity-limit attribute.

## ee_site definition

`ee_site` is placed at the origin of the URDF `end_effector` link (`pos="0 0 0"`, `quat="1 0 0 0"`). It represents the **KR810 end-effector link reference frame only**. It is explicitly **not** an official TCP, tool tip, or flange calibration frame — no such calibrated frame exists in the source data.

## MuJoCo validation performed

- `assets/kr810.xml` loads independently via `mujoco.MjModel.from_xml_path`.
- `nq = 7`, `nv = 7`, `nu = 0`, `njnt = 7`, movable joint order is `joint_1`…`joint_7`.
- `mj_forward` runs successfully with `qpos = 0`.
- `ee_site` resolves to the `end_effector` body; its position and orientation matrix are finite at the zero configuration.
- No absolute filesystem paths (`C:\`, `D:\`, `/home/`, `/d/`) appear in `kr810.xml`.

This is asset/model validation only. No Tier 0 kinematic accuracy validation, benchmark, or trajectory validation is claimed to be complete as part of this stage.

## Known limitations

See `known_limitations` in `assets/model_metadata.json`: fused fixed bodies, no native MJCF velocity-limit field, ±2π operational (not hard-stop) limits on 5 joints, placeholder inertia tensors in the source URDF, and `ee_site` being a link reference frame rather than a calibrated tool frame.
