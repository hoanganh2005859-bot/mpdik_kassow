# trajectories/

Trajectory definitions and trial manifests for Tier 2-4.

- `trajectory_manifest.csv` - one row per generated trajectory (type, path,
  waypoint count, duration, orientation mode, closed/open path, generation
  status).
- `trajectory_trials.csv` - one row per trial run against a trajectory
  (trial category, repeat id, seed, speed scale, control period, initial
  joint configuration).
- `line/`, `circle/`, `figure8/`, `helix/` - per-type generated trajectory
  data (populated in a later stage; empty in this scaffold).
- `custom/custom_trajectory_template.csv` - header-only template for
  hand-authored custom trajectories.

No trajectory data has been generated in this stage; only headers/templates
and this directory structure exist so far.
