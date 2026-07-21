# notebooks/

Kaggle/local notebook scaffolding for running the Tier 0-4 pipeline.

`KR810_Tier0_Tier4_Kaggle_Template.ipynb` wraps `pipelines.run_tier0_to_tier4` (the same CLI
described in the root `README.md`) for execution on Kaggle or locally. It does not
re-implement any FK/Jacobian/DLS/evaluation algorithm -- every cell either calls into the
repository's `kinematics/`, `algorithms/`, `evaluation/`, or `pipelines/` packages, or reads and
displays the JSON/CSV output those packages already produced.

## Running on Kaggle

1. **Upload the project as a Kaggle Dataset.** Zip (or directly upload) the whole repository
   and create a new Kaggle Dataset from it. Any Dataset name/slug works -- the notebook does
   not hardcode one; it discovers the attached Dataset by looking for `DATASET_MANIFEST.json`
   (with `dataset_name == "kassow-kr810-trajectory-tier0-tier4"`), `assets/kr810.xml`, and
   `pipelines/run_tier0_to_tier4.py` under `/kaggle/input/*/`.
2. **Create a new Kaggle Notebook.**
3. **Attach the Dataset** to the notebook (Add Input -> your uploaded Dataset).
4. **Upload or open** `KR810_Tier0_Tier4_Kaggle_Template.ipynb` as the notebook's content.
5. **Select the CPU accelerator** (Settings -> Accelerator -> None/CPU). The pipeline never
   renders MuJoCo and never uses a GPU.
6. **Edit Cell 2** (`USER CONFIGURATION`) if you want anything other than the default smoke
   run: preset, seed, methods, trajectory/trial/point-sample/waypoint limits, plotting, and
   resume/overwrite behavior. See "Cell 2 configuration" below for what each variable maps to.
7. **Run All.** The notebook will copy the Dataset into `/kaggle/working` (input is read-only),
   check/install dependencies, validate the dataset, run the pipeline, display the Tier 0-4
   results, and package them.
8. **Download the ZIP** produced under `/kaggle/working/KR810_Tier0_Tier4_<RUN_NAME>.zip`
   (Kaggle's Output tab, or the notebook's own file browser). It contains
   `run_manifest.json`, `resolved_config.json`, `FINAL_SUMMARY.json`, every tier's CSV/JSON
   output, figures, and the run log -- not the source dataset, meshes, or a `.venv`.

## Running locally

Open the notebook from within the repository (e.g. `jupyter lab notebooks/`) with the project's
virtual environment as the kernel. Outside Kaggle, the notebook skips the copy-to-`/kaggle/working`
step and runs directly against the local project root; results are written to
`results/<RUN_NAME>/` instead of `/kaggle/working/kr810_results/<RUN_NAME>`.

## Preset: smoke vs full

- **`smoke`**: a small subset (few point samples, 2 trajectories, a handful of trials, limited
  waypoints). It exercises every tier end to end in well under a minute and is meant as a fast
  sanity check -- **not** the dataset's official research result.
- **`full`**: the entire benchmark (1200 point samples, all 8 trajectories, both methods, every
  trial). It is the configuration whose output should be treated as this dataset's result, but
  it is CPU- and time-intensive; a Kaggle CPU session may need several sessions/resume passes
  for a full run (`RESUME = True` in Cell 2 reuses valid, up-to-date tier output instead of
  recomputing it, mirroring the CLI's `--resume` flag).

## Cell 2 configuration -> CLI override mapping

Cell 2's variables mirror `pipelines.run_tier0_to_tier4`'s CLI flags one-to-one (see the root
`README.md`'s "CLI overrides" section for the authoritative flag list); Cell 9 turns them into
the equivalent `argv` list before Cell 11 invokes the pipeline:

| Cell 2 variable      | CLI flag               | Notes                                        |
|----------------------|-------------------------|-----------------------------------------------|
| `PRESET`              | `--preset`              | `"smoke"` or `"full"` (required)              |
| `SEED`                | `--seed`                |                                                 |
| `METHODS`             | `--methods`             | subset of `warm_start`, `cold_start`          |
| `TRAJECTORY_IDS`      | `--trajectory-ids`      | omitted when `None` (preset default)          |
| `TRIAL_CATEGORY`      | `--trial-category`      | `repeatability` \| `robustness` \| `all`      |
| `TRIAL_LIMIT`         | `--trial-limit`         | omitted when `None`                           |
| `POINT_SAMPLE_LIMIT`  | `--point-sample-limit`  | omitted when `None`                           |
| `WAYPOINT_LIMIT`      | `--waypoint-limit`      | omitted when `None`                           |
| `RUN_PLOTS = False`   | `--no-plots`            |                                                 |
| `OVERWRITE`           | `--overwrite`           | mutually exclusive with `RESUME`              |
| `RESUME`              | `--resume`              | mutually exclusive with `OVERWRITE`           |

`POSITION_THRESHOLD_M` and `ORIENTATION_THRESHOLD_DEG` mirror
`configs/evaluation_config.json`'s defaults for display/comparison in the result tables; the
pipeline has no CLI flag to override them, so they are not forwarded to `argv`.

## What the notebook does not do

It does not add Tier 5 dynamics, PPO, MPDIK, or MAPPO; does not train any model; does not
modify `assets/`; and does not regenerate the benchmark or trajectory data (those are
Stage 4 outputs, read-only inputs to this pipeline). A smoke run's numbers in the notebook are a
sanity check, not this dataset's official research result -- see the root `README.md`'s
"On acceptance thresholds" section.
