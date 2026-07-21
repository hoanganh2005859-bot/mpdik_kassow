# Đặc tả Full DLS Evaluation (Tier 0 → Tier 4)

Tài liệu này mô tả chính xác những gì **Full DLS Evaluation** hiện tại (nhánh
`main`, các commit tới `51b552a`) thực sự đánh giá, dựa trên việc đọc trực
tiếp code trong `pipelines/`, `algorithms/`, `evaluation/`, `kinematics/`,
`generators/`, `configs/`, `schemas/`, dữ liệu thật trong `benchmarks/` và
`trajectories/`, cùng `notebooks/KR810_Tier0_Tier4_Kaggle_Template.ipynb`.
Không có Full run nào đã được thực thi để tạo tài liệu này — mọi con số về
dữ liệu (số sample, số trajectory, số trial, số row) đều lấy trực tiếp từ
dataset đã sinh sẵn trong repo; mọi con số về workload là **tính toán từ
định nghĩa trial/waypoint trong code**, không phải kết quả đo thực tế.

---

## A. Tóm tắt điều hành

**Full DLS Evaluation là gì.** Đây là một lần chạy `pipelines.run_tier0_to_tier4`
với `--preset full` và không có bất kỳ cờ giới hạn nào (`--point-sample-limit`,
`--trial-limit`, `--waypoint-limit`, `--trajectory-ids` đều để trống), sử
dụng toàn bộ 1200 sample Point-IK, toàn bộ 8 trajectory, toàn bộ 360 trial,
toàn bộ 400 waypoint/trajectory, và cả hai phương pháp Tier 2
(`warm_start`, `cold_start`). Nó chạy tuần tự Tier 0 → Tier 1 → Tier 2 →
Tier 3 → Tier 4 (`pipelines/run_tier0_to_tier4.py::main`).

**Khác Smoke Evaluation như thế nào.** Smoke (`--preset smoke`,
`configs/experiment_presets.json::smoke`) chỉ dùng 30 point sample, 2
trajectory (`line_fixed_orientation`, `circle_fixed_orientation`), tối đa 4
trial, tối đa 40 waypoint, và số sample validation Tier 0 rút gọn (FK 20,
Jacobian 8, singularity 20). README (`README.md` dòng 79-93, 181-182) và
`notebooks/README.md` (dòng 41-50) đều khẳng định: Smoke là kiểm tra nhanh
để phát triển cục bộ, **không phải kết quả nghiên cứu chính thức** của
dataset này. Full mới là cấu hình mà kết quả được xem là "official research
result".

**Mục tiêu nghiên cứu của Full DLS.** Đo hiệu năng thật, toàn diện của bộ
giải Damped-Least-Squares (DLS) hiện tại trên toàn bộ benchmark Point-IK và
toàn bộ trajectory/trial đã sinh, ở cả hai chế độ warm-start (liên tục theo
chuỗi waypoint) và cold-start (baseline không liên tục), để: (1) xác lập một
baseline kinematic-IK đáng tin cậy, (2) làm cơ sở so sánh công bằng sau này
với MPDIK/PPO/MAPPO (Tier 5/6, hiện **không** nằm trong repo).

**Tại sao phải chạy Full DLS trước MPDIK.** Tier 0 là gate kiểm tra FK/Jacobian
đúng trước khi tin bất kỳ solver nào (`README.md` dòng 16-18); Tier 1-4 là
chuỗi bằng chứng tăng dần độ khó (point → sequential → tracking → smoothness)
cho baseline DLS. Nếu baseline DLS này chưa được đánh giá đầy đủ (full, không
phải smoke) thì không có cơ sở định lượng nào để so sánh MPDIK "tốt hơn" hay
"tệ hơn" DLS — đây là lý do dự án yêu cầu Full DLS Evaluation phải hoàn tất
trước khi bất kỳ phương pháp học nào được đưa vào so sánh.

**Đây là kinematic evaluation, không phải dynamic evaluation.** Xác nhận
trực tiếp:
- `README.md` dòng 184-186: "Tier 0-4 evaluates kinematics only... it does
  not include dynamics or a controller."
- `assets/model_metadata.json::known_limitations` mục cuối: "No MuJoCo
  actuators are defined (nu=0); Tier 0-4 does not require actuation."
- `DATASET_MANIFEST.json::scope`: `includes_dynamic_control: false`,
  `includes_ppo: false`, `includes_mpdik: false`, `includes_mappo: false`.
- Toàn bộ Tier 0-4 chỉ gọi FK/Jacobian/DLS kinematic (`kinematics/`,
  `algorithms/*_dls.py`) — không có torque, không có bộ điều khiển, không có
  bước tích phân động lực học nào trong pipeline.

**Kết luận được phép rút ra sau khi có Full result:** độ chính xác vị trí/hướng
của DLS trên benchmark Point-IK và trajectory theo từng difficulty/loại
trajectory/tốc độ; mức độ warm-start cải thiện continuity so với cold-start;
mức độ mượt/khả thi động học (không phải động lực học) của quỹ đạo khớp;
DLS có đạt các ngưỡng chấp nhận **do dự án tự định nghĩa** hay không.

**Kết luận KHÔNG được phép rút ra:** bất kỳ tuyên bố nào về độ chính xác vật
lý thật của robot (chưa có dynamics/controller); bất kỳ tuyên bố "đạt chuẩn
ISO 9283" (metric chỉ *lấy cảm hứng từ* ISO 9283 — xem mục G, I); bất kỳ so
sánh với MPDIK/PPO/MAPPO (chưa tồn tại trong repo); bất kỳ tuyên bố về độ
chính xác TCP đã hiệu chuẩn (xem điểm tiếp theo); và không được suy diễn kết
quả Full khi chưa thực sự chạy Full.

**`ee_site` là gì.** `ee_site` là một site MuJoCo đặt tại gốc cục bộ của link
`end_effector` (`assets/model_metadata.json::end_effector_definition`;
`assets/ASSET_CONVERSION_REPORT.md` mục "ee_site definition"): *"KR810
end-effector link reference frame. This is NOT an officially calibrated TCP,
tool tip, or flange calibration frame."* Mọi số liệu vị trí/hướng trong Tier
1-4 đều đo tại `ee_site`, không phải một TCP hiệu chuẩn thực tế.

**Chưa có gì trong kết quả này:** actuator dynamics, torque/controller
dynamics, MPDIK, PPO, MAPPO — không tồn tại trong bất kỳ đường dẫn code nào
của `algorithms/`, `evaluation/`, `pipelines/` hiện tại.

---

## B. Lệnh chạy Full chính xác

Điều kiện để một run được coi là "Full" (suy từ
`pipelines/run_tier0_to_tier4.py::build_arg_parser` và
`pipelines/_common.py::build_resolved_config`):

1. `--preset full` (bắt buộc).
2. **Không** truyền bất kỳ cờ giới hạn nào (`--point-sample-limit`,
   `--trial-limit`, `--waypoint-limit`, `--trajectory-ids`) vì CLI có độ ưu
   tiên cao hơn preset (`pipelines/_common.py::build_resolved_config`, hàm
   `resolved(cli_key, preset_key, default)`) — nếu một cờ giới hạn được
   truyền, nó sẽ ghi đè giá trị `"all"` của preset `full` và thu hẹp phạm vi
   dù `--preset` vẫn là `full`.
3. `--trial-category all` (giá trị mặc định của CLI khi không truyền —
   `configs/experiment_presets.json` không định nghĩa `trial_category`, nên
   giá trị mặc định `all` từ chính CLI/`_common.py` áp dụng), để bao gồm cả
   360 trial (`repeatability` **và** `robustness`), không chỉ một loại.
4. `--methods warm_start,cold_start` (mặc định), để cả hai phương pháp Tier 2
   đều chạy — chỉ khi cả hai đều có mặt thì `warm_vs_cold.csv` mới có ý
   nghĩa so sánh.
5. Dưới preset `full`
   (`configs/experiment_presets.json::full`), mọi trường giới hạn có giá trị
   chuỗi `"all"`: `point_sample_limit`, `validation_fk_samples`,
   `validation_jacobian_samples`, `validation_singularity_samples`,
   `selected_trajectories`, `trajectory_trial_limit`, `waypoint_limit`.
   `pipelines/_common.py::_as_optional_int` / `_as_optional_str_list` map cả
   `"all"` lẫn `None` thành `None` (không giới hạn) — nghĩa là toàn bộ 1200
   point sample, toàn bộ 1000/200/300 sample validation Tier 0, toàn bộ 8
   trajectory, toàn bộ 360 trial, toàn bộ 400 waypoint/trajectory được dùng.

### 1. Windows PowerShell

```powershell
.venv\Scripts\python.exe -m pipelines.run_tier0_to_tier4 `
  --preset full `
  --output results/full_run `
  --seed 42 `
  --methods warm_start,cold_start `
  --trial-category all
```

(Không truyền `--point-sample-limit`, `--trial-limit`, `--waypoint-limit`,
`--trajectory-ids` — để chúng ở trạng thái mặc định `None` cho một run thật
sự "Full". Thêm `--overwrite` nếu `results/full_run` đã tồn tại, hoặc
`--resume` để tiếp tục một Full run dở dang.)

### 2. Kaggle / Linux

```bash
python -m pipelines.run_tier0_to_tier4 \
  --preset full \
  --output results/full_run \
  --seed 42 \
  --methods warm_start,cold_start \
  --trial-category all
```

### 3. Notebook Cell 2 (`notebooks/KR810_Tier0_Tier4_Kaggle_Template.ipynb`)

Cell 2 (`USER CONFIGURATION`, tế bào code đầu tiên) cần được chỉnh thành:

```python
PRESET = "full"
SEED = 42
METHODS = ["warm_start", "cold_start"]
TRAJECTORY_IDS = None       # None = dùng toàn bộ trajectory của preset full
TRIAL_CATEGORY = "all"      # "repeatability" | "robustness" | "all"
TRIAL_LIMIT = None
POINT_SAMPLE_LIMIT = None
WAYPOINT_LIMIT = None
RUN_PLOTS = True
RESUME = False               # đặt True nếu tiếp tục một Full run dở dang
OVERWRITE = True
RUN_NAME = "kr810_tier0_tier4_full"
```

Cell 8/9 (chuyển Cell 2 thành `argv`) và Cell 10 (gọi
`subprocess.run([sys.executable, "-m", "pipelines.run_tier0_to_tier4", *argv])`)
không cần chỉnh sửa gì thêm — bảng ánh xạ đầy đủ nằm ở
`notebooks/README.md` mục "Cell 2 configuration -> CLI override mapping".
Lưu ý: `RUN_NAME` là biến notebook-only (không phải cờ CLI của
`pipelines.run_tier0_to_tier4` — repo hiện **không có** cờ `--run-name`); nó
chỉ dùng để đặt tên thư mục/ZIP đầu ra.

### Ghi chú quan trọng

- Không có cờ `--run-name` trên CLI (`python -m pipelines.run_tier0_to_tier4
  --help` chỉ liệt kê các cờ ở mục CLI overrides của `README.md`, xác nhận
  qua `pipelines/run_tier0_to_tier4.py::build_arg_parser`); tên thư mục
  output do `--output` quyết định trực tiếp.
- Không có `-fk-samples`/`--jacobian-samples`/`--singularity-samples` trên
  `run_tier0_to_tier4`; số lượng sample validation Tier 0 dưới preset `full`
  luôn là toàn bộ (1000/200/300), lấy qua
  `configs/experiment_presets.json::full::validation_*_samples = "all"`.
- Nếu `--output` đã tồn tại và không có `--overwrite` hoặc `--resume`, run bị
  từ chối (không ghi đè âm thầm) — `pipelines/run_tier0_to_tier4.py::main`.

---

## C. Dữ liệu Full Evaluation

### 1. Point-IK benchmark (`benchmarks/point_ik/`)

- Tổng số sample: **1200** (`benchmarks/point_ik/point_ik_checksum.json`,
  `DATASET_MANIFEST.json::generation_summary.point_ik_sample_count`).
- 6 difficulty group, 200 sample/group
  (`configs/benchmark_config.json::point_ik_samples_per_group = 200`,
  `benchmarks/point_ik/difficulty_definition.json::sample_counts`):
  `near_target`, `medium_target`, `far_target`, `large_orientation_change`,
  `near_joint_limit`, `near_singularity` (`difficulty_id` 0-5).
- Array chính trong `point_ik_v1.npz` (`benchmarks/point_ik/point_ik_checksum.json`):
  `sample_id [1200]`, `q_initial`/`q_target [1200,7]`, `initial_position`/
  `target_position [1200,3]`, `initial_quaternion`/`target_quaternion [1200,4]`,
  `position_distance_m`/`orientation_distance_rad`/`joint_distance_rad
  [1200]`, `initial_sigma_min`/`target_sigma_min [1200]`,
  `minimum_initial_limit_margin`/`minimum_target_limit_margin [1200]`,
  `difficulty_id [1200]` (int32), `source_seed [1200]`.
- Target được sinh như thế nào: mỗi sample lấy một `q_target` hợp lệ (nằm
  trong operational limit), chạy forward kinematics để suy ra
  `target_position`/`target_quaternion` — không bao giờ chọn một điểm
  Cartesian tự do rồi giả định là reachable
  (`generators/generate_point_ik_dataset.py`, mô tả tại
  `benchmarks/README.md` dòng 45-48). Ngưỡng phân nhóm difficulty là **quantile
  thực nghiệm** (không phải số tùy ý) tính từ một pool 30.000 cặp `(q_initial,
  q_target)` (`benchmarks/point_ik/difficulty_definition.json::quantile_thresholds`,
  `generic_pool_size: 30000`), với thứ tự ưu tiên khi một cặp thỏa nhiều
  nhóm: `near_singularity > near_joint_limit > large_orientation_change >
  far_target > medium_target > near_target`.
- Reachability guarantee: mọi target Tier 1 và mọi state Tier 0 đều là cấu
  hình khớp hợp lệ được sample trực tiếp, hoặc ảnh forward-kinematics của một
  cấu hình như vậy — không có target Cartesian nào chỉ "giả định" là tới
  được (`benchmarks/README.md` mục "Reachability guarantee").

### 2. Tier 0 validation (`benchmarks/validation/`)

- `fk_test_states.npz`: **1000** state (5 nhóm × 200):
  `zero_or_home, random_interior, near_operational_lower_limit,
  near_operational_upper_limit, mixed_near_limits`
  (`benchmarks/validation/validation_checksum.json::group_definitions.fk_test_states`).
- `jacobian_test_states.npz`: **200** state (5 nhóm × 40):
  `regular, near_lower_limit, near_upper_limit, mixed_near_limits, low_sigma`
  (nhóm `low_sigma` chọn bằng cách thực sự tính `sigma_min` trên một pool
  ứng viên và giữ giá trị nhỏ nhất, không phải giả định).
- `singularity_test_states.npz`: **300** state (3 nhóm × 100):
  `regular, moderately_conditioned, near_singular`, phân theo `sigma_min`
  thật so với `configs/dls_config.json::singularity_sigma_threshold = 0.03`
  và bội số `configs/benchmark_config.json::validation_singularity_moderate_upper_multiplier
  = 3.0` (ngưỡng `moderately_conditioned_upper_bound = 0.09`).

### 3. Trajectory (`trajectories/`)

Tổng số trajectory: **8** (`trajectories/trajectory_manifest.csv`,
`DATASET_MANIFEST.json::generation_summary.trajectory_count = 8`). Tên chính
xác (`trajectory_id`): `circle_fixed_orientation`, `circle_variable_orientation`,
`figure8_fixed_orientation`, `figure8_variable_orientation`,
`line_fixed_orientation`, `line_variable_orientation`,
`helix_fixed_orientation`, `helix_variable_orientation`. Loại
(`type`): `line`, `circle`, `figure8`, `helix`
(`configs/trajectory_config.json::trajectory_types`). Chế độ hướng
(`orientation_mode`): `fixed` (giữ nguyên hướng của anchor pose) hoặc
`variable` (nội suy geodesic SO(3), `trajectories/README.md` mục
"Fixed vs. variable orientation"). Số waypoint mỗi file: **400**
(`num_waypoints` trong manifest = `configs/trajectory_config.json::default_waypoints`).
Duration: **10.0 s** (`default_duration_s`); control period thực tế:
**0.0250626566 s** (= 10.0/399, cột `control_period_s` trong manifest).
Closed/open path: `circle`, `figure8` là `closed_path=True`; `line`, `helix`
là `closed_path=False` (cột `closed_path` trong manifest). Geometric scale
(nominal, không co lại vì mọi file validate ở kích thước nominal —
`validation_waypoint_success_rate=1.0` cho cả 8 file): line length 0.12 m,
circle radius 0.045 m, figure-8 amplitude a=0.05 m/b=0.03 m, helix radius
0.04 m/height 0.08 m (`trajectories/README.md` mục "Geometry").

### 4. Trials (`trajectories/trajectory_trials.csv`)

Tổng số row: **360** (`trajectories/trajectory_trials.csv`, 361 dòng file −1
header; `DATASET_MANIFEST.json::generation_summary.trajectory_trial_count =
360`). Repeatability: **30 trial/trajectory** (10 repeat × 3 speed scale) =
240 tổng cộng. Robustness: **15 trial/trajectory** (5 `q_initial` độc lập ×
3 speed scale) = 120 tổng cộng. 30+15=45 trial/trajectory × 8 trajectory =
360. Speed scale: `[0.5, 1.0, 1.5]` (`configs/trajectory_config.json::speed_scales`).
Số repeat: repeatability dùng `repeat_id` 0-9 (10 repeat, cùng `q_initial`
là anchor, không nhiễu ngẫu nhiên — quyết định luận (deterministic) vì Tier
0-4 không có dynamics/noise); robustness dùng 5 `q_initial` độc lập, mỗi cấu
hình đã được xác nhận có thể giải waypoint đầu tiên bằng DLS trước khi được
chấp nhận vào file (`trajectories/README.md` mục "Repeatability vs.
robustness trials"). Initial-state policy: mọi `q*_init` nằm trong
operational limit của `configs/robot_config.json`.

**Full pipeline chọn row nào**: `pipelines/_common.py::select_trials`
lọc theo `trajectory_ids` (nếu chỉ định), theo `trial_category` (nếu khác
`"all"`), sắp xếp theo `trial_id`, rồi cắt theo `trial_limit`. Với Full run
(không giới hạn), **toàn bộ 360 row** được sử dụng.

### 5. Tổng workload dự kiến (tính từ định nghĩa trial thật, `select_trials`/`run_sequential_trial`)

Các con số dưới đây là **tính toán trực tiếp từ số lượng trial/waypoint đã
định nghĩa sẵn trong dataset** (không phải kết quả runtime), với giả định
không có solve nào bị cắt sớm do lỗi non-finite (fatal failure) — nếu có,
số row thực tế trong `waypoint_results.csv` sẽ **thấp hơn** các con số này.
Không nhân Cartesian tràn lan: Tier 1 không phụ thuộc trajectory/trial; Tier
2 áp dụng cho mỗi `(trial, method)` độc lập, mỗi trial giải đúng số waypoint
của trajectory nó thuộc về (400, không nhân thêm theo trajectory khác).

- **Point solves (Tier 1)**: `1200 sample × 1 (không có method warm/cold ở
  Tier 1) = 1.200 point-DLS solve`.
- **Số trial × method (Tier 2)**: `360 trial × 2 method = 720` tổ hợp
  `(trial_id, method)`.
- **Số waypoint solve (Tier 2)**: mỗi tổ hợp `(trial_id, method)` giải đúng
  400 waypoint (waypoint_limit=None ở Full) →
  `720 × 400 = 288.000 waypoint-DLS solve`, chia đều `144.000` cho
  `warm_start` và `144.000` cho `cold_start`.
- **Tổng số lệnh gọi DLS solve**: `1.200 (Tier 1) + 288.000 (Tier 2) =
  289.200`.
- **Công thức**: `waypoint_solves = (Σ trial theo mỗi trajectory) × |methods|
  × waypoints_per_trajectory = 360 × 2 × 400`. Không nhân theo số trajectory
  vì mỗi trial đã gắn với đúng một `trajectory_id`.

Tier 3/4 không giải solver mới — chỉ tiêu thụ output Tier 2 (xem mục F, G,
H) nên không cộng thêm solve call.

---

## D. Tier 0 — Kinematics Validation

**Input.** `benchmarks/validation/fk_test_states.npz` (1000),
`jacobian_test_states.npz` (200), `singularity_test_states.npz` (300);
`configs/dls_config.json::singularity_sigma_threshold=0.03`; model đã
compile qua `kinematics/model_loader.py::load_model_context` (kiểm tra
`nq=7`, `nv=7`, `joint_order`, `end_effector_site="ee_site"`,
`operational_lower_rad`/`operational_upper_rad`,
`velocity_limits_rad_s` khớp `assets/kr810.xml`). Dưới preset `full`, toàn
bộ 1000/200/300 state được dùng (`validation_*_samples = "all"`).

**Procedure** (`evaluation/kinematics_validation.py`, `kinematics/`):

- **FK finite checks** (`validate_fk_states`): `position_finite`,
  `rotation_finite` từ hai lệnh gọi `kinematics/forward_kinematics.py::forward_kinematics`
  độc lập trên cùng `q` (kiểm tra cả tính hữu hạn lẫn tính xác định —
  `np.array_equal` giữa hai kết quả).
- **Rotation orthogonality**: `orth_err = ||Rᵀ R − I||`
  (`kinematics/rotation_utils.py::validate_rotation_matrix`, `tol=1e-6`).
- **Rotation determinant**: `det = det(R)`, phải thỏa `|det−1| < tol`.
- **Quaternion normalization**: `quaternion_norm_error = |‖q‖ − 1|`
  (`kinematics/quaternion_utils.py::rotation_matrix_to_quaternion_wxyz`).
- **Analytical Jacobian**: `kinematics/jacobian.py::geometric_jacobian_world`
  (qua `mujoco.mj_jacSite` trên `ee_site`), shape (6,7).
- **Finite-difference Jacobian**: `kinematics/jacobian.py::finite_difference_jacobian_world`
  — central difference, cột hướng dùng `so3_log(R₊ R₋ᵀ)/(2ε)` (không phải
  hiệu Euler/quaternion), `ε` từ `jacobian_test_states.npz::finite_difference_epsilon`
  (hằng số 1e-6, nguồn `configs/benchmark_config.json::finite_difference_epsilon`).
- **Relative Frobenius error**: `kinematics/jacobian.py::jacobian_relative_error`
  = `‖J_analytic − J_fd‖_F / max(‖J_fd‖_F, 1e-12)` — mẫu số có sàn 1e-12 chỉ
  để tránh chia 0, **không phải** ngưỡng pass/fail.
- **Rank**: `kinematics/singularity_metrics.py::numerical_rank` (tolerance
  1e-6 tương đối so với singular value lớn nhất).
- **sigma_min/condition number**: SVD của `J_analytic`
  (`kinematics/singularity_metrics.py::singular_values`/`condition_number`,
  `condition_number` trả `inf` — không NaN — nếu `sigma_min ≤ 1e-12`).
- **Singularity distribution**: `validate_singularity_states` chỉ mô tả
  (`near_singular = sigma_min ≤ 0.03`), **không phải** kiểm tra pass/fail.

**Ngưỡng Jacobian relative error thực tế**: hằng số Python
`evaluation/kinematics_validation.py::DEFAULT_JACOBIAN_RELATIVE_ERROR_THRESHOLD
= 1e-4` — **không** nằm trong bất kỳ file `configs/*.json` nào (khác với mô
tả README chung chung "finite difference checks"); có thể ghi đè qua tham số
`jacobian_relative_error_threshold` của `run_tier0()` hoặc CLI
`--jacobian-error-threshold` (chỉ trên `run_tier0_kinematics.py` standalone).

**Gate condition** (`evaluation/kinematics_validation.py::compute_gate_result`):

```
gate_pass = (non_finite_fk == 0)
        AND (rotation_failures == 0)
        AND (jacobian_non_finite == 0)
        AND (max_jacobian_relative_error <= 1e-4)
```

- `near_singular`/`singularity` **không bao giờ** ảnh hưởng `gate_pass` — xác
  nhận rõ trong docstring module: *"Near-singular states are valid, expected
  test data — never treated as a failure on their own"*, và trong
  `summary.json::note`: *"near-singular states are expected test data, not a
  gate failure criterion."*
- **Lưu ý (gap giữa 5 kiểm tra và 4 điều kiện gate)**: FK non-determinism
  (`fk_determinism_failures`) được tính và báo cáo trong `gate_reasons`/
  `summary.json`, nhưng **không** nằm trong biểu thức boolean của
  `gate_pass` ở trên — chỉ 4 điều kiện (non-finite FK, rotation không hợp
  lệ, non-finite Jacobian, Jacobian relative error) mới gate. Vì FK ở đây là
  `mj_forward` xác định (deterministic), khả năng kích hoạt gap này trong
  thực tế rất thấp, nhưng đây là một khác biệt thật giữa code và mô tả trực
  quan "5 kiểm tra → gate".

**Xử lý khi Tier 0 fail** (`pipelines/run_tier0_to_tier4.py::main`, dòng
~197-231): nếu `gate_pass=False`, `fatal_error` được set, và **cả 4 tier
tier1-tier4** được ghi `{"status": "not_run", "reason": "Tier 0 gate
failed"}` trong `run_manifest.json::tiers`; các hàm `run_tier1`...`run_tier4`
**không bao giờ được gọi**. `overall_status="failed"`, tiến trình thoát với
exit code 1.

**Output files** (`tier0_kinematics/`, `_TIER_REQUIRED_FILES["tier0"]`):
`fk_validation.csv` (per-sample: `sample_id, group_id, position_finite,
rotation_finite, rotation_orthogonality_error, rotation_determinant,
quaternion_norm_error, rotation_valid, deterministic, passed`),
`jacobian_validation.csv` (per-sample: `sample_id, group_id, relative_error,
sigma_min, sigma_max, condition_number, numerical_rank, finite, passed`),
`singularity_validation.csv` (per-sample: `sample_id, group_id, sigma_min,
condition_number, near_singular`), `summary.json` (`gate_pass, gate_reasons,
fk_sample_count, fk_rotation_failures, fk_determinism_failures,
fk_reference_discrepancy_status, jacobian_sample_count,
max_jacobian_relative_error, mean_jacobian_relative_error,
p95_jacobian_relative_error, jacobian_relative_error_threshold,
jacobian_over_threshold_count, minimum_sigma_min, maximum_condition_number,
singularity_sample_count, singularity_minimum_sigma_min,
singularity_maximum_condition_number, near_singular_threshold, note`), cộng
`figures/jacobian_relative_error_histogram.png`,
`figures/singularity_sigma_min.png` (nếu không có `--no-plots`).
`fk_reference_discrepancy_status` luôn là `"unavailable"` vì
`fk_test_states.npz` không có pose tham chiếu độc lập nào để so sánh —
được ghi rõ thay vì bịa số.

---

## E. Tier 1 — Point DLS Evaluation

**Mỗi sample bắt đầu từ đâu.** `q_initial` = `point_ik_v1.npz::q_initial[idx]`
(`algorithms/point_dls.py::run_point_dls`). Target pose =
`target_position[idx]`/`target_quaternion[idx]` (chuyển sang rotation matrix
qua `kinematics/quaternion_utils.py::quaternion_wxyz_to_matrix`).

**`q_target` có được dùng làm initial guess không? KHÔNG.** Xác nhận trực
tiếp trong code: `q_target` chỉ được lưu lại làm
`PointIKResult.q_target_reference` để so sánh joint-space tùy chọn, không
bao giờ được truyền vào `solve_dls_until_converged`
(`algorithms/point_dls.py::_solve_one_sample`; docstring module dòng 5-8:
*"`q_target` itself is never used as an initial guess — it is only carried
through as `q_target_reference`... for optional joint-space comparison."*).

**Success condition** (per-sample, `kinematics/dls_solver.py::solve_dls_until_converged`):
`position_error_m ≤ configs/dls_config.json::position_success_threshold_m
(0.006 m)` **VÀ** `orientation_error_deg ≤
orientation_success_threshold_deg (10.0°)` đồng thời, kiểm tra cả trước vòng
lặp (early success) lẫn sau mỗi bước cập nhật.

**Max iterations**: `configs/dls_config.json::max_iterations = 100`.

**DLS weights**: `position_weight=1.0`, `orientation_weight=0.2`
(`W = diag([1,1,1,0.2,0.2,0.2])`).

**Damping**: `damping_mode="adaptive"` — `kinematics/adaptive_damping.py::compute_adaptive_damping`,
nội suy bậc hai giữa `lambda_min=0.0001` và `lambda_max=0.2` theo `sigma_min`
so với `singularity_sigma_threshold=0.03`; nếu `damping_mode≠"adaptive"` thì
dùng `lambda_default=0.01`.

**Max joint step**: `max_joint_step_rad=0.1`, clip từng thành phần
`delta_q` sau khi nhân `step_scale=1.0`.

**Joint-limit handling**: `clip_to_operational_limits=true` — nếu bước cập
nhật vi phạm operational limit, `kinematics/joint_limit_utils.py::clip_to_operational_limits`
kẹp `q_candidate` vào biên; nếu tắt, bước bị từ chối với
`failure_reason="joint_limit_failure"`.

**Null-space handling**: `joint_limit_avoidance=true`,
`null_space_gain=0.02` — dùng `J_pinv` và null-space projector `I − J⁺J` với
gradient centering `kinematics/joint_limit_utils.py::joint_centering_gradient`
đẩy `q` về giữa dải joint limit.

**Stagnation/failure handling**: cửa sổ 5 vòng lặp, ngưỡng cải thiện tương
đối tối thiểu `1e-3` (hằng số module `_STAGNATION_WINDOW=5`,
`_STAGNATION_MIN_RELATIVE_IMPROVEMENT=1e-3`, không phải config key) →
`failure_reason="stagnation"`.

**Ghi chú kỹ thuật:** `configs/dls_config.json::stop_on_nan=true` tồn tại
trong config nhưng **không được tham chiếu ở bất kỳ đâu trong code**
(xác nhận bằng grep toàn repo) — đây là một config key chết/không dùng.
Việc chặn NaN thực tế được thực hiện vô điều kiện qua kiểm tra
`np.isfinite` (không phụ thuộc key này) tại các nhánh `failure_reason
= "non_finite_jacobian"`/`"non_finite_input"`.

**Toàn bộ `failure_reason` có thể phát sinh**
(`kinematics/dls_solver.py`): `invalid_target`, `non_finite_jacobian`,
`non_finite_input`, `linear_solve_failure`, `joint_limit_failure`,
`stagnation`, `max_iterations`.

**Metric thực tế** (`evaluation/point_ik_metrics.py::compute_point_ik_metrics`,
tính cho `"overall"` và từng `difficulty_id`):

| Metric | Unit | Aggregation | Threshold | Acceptance role | Source code/config |
|---|---|---|---|---|---|
| `sample_count` | count | count | — | mô tả | `evaluation/point_ik_metrics.py::_summarize_group` |
| `success_count`/`success_rate` | count/fraction | count/mean | thành công per-sample: 0.006 m & 10° | góp phần acceptance tier | `dls_config.json::position_success_threshold_m,orientation_success_threshold_deg` |
| `success_rate_wilson_ci` | fraction | Wilson score interval, `confidence_level=0.95` (mặc định hàm, không đọc từ `evaluation_config.json` trong Tier 1) | — | mô tả độ tin cậy | `evaluation/confidence_intervals.py::wilson_confidence_interval` |
| `position_rmse_m`/`mae_m`/`median_m`/`p95_m`/`max_m` | m | RMSE/mean/median/P95/max của `position_error_m` | — | mô tả | `_summarize_group` |
| `orientation_rmse_deg`/`mae_deg`/`median_deg`/`p95_deg`/`max_deg` | deg | tương tự trên `orientation_error_deg` | — | mô tả | `_summarize_group` |
| `mean/median/p95_iterations` | count | thống kê `iterations` | — | mô tả | `_summarize_group` |
| `mean/median/p95_solve_time_ms` | ms | thống kê `solve_time_ms` | — | mô tả | `_summarize_group` |
| `failure_reason_counts` | count | `Counter` trên `failure_reason` (chỉ sample fail) | — | mô tả | `_summarize_group` |
| `joint_limit_violation_rate` | fraction | mean(vi phạm operational limit của `q_solution` cuối) | — | mô tả | `_summarize_group` |
| `success_by_sigma_min_bin` | list | 5 bin quantile theo `final_sigma_min`, mỗi bin: count/success_rate/mean position error | — | mô tả | `_bin_by_sigma_min` |
| per-difficulty breakdown | — | lặp lại toàn bộ bảng trên cho từng `difficulty_id` 0-5 | — | mô tả | `compute_point_ik_metrics` |

**Threshold success của mỗi sample** = 0.006 m & 10° (config
`dls_config.json`). **Threshold acceptance của toàn Tier** = `success_rate ≥
minimum_success_rate` — **`minimum_success_rate=0.95` là giá trị mặc định
hard-code trong chữ ký hàm `run_tier1()`** (`pipelines/run_tier1_point_dls.py`),
**không** đọc từ `configs/evaluation_config.json` (file này không có key
riêng cho Tier 1 acceptance) — cũng có thể ghi đè qua CLI
`--minimum-success-rate` (chỉ trên `run_tier1_point_dls.py` standalone).

**`execution_status` khác `acceptance_status`**: `metrics_overall.json`
ghi cả hai — `execution_status="completed"` (hard-code, luôn là giá trị này
khi `run_tier1` chạy xong, không có nhánh nào set `"failed"`), và
`acceptance_status="passed"`/`"failed"` theo `success_rate ≥
minimum_success_rate`. Tier 1 **luôn chạy tới cuối và không bao giờ gate**
phần còn lại của pipeline (`README.md` dòng 76-77; xác nhận trong
`pipelines/run_tier0_to_tier4.py`: không có nhánh nào bỏ qua Tier 2-4 dựa
trên `acceptance_status` của Tier 1).

**Output files** (`tier1_point_dls/`): `point_results.csv` (per-sample:
`sample_id, difficulty_id, success, q_initial_q1..q7,
q_target_reference_q1..q7, q_solution_q1..q7, position_error_m,
orientation_error_rad, orientation_error_deg, iterations, solve_time_ms,
initial_sigma_min, final_sigma_min, initial_condition_number,
final_condition_number, minimum_joint_limit_margin, joint_limit_violation,
failure_reason`), `metrics_overall.json` (aggregate — key liệt kê ở trên
cộng `acceptance_criterion{name,value,threshold,passed,unit,source}`),
`metrics_by_difficulty.csv` (aggregate, 1 row/group),
`failure_reasons.csv` (aggregate breakdown theo `(group, failure_reason)`),
`failure_cases.json` (per-sample fail: `sample_id, difficulty_id,
failure_reason, position_error_m, orientation_error_deg, iterations`),
`figures/position_error_cdf.png`, `figures/orientation_error_cdf.png`,
`figures/iterations_histogram.png`, `figures/runtime_histogram.png`,
`figures/success_by_difficulty.png` (nếu có plots).

---

## F. Tier 2 — Sequential DLS

### Warm-start (`algorithms/warm_start_dls.py::run_warm_start_dls`)

- Waypoint 0: seed = `q_trial_initial` (cột `q1_init..q7_init` của trial
  row trong `trajectory_trials.csv`).
- Waypoint `i>0`: seed = **`q_solution` đã GIẢI của waypoint trước** (nếu
  hội tụ), không bao giờ là target của waypoint trước.
- **Recovery policy sau failure** (`_recover_q_initial`, thứ tự ưu tiên):
  1. `q_solution` của chính waypoint fail (luôn hữu hạn trong thực tế, vì
     `dls_single_update` không bao giờ tiến `q` qua một bước non-finite).
  2. Nếu không, `q` của waypoint thành công gần nhất trước đó
     (`last_successful_q`).
  3. Nếu chưa có waypoint nào thành công, `q_trial_initial` của trial.
- Failure thường (không fatal) **không dừng chuỗi** — chỉ ghi
  `success=False` rồi tiếp tục waypoint kế; chỉ trạng thái non-finite mới
  cắt chuỗi sớm (xem mục "fatal failure" bên dưới).

### Cold-start (`algorithms/cold_start_dls.py::run_cold_start_dls`)

- **Mọi** waypoint đều seed từ **cùng một** `q_trial_initial` cố định — xác
  nhận trong code (`solve_dls_until_converged(model_context, q_trial_initial,
  ...)` gọi cho mọi `k`), không bao giờ dùng q của waypoint trước.
- Đây là baseline đo lợi ích của continuity vì nó loại bỏ hoàn toàn thông
  tin liên tục giữa các waypoint — docstring module: *"Cold-start exists
  purely as a baseline to quantify the benefit of sequential warm-starting;
  it carries no cross-waypoint continuity."*

### Speed scale và timing

- **Chỉ ảnh hưởng timing, không đổi hình học path**: `target_position`/
  `target_quaternion` dùng nguyên vẹn từ NPZ, không phụ thuộc
  `speed_scale` — chỉ `time_s_scaled = time_s / speed_scale` và
  `control_period_s_scaled = control_period_s / speed_scale` bị chia tỷ lệ
  (`algorithms/sequential_dls.py::run_sequential_trial`).

### Failure handling

- **Waypoint failure thường**: không dừng trial, tiếp tục waypoint kế.
- **Fatal numerical failure** = `q_solution` non-finite (NaN/Inf) —
  `warm_start_dls.py`/`cold_start_dls.py` sẽ `break` vòng lặp waypoint khi
  `fail_fast=False` (giá trị Tier 2 luôn dùng), trả về danh sách kết quả bị
  cắt ngắn mà **không raise exception**.
- **Fatal failure ở cấp trial** (exception thực sự, ví dụ lỗi không lường
  trước): `pipelines/run_tier2_sequential_dls.py::run_tier2` bọc mỗi tổ hợp
  `(trial_id, method)` trong `try/except`, ghi vào `failure_cases.json`,
  `continue` sang tổ hợp kế — **không bao giờ dừng toàn bộ Tier 2**, các
  trial khác vẫn tiếp tục.

### Metric (`evaluation/waypoint_metrics.py::compute_waypoint_metrics`)

`waypoint_count`, `successful_waypoints`/`failed_waypoints`,
`waypoint_success_rate` (+ Wilson CI), **`full_trajectory_completed` =
`bool(waypoint_count == expected_waypoint_count)`** — đây là tổng số
waypoint **đã được xử lý** so với số mong đợi, **không phải** "100%
waypoint success rate"; một trial có thể có
`full_trajectory_completed=True` (xử lý đủ 400 waypoint) trong khi
`waypoint_success_rate < 1.0` (một số waypoint không hội tụ nhưng vẫn cho
q hữu hạn để đi tiếp) — hai khái niệm hoàn toàn khác nhau và không được
gộp lẫn. `maximum_failure_streak`/`number_of_failure_streaks`,
`recovery_attempts`/`successful_recoveries`/`recovery_rate` (đếm mỗi lần
waypoint `i-1` fail rồi waypoint `i` có thành công hay không), `mean/p95_iterations`,
`mean/p95_runtime_ms`, `deadline_miss_count`/`deadline_miss_rate` (so
`solve_time_ms` với `deadline_s × 1000`, `deadline_s = control_period_s_scaled`;
`None` nếu không truyền deadline — không âm thầm trả 0). `sigma_min`,
`condition_number`, `minimum_joint_limit_margin` không nằm trong
`waypoint_metrics.py` — được tính trực tiếp per-waypoint trong
`algorithms/sequential_dls.py::_build_waypoint_results` rồi tổng hợp
min/max trong `pipelines/run_tier2_sequential_dls.py::_build_trial_summary`.

### Output files (`tier2_sequential_dls/`)

- **`waypoint_results.csv`** — **nguồn sự thật gốc**, per-waypoint (mỗi
  waypoint mỗi trial mỗi method một row): `trial_id, trajectory_id,
  trial_category, method, waypoint_id, time_s, q_initial_used_q1..q7,
  q_solution_q1..q7, target_position_x/y/z, actual_position_x/y/z,
  target_quaternion_qw/qx/qy/qz, actual_quaternion_qw/qx/qy/qz,
  position_error_m, orientation_error_rad, orientation_error_deg, success,
  iterations, solve_time_ms, sigma_min, condition_number, manipulability,
  minimum_joint_limit_margin, recovered_after_previous_failure,
  failure_reason`.
- **`trajectory_trial_summaries.csv`** — **summary suy ra**, 1 row/(trial_id,
  method): `trial_id, trajectory_id, trial_category, method, repeat_id,
  seed, speed_scale, control_period_s (đã scale), waypoint_count,
  successful_waypoints, failed_waypoints, waypoint_success_rate,
  full_trajectory_completed, maximum_failure_streak, recovery_rate,
  position_rmse_m, position_mae_m, position_median_m, position_p95_m,
  position_max_m, orientation_rmse_deg, orientation_p95_deg,
  orientation_max_deg, mean_iterations, p95_iterations, mean_solve_time_ms,
  p95_solve_time_ms, deadline_miss_rate, minimum_sigma_min,
  maximum_condition_number, minimum_joint_limit_margin`.
- **`warm_vs_cold.csv`** — **so sánh suy ra hai lần** (chỉ từ
  `trajectory_trial_summaries.csv`, không đụng `waypoint_results.csv`), 1
  row/`trial_id` có mặt ở cả hai method: `trial_id, trajectory_id,
  trial_category, speed_scale`, cộng `warm_{metric}`/`cold_{metric}` cho 12
  metric (`waypoint_success_rate, full_trajectory_completed,
  maximum_failure_streak, recovery_rate, position_rmse_m, position_p95_m,
  orientation_rmse_deg, mean_iterations, mean_solve_time_ms,
  p95_solve_time_ms, deadline_miss_rate, minimum_sigma_min`).
- **`failure_cases.json`** — 2 dạng row: fatal-trial-level
  (`trial_id, trajectory_id, method, fatal_error, traceback`) và
  ordinary-waypoint-level (`trial_id, trajectory_id, method, waypoint_id,
  failure_reason, position_error_m, orientation_error_deg`).

---

## G. Tier 3 — Trajectory Tracking

**Tier 3 dùng output Tier 2, không chạy solver lại.** Docstring
`pipelines/run_tier3_trajectory_tracking.py`: *"never re-runs the DLS
solver: every metric here is a deterministic function of the already-solved
target/actual pose columns."* Input trực tiếp:
`tier2_sequential_dls/waypoint_results.csv` (hoặc DataFrame trong bộ nhớ khi
chạy qua `run_tier0_to_tier4`), cộng `trajectories/trajectory_manifest.csv`
(lấy `closed_path`) và `trajectories/trajectory_trials.csv` (lấy
`speed_scale`).

**Đồng bộ target/actual**: index-aligned theo `waypoint_id` sau khi sort —
**không resampling/interpolation**; một mismatch độ dài luôn raise
`ValueError` (`evaluation/trajectory_metrics.py`).

### Metric

- Position: `rmse_m, mae_m, median_m, p95_m, max_m, endpoint_error_m,
  start_point_error_m`, component-wise `rmse_x_m/y_m/z_m`,
  `centroid_offset_m`, `target_path_length_m`/`actual_path_length_m`,
  `path_length_ratio` (`evaluation/trajectory_metrics.py::compute_position_tracking_metrics`).
  Lưu ý: `rmse_x/y/z_m` và `path_length_abs_diff_m` được tính trong
  dataclass nhưng **không** được ghi ra `trajectory_metrics.csv` (không nằm
  trong `_TRACKING_COLUMNS`).
- Orientation: geodesic SO(3)-log angle (`kinematics/rotation_utils.py::rotation_geodesic_angle`,
  `arccos(clip((trace(R1ᵀR2)−1)/2,−1,1))`), tổng hợp RMSE/MAE/median/P95/max
  (`evaluation/orientation_metrics.py::summarize_orientation_errors`).
- **Cross-track**: chiếu điểm actual lên **đoạn polyline gần nhất** (không
  phải waypoint gần nhất), tham số hóa `t` kẹp `[0,1]` trên mỗi segment
  (`evaluation/cross_track_metrics.py::project_point_to_polyline`). Với
  path đóng (`closed_path=True`, circle/figure8), một segment "khép kín"
  nối điểm cuối về điểm đầu được thêm vào để tránh phạt sai điểm gần mối
  nối. Metric: `cross_track_rmse_m/mae_m/p95_m/max_m`.
- **Along-track**: tọa độ dọc path tích lũy `along_coord`, `backward_progress_count`
  (đếm điểm lùi quá `1e-9`), `final_progress_ratio = along[-1] /
  total_path_length`, và `synchronized_along_track_rmse_m` (chỉ tính khi
  target/actual cùng độ dài, so `along` thực tế với `along` của path lệnh
  gốc — không nội suy).

### ISO 9283-inspired (`evaluation/iso9283_metrics.py`)

- **Không phải chứng nhận ISO 9283** — disclaimer trong docstring module và
  `configs/evaluation_config.json::note`, `DATASET_MANIFEST.json::acceptance_criteria.note`.
- `ATp` (accuracy): `atp_m = max_j ‖commanded[j] − mean_attained[j]‖` trên
  tập repeat cùng điều kiện (`compute_path_accuracy`).
- `RTp` (repeatability): `RTp_j = r̄_j + 3·s_{r,j}` (mean + 3×std bán kính
  lệch), `rtp_m = max_j RTp_j` (`compute_path_repeatability`).
- **Grouping**: chỉ trial có `trial_category="repeatability"`, nhóm theo
  `(trajectory_id, method, speed_scale)`. **Robustness trial bị loại hoàn
  toàn** khỏi metric này (raise `ValueError` nếu lẫn vào).
- **Minimum repeat**: `MIN_REPEATS_FOR_STD=2` (bắt buộc để có std mẫu, nếu
  không nhóm bị bỏ qua, không ghi row); `RECOMMENDED_MIN_REPEATS=10` — nếu
  2 ≤ n < 10, một `warning` (không fatal) được đính vào row: *"only {n}
  repeats available; ISO 9283-inspired repeatability is conventionally
  reported from >= 10 repeats."* Với Full run, mỗi group repeatability có
  đúng 10 repeat nên điều kiện khuyến nghị luôn được thỏa.
- **Trường hợp "unavailable"**: nhóm bị bỏ qua nếu < 2 repeat hợp lệ, hoặc
  nếu commanded path không nhất quán giữa các repeat trong nhóm; nếu
  `iso9283_metrics.csv` rỗng toàn bộ, pipeline log warning giải thích đây
  là kỳ vọng dưới `--trial-limit` nhỏ, không phải lỗi tính toán.
- **Vì sao mô phỏng deterministic có thể cho RTp gần 0**: docstring hàm
  `compute_path_repeatability`: *"a deterministic kinematic simulation with
  no injected noise is expected to give an RTp near 0 — that is the expected
  result here, not a sign of a broken metric."*

### Confidence interval (`evaluation/confidence_intervals.py`)

- **Bootstrap cấp trial** (một giá trị scalar/trial, ví dụ RMSE của cả
  trial), **không** bootstrap từng waypoint riêng lẻ trong một trial (chúng
  không độc lập). Chỉ 2 metric được bootstrap: `position_rmse_m`,
  `orientation_rmse_deg`, mỗi metric theo `method`.
- Số resample: `bootstrap_resamples=10000`
  (`configs/evaluation_config.json`, cũng là default hàm `run_tier3`).
- Seed: `42` (default hàm `run_tier3`, `np.random.default_rng(seed)`).
- Phương pháp: percentile bootstrap của mean, khoảng tin cậy
  `confidence_level=0.95` (`evaluation_config.json::confidence_level`).

### Output files (`tier3_trajectory_tracking/`)

`trajectory_metrics.csv` (1 row/(trial_id,method)), `cross_track_metrics.csv`
(1 row/(trial_id,method)), `iso9283_metrics.csv` (1 row/(trajectory_id,
method, speed_scale) nhóm đủ điều kiện — có thể rỗng nhưng vẫn đúng header),
`confidence_intervals.csv` (1 row/(method,metric), tối đa 4 row), cộng
`figures/` (biểu đồ 3D/xyz/lỗi theo trial đại diện + 1 CDF toàn cục).

---

## H. Tier 4 — Joint Smoothness và Feasibility

**q trajectory lấy từ đâu**: cột `q_solution_q1..q7` của
`tier2_sequential_dls/waypoint_results.csv`, nhóm theo `(trial_id, method)`
— **cả warm_start lẫn cold_start đều được phân tích như nhau**, không lọc
theo `method`. Các waypoint có `success=False` vẫn được đưa vào (vì
`q_solution` luôn hữu hạn cho waypoint thường-fail, chỉ fatal
non-finite mới thực sự thiếu dữ liệu).

**Time vector**: cột `time_s` thật (đã scale theo speed) của Tier 2, dùng
qua `np.gradient(values, time_s, axis=0, edge_order=...)` —
**không giả định uniform sampling**; `control_period_s` chỉ dùng để tính
deadline runtime, không dùng cho đạo hàm.

**Derivative method**: finite-difference kiểu central-difference của
`numpy.gradient` chống nonuniform-spacing (`evaluation/smoothness_metrics.py::_gradient`),
`edge_order=2` nếu ≥3 sample, else 1. `velocity = ∇q`, `acceleration =
∇velocity`, `jerk = ∇acceleration`. Ngưỡng số điểm tối thiểu:
`MIN_POINTS_VELOCITY=2`, `_ACCELERATION=3`, `_JERK=4` — dưới ngưỡng, field
tương ứng trả `None` với `*_available=False`, không bịa số.

**Solver failure ảnh hưởng coverage ra sao**: nếu một trial bị `break` sớm
(gặp non-finite) hoặc bị `--waypoint-limit` cắt, số row nhóm được ít hơn
`num_waypoints` (từ `trajectory_manifest.csv`) → `coverage_ratio =
len(group)/expected < 1.0`, ghi rõ trong output thay vì coi dữ liệu cắt là
đầy đủ. Không nội suy/lấp khoảng trống — đạo hàm chỉ tính trên các sample
thực sự có mặt.

### Metric

`max_joint_jump_rad` (Δq lớn nhất giữa 2 sample liên tiếp, mọi khớp), per-joint
jump, vị trí xảy ra (`joint_index`, `timestep_index`); velocity (max/rms
toàn cục), `velocity_utilization = |velocity|/velocity_limits_rad_s`
(`configs/robot_config.json`), `velocity_violation_count` (utilization > 1
+ dung sai 1e-6); acceleration (nếu ≥3 sample); jerk và `global_rms_jerk`
(nếu ≥4 sample); `max_total_joint_variation_rad` (tổng Σ|Δq| toàn quỹ đạo);
`second_difference_norm_rad`; `minimum_normalized_joint_limit_margin`,
`operational_limit_violation_count/rate` (so
`operational_lower_rad`/`operational_upper_rad`); `acceleration_status`
(xem dưới); `sigma_min`, `condition_number` — **đọc trực tiếp từ cột
`sigma_min`/`condition_number` đã có sẵn trong `waypoint_results.csv` của
Tier 2, không tính lại từ Jacobian**; `near_singular_fraction/count` (so
`sigma_min ≤ dls_config.json::singularity_sigma_threshold=0.03`);
`p05_sigma_min`, `worst_waypoint_index`; runtime (mean/median/p90/p95/p99/max/std,
`evaluation/runtime_metrics.py::compute_runtime_metrics`), `deadline_miss_count/rate`
(deadline = `control_period_s/speed_scale × 1000` ms); `coverage_ratio`.

**Acceleration limit có thật hay không? KHÔNG.**
`configs/robot_config.json` không có key acceleration-limit nào. Tier 4
luôn gọi `compute_joint_feasibility_metrics(..., acceleration_limits=None)`
→ `acceleration_status="unavailable"`, `maximum_acceleration_utilization=None`,
`acceleration_violation_count=None` — **không bao giờ bịa một con số**
(docstring `evaluation/joint_feasibility_metrics.py`: *"Acceleration limits
are never fabricated... acceleration utilization is only computed when the
caller explicitly supplies `acceleration_limits`."*).

**Cold-start discontinuity không được làm mượt.** Docstring pipeline: *"Cold-start
solutions can jump discontinuously between waypoints... This is measured,
not smoothed away — `max_joint_jump_rad` is reported as-is for both
methods."* Không có bước lọc/smoothing nào trong `smoothness_metrics.py`.

### Output files (`tier4_joint_feasibility/`)

`smoothness_metrics.csv` (1 row/(trial_id,method): `trial_id, trajectory_id,
method, joint_count, sample_count, velocity_available,
global_max_abs_velocity_rad_s, global_rms_velocity_rad_s,
acceleration_available, global_max_abs_acceleration_rad_s2,
global_rms_acceleration_rad_s2, jerk_available, global_max_abs_jerk_rad_s3,
global_rms_jerk_rad_s3, max_joint_jump_rad, max_joint_jump_joint_index,
max_joint_jump_timestep_index, max_total_joint_variation_rad,
second_difference_norm_rad, coverage_ratio`), `joint_feasibility_metrics.csv`
(1 row/(trial_id,method): `..., operational_limit_violation_count/rate,
minimum_normalized_joint_limit_margin, maximum_velocity_utilization,
velocity_violation_count, acceleration_status,
maximum_acceleration_utilization, acceleration_violation_count,
coverage_ratio`), `singularity_path_metrics.csv` (1 row/(trial_id,method):
`..., waypoint_count, minimum_sigma_min, p05_sigma_min,
maximum_condition_number, near_singular_count, near_singular_fraction,
worst_waypoint_index`), `runtime_metrics.csv` (1 row/(trial_id,method):
`..., count, mean_ms, median_ms, p90_ms, p95_ms, p99_ms, max_ms,
deadline_miss_count, deadline_miss_rate`), cộng `figures/` (quỹ đạo khớp,
sigma_min, velocity/acceleration/jerk theo trial đại diện — 2 loại cuối chỉ
khi `*_available=True`).

---

## I. Project Acceptance Criteria

Trích chính xác từ `pipelines/run_tier0_to_tier4.py::_build_final_summary`
và `configs/evaluation_config.json`:

| Criterion | Threshold | Unit | Tier | Pass condition | Nguồn config/code | Project/ISO status |
|---|---|---|---|---|---|---|
| `tier1_minimum_success_rate` | 0.95 | fraction | Tier 1 | `success_rate ≥ 0.95` | **hard-code** trong `run_tier1_point_dls.py::run_tier1` (mặc định tham số hàm), **không** trong `evaluation_config.json` | project_criterion |
| `tier2_{method}_waypoint_success_rate` | 0.95 | fraction | Tier 2 | `waypoint_success_rate ≥ 0.95` | `evaluation_config.json::minimum_waypoint_success_rate` | project_criterion |
| `tier2_{method}_trajectory_completion_rate` | 1.0 | fraction | Tier 2 | `full_completion_rate ≥ 1.0` | `evaluation_config.json::required_trajectory_completion_rate` | project_criterion |
| `tier3_{method}_position_rmse_mm` | 4.0 | mm | Tier 3 | `position_rmse_mm ≤ 4.0` | `evaluation_config.json::kinematic_position_rmse_target_mm` | project_criterion |
| `tier3_{method}_position_p95_mm` | 6.0 | mm | Tier 3 | `position_p95_mm ≤ 6.0` | `evaluation_config.json::kinematic_position_p95_target_mm` | project_criterion |
| `tier3_{method}_orientation_p95_deg` | 10.0 | deg | Tier 3 | `orientation_p95_deg ≤ 10.0` | `evaluation_config.json::orientation_p95_target_deg` | project_criterion |

Mỗi entry (2 method × 6 tiêu chí có method-suffix = tối đa 11 entry: Tier1
không có method-suffix nên 1 + Tier2 2×2 + Tier3 2×3 = 1+4+6=11) được ghi
vào `FINAL_SUMMARY.json::acceptance_criteria[]` với schema bắt buộc
`{name, tier, value, threshold, passed, unit, source="project_criterion"}`.

**Các threshold KHÔNG được biến thành acceptance criterion dù có trong
config**: `evaluation_config.json::position_threshold_m` (0.006 m) và
`orientation_threshold_deg` (10.0°) — đây là ngưỡng **per-sample DLS
success** (trùng giá trị với `dls_config.json::position_success_threshold_m`/
`orientation_success_threshold_deg`, dùng ở cấp thuật toán, không phải cấp
Tier acceptance); `kinematic_position_max_target_mm` (10.0 mm) — chỉ được
**hiển thị** (`position_max_mm` trong block Tier1/Tier3 của
`FINAL_SUMMARY.json`) chứ **không được so sánh với ngưỡng nào**. **Tier 4
không có acceptance criterion nào cả** — mọi metric Tier 4 chỉ mang tính mô
tả trong `FINAL_SUMMARY.json`.

**Đây là project criteria, không phải chứng nhận ISO.** Ghi rõ trong
`README.md` mục "On acceptance thresholds", `configs/evaluation_config.json::note`,
và `note` cuối mỗi `FINAL_SUMMARY.json` (`_build_final_summary`, dòng
547-551).

**Acceptance failure không nhất thiết làm pipeline execution fail.** Exit
code của `main()` chỉ phụ thuộc `overall_status` (`completed`→0,
`failed`→1), mà `overall_status` chỉ phụ thuộc `fatal_error is None` (một
exception không bắt được hoặc Tier 0 gate fail) — **không** phụ thuộc bất
kỳ `acceptance_criteria[i]["passed"]` nào. Một run mà mọi tiêu chí Tier
1/2/3 đều fail nhưng không có exception/Tier 0 gate fail vẫn thoát với exit
code **0**.

---

## J. Output Structure

Cây thư mục Full run thực tế (`pipelines/_common.py::ensure_output_structure`,
`pipelines/run_tier0_to_tier4.py`, và từng tier runner):

```
results/full_run/
├── run.log
├── resolved_config.json
├── run_manifest.json
├── FINAL_SUMMARY.json
├── tier0_kinematics/
│   ├── fk_validation.csv
│   ├── jacobian_validation.csv
│   ├── singularity_validation.csv
│   ├── summary.json
│   └── figures/
│       ├── jacobian_relative_error_histogram.png
│       └── singularity_sigma_min.png
├── tier1_point_dls/
│   ├── point_results.csv
│   ├── metrics_overall.json
│   ├── metrics_by_difficulty.csv
│   ├── failure_reasons.csv
│   ├── failure_cases.json
│   └── figures/ (5 file .png)
├── tier2_sequential_dls/
│   ├── waypoint_results.csv
│   ├── trajectory_trial_summaries.csv
│   ├── warm_vs_cold.csv
│   ├── failure_cases.json
│   └── figures/ (3 file so sánh + N file per-trajectory)
├── tier3_trajectory_tracking/
│   ├── trajectory_metrics.csv
│   ├── cross_track_metrics.csv
│   ├── iso9283_metrics.csv
│   ├── confidence_intervals.csv
│   └── figures/ (4 file/nhóm đại diện + 1 CDF toàn cục)
└── tier4_joint_feasibility/
    ├── smoothness_metrics.csv
    ├── joint_feasibility_metrics.csv
    ├── singularity_path_metrics.csv
    ├── runtime_metrics.csv
    └── figures/ (2-5 file/nhóm đại diện)
```

`figures/` luôn được tạo (kể cả rỗng) trừ khi `--no-plots`. Không file nào
được tạo chỉ để "cho đủ" — nếu metric hợp lệ không tính được (ví dụ
acceleration), giá trị được ghi `"unavailable"`/`null`, không phải số bịa.

| Output file | Tier | Granularity | Primary purpose | Nguồn sự thật/summary |
|---|---|---|---|---|
| `fk_validation.csv`, `jacobian_validation.csv`, `singularity_validation.csv` | 0 | per-sample | Kiểm tra FK/Jacobian/singularity từng state | Nguồn sự thật |
| `tier0_kinematics/summary.json` | 0 | aggregate | Gate pass/fail + thống kê tổng | Summary |
| `point_results.csv` | 1 | per-sample | Chi tiết từng lần giải Point-IK | Nguồn sự thật |
| `metrics_overall.json`, `metrics_by_difficulty.csv`, `failure_reasons.csv` | 1 | aggregate | Success rate, sai số theo difficulty | Summary |
| `failure_cases.json` (Tier1/2) | 1/2 | per-sample/per-waypoint (fail only) | Điều tra nguyên nhân fail | Summary có chọn lọc |
| `waypoint_results.csv` | 2 | per-waypoint | Toàn bộ lịch sử giải Sequential DLS | **Nguồn sự thật gốc, quan trọng nhất cho Tier 3/4** |
| `trajectory_trial_summaries.csv` | 2 | per-(trial,method) | Rollup mỗi trial: success rate, RMSE, runtime | Summary suy từ `waypoint_results.csv` |
| `warm_vs_cold.csv` | 2 | per-trial (paired) | So sánh trực tiếp warm vs cold cùng trial | Summary suy 2 lần |
| `trajectory_metrics.csv`, `cross_track_metrics.csv` | 3 | per-(trial,method) | Độ chính xác tracking, cross-track | Summary từ `waypoint_results.csv` |
| `iso9283_metrics.csv` | 3 | per-(trajectory,method,speed_scale) | Accuracy/repeatability kiểu ISO-inspired | Summary |
| `confidence_intervals.csv` | 3 | per-(method,metric) | Độ tin cậy bootstrap của RMSE | Summary |
| `smoothness_metrics.csv`, `joint_feasibility_metrics.csv`, `singularity_path_metrics.csv`, `runtime_metrics.csv` | 4 | per-(trial,method) | Độ mượt, khả thi khớp, singularity dọc path, runtime | Summary từ `waypoint_results.csv` |
| `FINAL_SUMMARY.json` | tất cả | aggregate toàn run | Trạng thái tổng + acceptance criteria | Summary cấp cao nhất |
| `run_manifest.json` | tất cả | aggregate toàn run | Provenance: seed, checksum, trạng thái từng tier, resume | Provenance |
| `resolved_config.json` | tất cả | aggregate toàn run | Snapshot config đã merge CLI>preset>default | Provenance |

---

## K. FINAL_SUMMARY và Status Semantics

Vocabulary trạng thái đầy đủ tìm thấy trong code
(`schemas/summary_schema.json`, `pipelines/run_tier0_to_tier4.py`):

- **`overall_status`** (`completed`|`failed`): chỉ phụ thuộc
  `fatal_error is None`. `fatal_error` chỉ được set bởi (a) Tier 0 gate
  fail, hoặc (b) một exception không bắt được ở cấp `main()`.
- **`tier0.status`** (`passed`|`failed`|`not_run`): `passed` nếu
  `gate_pass=True`, else `failed`.
- **`tier1.status`** = `acceptance_status` (`passed`|`failed`) — **khác**
  với **`tier1.execution_status`** (`completed`|`failed`, thực tế luôn
  `completed`).
- **`tier2/3/4.status`** (`completed`|`not_run` — **không có `"failed"`
  ở cấp FINAL_SUMMARY cho các tier này**).
- **`acceptance_status`** (Tier 1 only, cả 2 file): `passed`|`failed`.
- **`gate_pass`**: boolean thuần (không phải chuỗi enum), chỉ ở Tier 0.
- **`acceleration_status`** (Tier 4, per method): `available`|`unavailable`
  — trong thực tế luôn `unavailable` vì repo chưa cấu hình acceleration
  limit.
- `run_manifest.json::tiers[tier].status` dùng vocabulary khác:
  `completed`|`not_run`|`skipped`.

Các trường hợp cụ thể:

1. **Pipeline completed nhưng Tier 1 acceptance failed**: `overall_status="completed"`
   (không có exception nào xảy ra), nhưng
   `FINAL_SUMMARY.json::tier1.status="failed"` và
   `acceptance_criteria[]` có entry `tier1_minimum_success_rate` với
   `passed=false`. Notebook Cell 11 nhắc rõ: *"'Tier execution completed'
   means each tier ran and produced valid output — it does NOT mean every
   acceptance_criterion below passed."*
2. **ISO metric unavailable**: nếu không nhóm nào đủ ≥2 repeat, `atp_mm`/
   `rtp_mm` trong `tier3.{method}` là `null`, cộng một chuỗi cảnh báo trong
   `warnings[]` — không có field `iso_status` riêng, tình trạng "unavailable"
   thể hiện thuần túy qua `null` + `warnings`.
3. **Acceleration unavailable**: `tier4.{method}.acceleration_status="unavailable"`
   luôn luôn (vì `acceleration_limits=None` không đổi trong code hiện tại).
4. **Một trial fatal nhưng các trial khác tiếp tục**: không có trường
   partial-failure cấp FINAL_SUMMARY nào — chỉ có
   `tier2_sequential_dls/failure_cases.json` ghi lại trial đó, và các
   rate tổng hợp (`waypoint_success_rate`...) chỉ tính trên các trial còn
   lại (không tính trial fatal). `full_trajectory_completed` (per trial
   trong `trajectory_trial_summaries.csv`) là chỉ báo gần nhất cho
   pass/fail cấp trial.
5. **Tier 0 gate failed**: `tier0.status="failed"`, và **`tier1`, `tier2`,
   `tier3`, `tier4`** trong `FINAL_SUMMARY.json` **đều là `"not_run"`**,
   `acceptance_criteria=[]` (rỗng — không tier nào chạy tới bước append),
   `overall_status="failed"`, exit code 1.

**Điểm không nhất quán đã phát hiện (nội bộ status semantics)**: khi Tier
2 bị bỏ qua vì "no Tier 2 waypoint results",
`run_manifest.json::tiers.tier3/tier4.status="skipped"` (kèm `reason`), còn
`FINAL_SUMMARY.json::tier3/tier4.status="not_run"` — **hai file dùng hai
từ khác nhau (`skipped` vs `not_run`) cho cùng một tình huống**; cần đọc
đúng file khi kiểm tra trạng thái (xem thêm mục O).

---

## L. Cách phân tích Full result sau khi chạy

| # | Câu hỏi nghiên cứu | File(s) | Cột/khóa | So sánh/thống kê | Diễn giải |
|---|---|---|---|---|---|
| 1 | DLS thất bại nhiều nhất ở difficulty nào? | `tier1_point_dls/metrics_by_difficulty.csv` | `difficulty_name, success_rate` | So `success_rate` 6 nhóm | Nhóm có `success_rate` thấp nhất là "khó nhất" cho Point-DLS |
| 2 | Failure reason chủ yếu là gì? | `tier1_point_dls/failure_reasons.csv`, `tier2_sequential_dls/failure_cases.json` | `failure_reason, count` | Xếp hạng count giảm dần theo group | So `stagnation` vs `max_iterations` vs `joint_limit_failure` v.v. |
| 3 | Failure có tương quan với sigma_min? | `tier1_point_dls/point_results.csv` (`success`, `final_sigma_min`), `metrics_overall.json::success_by_sigma_min_bin` | `success, final_sigma_min` | So `success_rate` giữa 5 bin sigma_min | Bin sigma_min thấp có `success_rate` thấp hơn rõ rệt → tương quan |
| 4 | Near-joint-limit có làm success giảm? | `tier1_point_dls/point_results.csv` (`minimum_joint_limit_margin`), `metrics_by_difficulty.csv` (`difficulty_name="near_joint_limit"`) | `success, minimum_joint_limit_margin` | So `success_rate` nhóm `near_joint_limit` với overall | So sánh trực tiếp |
| 5 | Position hay orientation là bottleneck? | `tier1_point_dls/point_results.csv`, `tier3_trajectory_tracking/trajectory_metrics.csv` | `position_error_m` vs `orientation_error_deg` (đã chuẩn hóa theo threshold 0.006 m / 10°) | So tỷ lệ sample vượt từng ngưỡng riêng | Ngưỡng nào bị vi phạm thường xuyên hơn là bottleneck |
| 6 | Warm-start giảm bao nhiêu iteration so với cold-start? | `tier2_sequential_dls/warm_vs_cold.csv` | `warm_mean_iterations, cold_mean_iterations` (từ `trajectory_trial_summaries.csv::mean_iterations`) | `cold − warm` per trial, trung bình | Số dương lớn → warm-start hiệu quả rõ |
| 7 | Warm-start có giảm runtime? | `tier2_sequential_dls/warm_vs_cold.csv` | `warm_mean_solve_time_ms, cold_mean_solve_time_ms` | So `p95_solve_time_ms` cả 2 | Runtime là proxy gián tiếp (phụ thuộc phần cứng) |
| 8 | Cold-start có gây joint jump lớn hơn? | `tier4_joint_feasibility/smoothness_metrics.csv` (lọc `method`) | `max_joint_jump_rad` | So warm vs cold cùng `trial_id`/`trajectory_id` | Cold-start dự kiến có jump lớn hơn do mất continuity |
| 9 | Variable orientation khó hơn fixed? | `tier3_trajectory_tracking/trajectory_metrics.csv` (join `trajectory_id`→`orientation_mode` qua `trajectory_manifest.csv`) | `orientation_rmse_deg, orientation_p95_deg` | So trung bình theo `orientation_mode` | RMSE/P95 cao hơn ở `variable` → khó hơn |
| 10 | Loại trajectory nào khó nhất? | `tier3_trajectory_tracking/trajectory_metrics.csv`, `cross_track_metrics.csv` | `position_rmse_m, cross_track_rmse_m` theo `trajectory_id`/`type` | Nhóm theo `type` (line/circle/figure8/helix), so trung bình | Loại có RMSE/cross-track cao nhất |
| 11 | Speed scale ảnh hưởng deadline và velocity? | `tier2_sequential_dls/trajectory_trial_summaries.csv` (`deadline_miss_rate`), `tier4_joint_feasibility/joint_feasibility_metrics.csv` (`maximum_velocity_utilization`) | theo `speed_scale` (join qua `trial_id`→`trajectory_trials.csv`) | So 3 mức speed_scale 0.5/1.0/1.5 | Speed cao hơn dự kiến tăng deadline miss và velocity utilization |
| 12 | Cross-track error khác synchronized error thế nào? | `tier3_trajectory_tracking/cross_track_metrics.csv` | `cross_track_rmse_m` vs `synchronized_along_track_rmse_m` | So 2 cột cùng row | Cross-track đo lệch hình học path; synchronized đo lệch tiến độ theo thời gian |
| 13 | Có trajectory completed nhưng waypoint success <100%? | `tier2_sequential_dls/trajectory_trial_summaries.csv` | `full_trajectory_completed, waypoint_success_rate` | Lọc `full_trajectory_completed=True AND waypoint_success_rate<1.0` | Xác nhận 2 khái niệm độc lập (mục F) |
| 14 | DLS có đạt project criteria không? | `FINAL_SUMMARY.json::acceptance_criteria` | `name, value, threshold, passed` | Đếm `passed=false` | Trả lời trực tiếp — không suy diễn trước khi có Full result thật |
| 15 | Phần nào là mục tiêu hợp lý cho MPDIK? | Toàn bộ `tier1-4` + `acceptance_criteria` fail | mọi metric có `passed=false` hoặc `success_rate` thấp theo difficulty/loại trajectory | Xếp hạng theo mức độ lệch threshold | Vùng DLS yếu nhất (ví dụ near_singularity, high speed_scale) là ứng viên để MPDIK cải thiện — **chỉ nêu vùng cần cải thiện, không kết luận MPDIK sẽ tốt hơn** |

---

## M. Điều kiện khóa DLS baseline

Checklist kỹ thuật để một Full DLS run đủ điều kiện dùng làm baseline so
sánh MPDIK sau này:

- [ ] Tier 0 `gate_pass=True` (`tier0_kinematics/summary.json::gate_pass`,
      `FINAL_SUMMARY.json::tier0.status="passed"`).
- [ ] Chạy với `--preset full`, không có `--point-sample-limit`,
      `--trial-limit`, `--waypoint-limit`, `--trajectory-ids` nào được
      truyền (`resolved_config.json::effective` xác nhận mọi limit là
      `null`/toàn bộ dataset).
- [ ] `--seed` được ghi lại (`run_manifest.json::seed`).
- [ ] `resolved_config.json` tồn tại và khớp `configs/*.json` hiện tại của
      repo tại thời điểm chạy.
- [ ] Checksum input tồn tại: `run_manifest.json::checksums`
      (`model_sha256`, `point_benchmark_sha256`,
      `trajectory_file_checksums`), khớp
      `assets/model_metadata.json`/`benchmarks/point_ik/point_ik_checksum.json`/
      `trajectories/trajectory_manifest.csv::sha256`.
- [ ] Cả `warm_start` và `cold_start` đã chạy
      (`tier2_sequential_dls/warm_vs_cold.csv` có đủ 360 `trial_id`).
- [ ] Output không thiếu — mọi file bắt buộc theo
      `pipelines/run_tier0_to_tier4.py::_TIER_REQUIRED_FILES` tồn tại và
      valid JSON/CSV.
- [ ] Failure không bị bỏ sót: `tier1_point_dls/failure_cases.json`,
      `tier2_sequential_dls/failure_cases.json` được kiểm tra, không bị xóa.
- [ ] Runtime được ghi (`solve_time_ms` trong `waypoint_results.csv`,
      `point_results.csv`; `runtime_metrics.csv` Tier 4) — **runtime phụ
      thuộc phần cứng chạy, không so sánh tuyệt đối giữa các máy khác nhau**.
- [ ] Acceptance criteria (mục I) không bị chỉnh sửa giữa các lần chạy so
      sánh (`configs/evaluation_config.json` không đổi).
- [ ] `result ZIP` (do notebook hoặc thủ công) được lưu lại kèm
      `run_manifest.json`, `resolved_config.json`, `FINAL_SUMMARY.json`.
- [ ] Git commit/version được ghi nếu có — `run_manifest.json` hiện **không**
      ghi git commit hash của chính run đó (chỉ `scripts/package_kaggle_release.py::git_release_info`
      mới ghi `source_git_commit` khi đóng gói release, không phải khi chạy
      pipeline) — nên ghi thủ công commit hash bên ngoài nếu cần truy vết.

**Phân biệt hai loại điều kiện:**

- **Điều kiện kỹ thuật để baseline hợp lệ** (checklist trên): đảm bảo run
  là "Full", tái lập được, không thiếu dữ liệu — **không** yêu cầu DLS phải
  "thắng" bất kỳ threshold nào.
- **Điều kiện để DLS đạt tiêu chí nghiên cứu**: `FINAL_SUMMARY.json::acceptance_criteria`
  toàn bộ `passed=true` (mục I) — đây là câu hỏi kết quả, chỉ trả lời được
  **sau khi** đã có Full run hợp lệ, không phải điều kiện tiên quyết để
  chạy hay để dùng làm baseline so sánh.

---

## N. Yêu cầu so sánh MPDIK sau này

**Không thiết kế MPDIK ở đây.** Chỉ liệt kê điều kiện để một so sánh sau
này (nếu và khi MPDIK được triển khai) là công bằng với baseline DLS Full
đã mô tả:

- Cùng bộ 1200 Point-IK sample (`benchmarks/point_ik/point_ik_v1.npz`,
  cùng checksum).
- Cùng 8 trajectory, cùng 400 waypoint/trajectory
  (`trajectories/trajectory_manifest.csv`, cùng sha256).
- Cùng 360 trial (`trajectories/trajectory_trials.csv`, cùng `q*_init`).
- Cùng `q_initial` xuất phát cho mỗi sample/trial (không đổi seed random
  hoặc random lại `q_initial`).
- Cùng threshold: `position_success_threshold_m=0.006`,
  `orientation_success_threshold_deg=10.0` (per-sample/per-waypoint), và
  cùng bộ acceptance criteria ở mục I (project-level).
- Cùng số bước tối đa (`max_iterations=100` cho DLS — nếu MPDIK có khái
  niệm tương đương, phải công bố và giữ cố định để so sánh công bằng).
- Cùng `control_period_s`/`speed_scale` khi so runtime/deadline.
- Cùng bộ metric (mục E-H — RMSE, P95, success rate, cross-track, ISO
  9283-inspired, smoothness/feasibility).
- Cùng failure policy khi có thể áp dụng (ví dụ cách xử lý fatal
  non-finite, cách một waypoint fail không dừng trial).
- Cùng chính sách đo runtime/phần cứng (chạy trên cùng máy hoặc công bố rõ
  cấu hình phần cứng khi so sánh số runtime tuyệt đối).
- **So sánh theo cặp (paired comparison)**: so từng `sample_id`/`trial_id`
  tương ứng, không so trung bình tổng quát giữa hai tập dữ liệu khác nhau.

**Output DLS phải giữ làm baseline** (không ghi đè):
`tier1_point_dls/point_results.csv`, `tier2_sequential_dls/waypoint_results.csv`
và `trajectory_trial_summaries.csv`, `tier3_trajectory_tracking/trajectory_metrics.csv`
và `cross_track_metrics.csv` và `iso9283_metrics.csv`,
`tier4_joint_feasibility/*.csv`, cùng `FINAL_SUMMARY.json` và
`run_manifest.json` (để chứng minh cấu hình/checksum/seed đã dùng).

---

## O. Rủi ro, giới hạn và bất nhất

**Giới hạn đã chứng minh từ repository:**

- **Chỉ kinematic-only**: không actuator dynamics, không torque/controller
  (`README.md` dòng 184-186; `assets/model_metadata.json`; không actuator
  nào trong `assets/kr810.xml`, `nu=0`).
- **`ee_site` chưa phải TCP hiệu chuẩn**: chỉ là link reference frame
  (`assets/ASSET_CONVERSION_REPORT.md` mục "ee_site definition").
- **Operational joint limit của joint liên tục**: `joint_1, joint_3,
  joint_5, joint_6, joint_7` dùng dải ±2π từ URDF như một quy ước continuous
  joint, **không phải hard-stop cơ khí đã kiểm chứng**
  (`assets/model_metadata.json::operational_limits_rad.note`).
- **Acceleration limit không tồn tại**: Tier 4 luôn báo
  `acceleration_status="unavailable"` (mục H).
- **Repeatability tất định**: không nhiễu ngẫu nhiên, không dynamics →
  `RTp` gần 0 là kỳ vọng, không phải bug (mục G).
- **MuJoCo fixed-link fusion**: các link/joint cố định (`dummy, jointPL2,
  jointPL4, jointPL6`) bị gộp vào body cha khi compile URDF→MJCF, không
  còn xuất hiện riêng trong `kr810.xml` (không ảnh hưởng 7-DOF kinematic
  chain) (`assets/ASSET_CONVERSION_REPORT.md` mục "Fixed-joint mapping").
- **Runtime phụ thuộc phần cứng**: `solve_time_ms` không có ý nghĩa tuyệt
  đối giữa các máy khác nhau; chỉ nên so sánh tương đối (warm vs cold,
  cùng máy).
- **Full run có thể tốn nhiều CPU/thời gian**: 289.200 lệnh gọi DLS solve
  (mục C.5) — README ước tính "8 trajectories x up to 360 trial definitions
  x up to 400 waypoints x 2 methods" có thể CPU/time-intensive
  (`README.md` dòng 95-97).
- **100% reachability lúc sinh dữ liệu không phải kết quả evaluation**:
  `validation_waypoint_success_rate=1.0` trong `trajectory_manifest.csv`
  chỉ xác nhận warm-started sequential DLS **lúc generation** giải được mọi
  waypoint ở ngưỡng đó — đây **không** phải kết quả Tier 2 evaluation
  (`trajectories/README.md` mục "Generation-time reachability validation":
  *"This is a data-generation reachability check only, not a Tier 2
  evaluation result."*).
- **Smoke result không phải kết quả nghiên cứu chính thức** (mục A).

**Bất nhất tìm thấy giữa config/code/README/notebook/schema/dataset manifest:**

1. **`DATASET_MANIFEST.json` lỗi thời so với code hiện tại.** File này còn
   ghi `"status": "scaffold"` và
   `"algorithms/, evaluation/, and pipelines/ remain unimplemented; no Tier
   0-4 evaluation has been run"` — nhưng `algorithms/`, `evaluation/`,
   `pipelines/` hiện đã **được triển khai đầy đủ** (xác nhận qua toàn bộ
   mục D-H ở trên, và lịch sử commit `6f80bb2 Stage 6: implement Tier0-Tier4
   evaluation pipelines`, `ac506de Stage 5: implement DLS algorithms and
   Tier1-Tier4 metrics`). `DATASET_MANIFEST.json` chưa được cập nhật kể từ
   Stage 4 — cần đọc theo code hiện tại (mục D-J ở trên), không theo văn bản
   cũ này, khi đánh giá trạng thái triển khai.
2. **Status vocabulary khác nhau giữa `run_manifest.json` và
   `FINAL_SUMMARY.json`** cho cùng tình huống "Tier 2 không có dữ liệu":
   `run_manifest.json::tiers.tier3/tier4.status="skipped"` (có `reason`)
   trong khi `FINAL_SUMMARY.json::tier3/tier4.status="not_run"` (mục K).
3. **`configs/dls_config.json::stop_on_nan`** tồn tại trong config nhưng
   **không được code tham chiếu ở đâu cả** — kiểm tra NaN được thực hiện vô
   điều kiện qua `np.isfinite`, không phụ thuộc key này (mục E).
4. **`requirements.txt` liệt kê `jsonschema>=4.20,<5`** nhưng
   `pipelines/_output_validation.py` không dùng `jsonschema.validate` để
   kiểm tra `run_manifest.json`/`FINAL_SUMMARY.json` theo
   `schemas/*.json` — chỉ kiểm tra JSON/CSV parse được và các key bắt buộc
   thủ công; các file `schemas/*.json` mô tả cấu trúc nhưng không được
   dùng làm validator tự động trong pipeline.
5. **`kinematic_position_max_target_mm`** có trong
   `configs/evaluation_config.json` và `DATASET_MANIFEST.json::acceptance_criteria`
   nhưng **không** được biến thành một `acceptance_criteria` entry thật
   trong `FINAL_SUMMARY.json` (chỉ hiển thị, không so sánh threshold) —
   xem mục I.
6. **Tier 0 gate không bao gồm FK non-determinism** dù giá trị này được
   tính và báo cáo (`fk_determinism_failures`) — chỉ 4/5 điều kiện kiểm
   tra thực sự gate (mục D).

Ngoài các mục trên, **không phát hiện bất nhất nào khác có ảnh hưởng**
giữa hành vi thực thi của code (`pipelines/`, `algorithms/`, `evaluation/`)
và mô tả cấp cao trong `README.md`, `benchmarks/README.md`,
`trajectories/README.md`, `notebooks/README.md` — các tài liệu này khớp
với code ở mức chi tiết đã kiểm tra.

---

## P. Full run expected workload và thời gian

(Xem công thức đầy đủ ở mục C.5.) Tóm tắt:

| Đại lượng | Giá trị | Nguồn |
|---|---|---|
| Số point solve (Tier 1) | 1.200 | `1200 sample × 1` |
| Số waypoint solve — warm-start | 144.000 | `360 trial × 400 waypoint` |
| Số waypoint solve — cold-start | 144.000 | `360 trial × 400 waypoint` |
| Tổng waypoint solve (Tier 2) | 288.000 | `360 × 2 × 400` |
| **Tổng DLS solve calls** | **289.200** | `1.200 + 288.000` |
| Số row `point_results.csv` dự kiến | 1.200 | 1 row/sample |
| Số row `waypoint_results.csv` dự kiến (nếu không có fatal failure) | ≤288.000 | 1 row/waypoint solve; **có thể thấp hơn** nếu có trial gặp non-finite failure |
| Số row `trajectory_trial_summaries.csv`/`warm_vs_cold.csv` dự kiến | 720 / 360 | 1 row/(trial,method) và 1 row/trial ghép cặp |
| Số row `trajectory_metrics.csv`/`cross_track_metrics.csv` (Tier 3) | 720 mỗi file | 1 row/(trial,method) |
| Số row `iso9283_metrics.csv` (Tier 3) | ≤48 (`estimated`, phụ thuộc số nhóm đủ ≥2 repeat) | `8 trajectory × 2 method × 3 speed_scale`, chỉ trial `repeatability` |
| Số row `confidence_intervals.csv` (Tier 3) | 4 | `2 method × 2 metric` |
| Số row `smoothness/joint_feasibility/singularity_path/runtime_metrics.csv` (Tier 4) | 720 mỗi file | 1 row/(trial,method) |

**Ước tính thời gian chạy: KHÔNG có sẵn.** Repository hiện tại không chứa
bất kỳ `run_manifest.json`/`FINAL_SUMMARY.json`/log runtime nào từ một lần
chạy smoke hay full trước đó (`results/`, `temporary_results/` trống, không
tìm thấy `run_manifest.json` hay `FINAL_SUMMARY.json` nào trong repo ngoài
`schemas/`). Do đó tài liệu này **không đưa ra** con số thời gian tuyệt đối
hay khoảng ước lượng — làm vậy sẽ là bịa số. `README.md` chỉ mô tả định
tính: *"can be CPU- and time-intensive... does not require a GPU and never
renders MuJoCo"* (dòng 95-97). Muốn có ước tính thời gian thật, cần chạy
Full (hoặc ít nhất một phần) và đo `solve_time_ms` thực tế trên phần cứng
mục tiêu — nằm ngoài phạm vi audit này.

---

## Q. Bảng Source Traceability

| Topic | Source files/functions/config keys |
|---|---|
| Dataset selection (trial/point/trajectory/waypoint limit) | `pipelines/_common.py::select_trials, build_resolved_config, _as_optional_int, _as_optional_str_list`; `configs/experiment_presets.json` |
| Tier 0 | `pipelines/run_tier0_kinematics.py::run_tier0, main`; `evaluation/kinematics_validation.py::validate_fk_states, validate_jacobian_states, validate_singularity_states, compute_gate_result`; `kinematics/forward_kinematics.py, jacobian.py, rotation_utils.py, quaternion_utils.py, singularity_metrics.py, model_loader.py`; `configs/dls_config.json::singularity_sigma_threshold` |
| Tier 1 | `pipelines/run_tier1_point_dls.py::run_tier1`; `algorithms/point_dls.py::run_point_dls, _solve_one_sample`; `kinematics/dls_solver.py::solve_dls_until_converged, dls_single_update`; `kinematics/adaptive_damping.py, joint_limit_utils.py`; `evaluation/point_ik_metrics.py::compute_point_ik_metrics, _summarize_group, _bin_by_sigma_min`; `evaluation/confidence_intervals.py::wilson_confidence_interval`; `configs/dls_config.json, configs/benchmark_config.json` |
| Tier 2 | `pipelines/run_tier2_sequential_dls.py::run_tier2, _build_trial_summary, _build_warm_vs_cold`; `algorithms/sequential_dls.py::run_sequential_trial, _build_waypoint_results`; `algorithms/warm_start_dls.py::run_warm_start_dls, _recover_q_initial`; `algorithms/cold_start_dls.py::run_cold_start_dls`; `evaluation/waypoint_metrics.py::compute_waypoint_metrics`; `evaluation/runtime_metrics.py::compute_runtime_metrics`; `trajectories/trajectory_trials.csv` |
| Tier 3 | `pipelines/run_tier3_trajectory_tracking.py::run_tier3`; `evaluation/trajectory_metrics.py::compute_position_tracking_metrics, compute_trajectory_tracking_metrics`; `evaluation/cross_track_metrics.py::compute_cross_track_metrics, project_point_to_polyline`; `evaluation/orientation_metrics.py::summarize_orientation_errors, geodesic_angles_from_quaternions`; `evaluation/iso9283_metrics.py::compute_path_accuracy, compute_path_repeatability, MIN_REPEATS_FOR_STD`; `evaluation/confidence_intervals.py::bootstrap_confidence_interval`; `trajectories/trajectory_manifest.csv` |
| Tier 4 | `pipelines/run_tier4_joint_feasibility.py::run_tier4`; `evaluation/smoothness_metrics.py::compute_smoothness_metrics, _gradient`; `evaluation/joint_feasibility_metrics.py::compute_joint_feasibility_metrics`; `evaluation/runtime_metrics.py::compute_runtime_metrics`; `configs/robot_config.json::velocity_limits_rad_s, operational_lower_rad, operational_upper_rad` |
| Acceptance | `pipelines/run_tier0_to_tier4.py::_build_final_summary, _acceptance_entry`; `configs/evaluation_config.json`; `schemas/summary_schema.json::$defs.acceptance_criterion` |
| Status | `pipelines/run_tier0_to_tier4.py::main, _build_final_summary`; `schemas/summary_schema.json`; `schemas/run_manifest_schema.json::$defs.tier_state` |
| Resume | `pipelines/run_tier0_to_tier4.py::_load_previous_manifest, _tier_output_valid, _can_reuse_tier`; `pipelines/_output_validation.py::validate_json_file, validate_csv_file`; `pipelines/_common.py::canonical_signature` |
| Result packaging | `notebooks/KR810_Tier0_Tier4_Kaggle_Template.ipynb` (cell "Package results"); `scripts/package_kaggle_release.py::build_zip, validate_zip` (đóng gói dataset nguồn, không phải kết quả run); `utils/safe_zip.py::safe_extract_zip, is_unsafe_member_name` |
| Notebook invocation | `notebooks/KR810_Tier0_Tier4_Kaggle_Template.ipynb` (cell "USER CONFIGURATION", cell chuyển `argv`, cell `subprocess.run([...,"pipelines.run_tier0_to_tier4",*argv])`); `notebooks/README.md` mục "Cell 2 configuration -> CLI override mapping" |

---

*Tài liệu này chỉ mô tả pipeline và dataset hiện có trong repository tại
thời điểm audit; nó không chứa bất kỳ kết quả Full run giả định nào và
không kết luận DLS "đạt" hay "chưa đạt" tiêu chí dự án — điều đó chỉ có thể
xác định bằng cách thực sự chạy Full theo mục B và đọc `FINAL_SUMMARY.json`
thật.*
