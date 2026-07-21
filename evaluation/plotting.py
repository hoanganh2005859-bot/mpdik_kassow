"""Plotting helpers for Tier 0-4 evaluation reports (error distributions, trajectory overlays).

Headless matplotlib (Agg backend, set before pyplot is imported) -- never calls plt.show(), never
uses seaborn, never hardcodes colors/styles (the default matplotlib color cycle is used
throughout so figures stay consistent with whatever rcParams the caller has configured). Every
function saves exactly one figure to a caller-supplied pathlib.Path, creates the parent
directory if needed, and closes the figure afterward so repeated calls do not leak memory.
"""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

PathLike = Union[str, Path]


def _save_and_close(fig, save_path: PathLike) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def _title(base: str, title: Optional[str]) -> str:
    return f"{base} - {title}" if title else base


def plot_position_error_cdf(errors_m: np.ndarray, save_path: PathLike, title: Optional[str] = None) -> Path:
    """Empirical CDF of Cartesian position error."""
    errors_m = np.sort(np.asarray(errors_m, dtype=np.float64))
    cdf = np.arange(1, errors_m.shape[0] + 1) / errors_m.shape[0]

    fig, ax = plt.subplots()
    ax.plot(errors_m, cdf)
    ax.set_xlabel("Position error (m)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(_title("Position error CDF", title))
    ax.grid(True)
    return _save_and_close(fig, save_path)


def plot_orientation_error_cdf(errors_deg: np.ndarray, save_path: PathLike, title: Optional[str] = None) -> Path:
    """Empirical CDF of SO(3) geodesic orientation error."""
    errors_deg = np.sort(np.asarray(errors_deg, dtype=np.float64))
    cdf = np.arange(1, errors_deg.shape[0] + 1) / errors_deg.shape[0]

    fig, ax = plt.subplots()
    ax.plot(errors_deg, cdf)
    ax.set_xlabel("Orientation error (deg)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(_title("Orientation error CDF", title))
    ax.grid(True)
    return _save_and_close(fig, save_path)


def plot_iterations_histogram(iterations: np.ndarray, save_path: PathLike, title: Optional[str] = None) -> Path:
    """Histogram of DLS iteration counts."""
    iterations = np.asarray(iterations)

    fig, ax = plt.subplots()
    ax.hist(iterations, bins="auto")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Count")
    ax.set_title(_title("Iteration count distribution", title))
    return _save_and_close(fig, save_path)


def plot_runtime_histogram(solve_times_ms: np.ndarray, save_path: PathLike, title: Optional[str] = None) -> Path:
    """Histogram of DLS per-sample solve time (milliseconds)."""
    solve_times_ms = np.asarray(solve_times_ms, dtype=np.float64)

    fig, ax = plt.subplots()
    ax.hist(solve_times_ms, bins="auto")
    ax.set_xlabel("Solve time (ms)")
    ax.set_ylabel("Count")
    ax.set_title(_title("Runtime distribution", title))
    return _save_and_close(fig, save_path)


def plot_success_rate_by_group_bar(
    group_labels: Sequence[str], success_rates: Sequence[float], save_path: PathLike, title: Optional[str] = None
) -> Path:
    """Bar chart of success rate per named group (e.g. Tier 1 difficulty groups)."""
    success_rates = np.asarray(success_rates, dtype=np.float64)
    if success_rates.shape[0] != len(group_labels):
        raise ValueError("group_labels and success_rates must have the same length")

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(group_labels)), success_rates)
    ax.set_xticks(np.arange(len(group_labels)))
    ax.set_xticklabels(group_labels, rotation=30, ha="right")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(_title("Success rate by group", title))
    ax.grid(True, axis="y")
    return _save_and_close(fig, save_path)


def plot_target_vs_actual_3d(
    target_positions: np.ndarray, actual_positions: np.ndarray, save_path: PathLike, title: Optional[str] = None
) -> Path:
    """3D overlay of the commanded path vs. the achieved end-effector path."""
    target_positions = np.asarray(target_positions, dtype=np.float64)
    actual_positions = np.asarray(actual_positions, dtype=np.float64)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], label="target")
    ax.plot(actual_positions[:, 0], actual_positions[:, 1], actual_positions[:, 2], label="actual")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(_title("Target vs. actual path", title))
    ax.legend()
    return _save_and_close(fig, save_path)


def plot_xyz_tracking(
    time_s: np.ndarray, target_positions: np.ndarray, actual_positions: np.ndarray, save_path: PathLike, title: Optional[str] = None
) -> Path:
    """Per-axis (x, y, z) target-vs-actual tracking over time (one subplot per axis)."""
    time_s = np.asarray(time_s, dtype=np.float64)
    target_positions = np.asarray(target_positions, dtype=np.float64)
    actual_positions = np.asarray(actual_positions, dtype=np.float64)

    fig, axes = plt.subplots(3, 1, sharex=True)
    labels = ["x (m)", "y (m)", "z (m)"]
    for i, ax in enumerate(axes):
        ax.plot(time_s, target_positions[:, i], label="target")
        ax.plot(time_s, actual_positions[:, i], label="actual")
        ax.set_ylabel(labels[i])
        ax.grid(True)
    axes[0].legend()
    axes[0].set_title(_title("XYZ tracking", title))
    axes[-1].set_xlabel("Time (s)")
    return _save_and_close(fig, save_path)


def plot_position_error_over_time(
    time_s: np.ndarray, position_errors_m: np.ndarray, save_path: PathLike, title: Optional[str] = None
) -> Path:
    """Cartesian position error vs. time."""
    fig, ax = plt.subplots()
    ax.plot(np.asarray(time_s, dtype=np.float64), np.asarray(position_errors_m, dtype=np.float64))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (m)")
    ax.set_title(_title("Position error over time", title))
    ax.grid(True)
    return _save_and_close(fig, save_path)


def plot_orientation_error_over_time(
    time_s: np.ndarray, orientation_errors_deg: np.ndarray, save_path: PathLike, title: Optional[str] = None
) -> Path:
    """SO(3) geodesic orientation error vs. time."""
    fig, ax = plt.subplots()
    ax.plot(np.asarray(time_s, dtype=np.float64), np.asarray(orientation_errors_deg, dtype=np.float64))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Orientation error (deg)")
    ax.set_title(_title("Orientation error over time", title))
    ax.grid(True)
    return _save_and_close(fig, save_path)


def _plot_joint_series(time_s, series, ylabel: str, base_title: str, save_path: PathLike, title: Optional[str]) -> Path:
    time_s = np.asarray(time_s, dtype=np.float64)
    series = np.asarray(series, dtype=np.float64)

    fig, ax = plt.subplots()
    for j in range(series.shape[1]):
        ax.plot(time_s, series[:, j], label=f"joint_{j + 1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(_title(base_title, title))
    ax.legend(ncol=2, fontsize="small")
    ax.grid(True)
    return _save_and_close(fig, save_path)


def plot_joint_trajectory(time_s, q_trajectory, save_path: PathLike, title: Optional[str] = None) -> Path:
    """Joint position q(t) for all joints on one axes."""
    return _plot_joint_series(time_s, q_trajectory, "Joint position (rad)", "Joint trajectory", save_path, title)


def plot_joint_velocity(time_s, joint_velocity, save_path: PathLike, title: Optional[str] = None) -> Path:
    """Joint velocity dq/dt for all joints on one axes."""
    return _plot_joint_series(time_s, joint_velocity, "Joint velocity (rad/s)", "Joint velocity", save_path, title)


def plot_joint_acceleration(time_s, joint_acceleration, save_path: PathLike, title: Optional[str] = None) -> Path:
    """Joint acceleration d2q/dt2 for all joints on one axes."""
    return _plot_joint_series(
        time_s, joint_acceleration, "Joint acceleration (rad/s^2)", "Joint acceleration", save_path, title
    )


def plot_joint_jerk(time_s, joint_jerk, save_path: PathLike, title: Optional[str] = None) -> Path:
    """Joint jerk d3q/dt3 for all joints on one axes."""
    return _plot_joint_series(time_s, joint_jerk, "Joint jerk (rad/s^3)", "Joint jerk", save_path, title)


def plot_sigma_min_over_time(
    time_s: np.ndarray, sigma_min_series: np.ndarray, save_path: PathLike, title: Optional[str] = None
) -> Path:
    """Smallest Jacobian singular value (singularity proximity) vs. time."""
    fig, ax = plt.subplots()
    ax.plot(np.asarray(time_s, dtype=np.float64), np.asarray(sigma_min_series, dtype=np.float64))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("sigma_min")
    ax.set_title(_title("Minimum singular value over time", title))
    ax.grid(True)
    return _save_and_close(fig, save_path)


def plot_warm_vs_cold_summary(
    warm_values: Sequence[float],
    cold_values: Sequence[float],
    labels: Sequence[str],
    save_path: PathLike,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> Path:
    """Grouped bar chart comparing a warm-start vs. cold-start metric across ``labels`` categories."""
    warm_values = np.asarray(warm_values, dtype=np.float64)
    cold_values = np.asarray(cold_values, dtype=np.float64)
    if warm_values.shape != cold_values.shape or warm_values.shape[0] != len(labels):
        raise ValueError("warm_values, cold_values, and labels must all have the same length")

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, warm_values, width, label="warm_start")
    ax.bar(x + width / 2, cold_values, width, label="cold_start")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(_title("Warm-start vs. cold-start", title))
    ax.legend()
    return _save_and_close(fig, save_path)
