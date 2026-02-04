import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))

@dataclass
class ScalarSeries:
    steps: np.ndarray
    wall_time: np.ndarray
    values: np.ndarray


def _find_event_files(path: str) -> List[str]:
    """
    Find TensorBoard event files under a file path or directory.
    """
    if os.path.isfile(path):
        return [path]

    pattern = os.path.join(path, "**", "events.out.tfevents.*")
    files = glob.glob(pattern, recursive=True)
    files.sort()
    return files


def load_scalars_from_event_files(
    path: str,
    tags: Optional[List[str]] = None,
    size_guidance_scalars: int = 0,
) -> Dict[str, ScalarSeries]:
    """
    Load scalar series from a TensorBoard event file or a log directory.

    path:
        A single event file path OR a directory containing event files.
    tags:
        If provided, only load these scalar tags.
    size_guidance_scalars:
        0 means load all scalar points. You may set a positive number to limit memory.
    """
    event_files = _find_event_files(path)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {path}")

    merged: Dict[str, List[Tuple[int, float, float]]] = {}
    # Each point is (step, wall_time, value)

    for f in event_files:
        ea = EventAccumulator(
            f,
            size_guidance={
                "scalars": size_guidance_scalars,
            },
        )
        ea.Reload()

        all_tags = ea.Tags().get("scalars", [])
        use_tags = all_tags if tags is None else [t for t in tags if t in all_tags]

        for tag in use_tags:
            events = ea.Scalars(tag)
            if tag not in merged:
                merged[tag] = []
            for e in events:
                merged[tag].append((int(e.step), float(e.wall_time), float(e.value)))

    # Sort and convert to numpy arrays
    out: Dict[str, ScalarSeries] = {}
    for tag, points in merged.items():
        points.sort(key=lambda x: x[0])  # sort by step
        steps = np.array([p[0] for p in points], dtype=np.int64)
        wall_time = np.array([p[1] for p in points], dtype=np.float64)
        values = np.array([p[2] for p in points], dtype=np.float64)
        out[tag] = ScalarSeries(steps=steps, wall_time=wall_time, values=values)

    return out


def _make_x_axis(series: ScalarSeries, x_axis: str) -> np.ndarray:
    """
    Build x-axis array consistent with TensorBoard common choices.
    x_axis: "step" | "episode" | "wall_time" | "relative_time"
    """
    if x_axis in ["step", "episode"]:
        return series.steps
    if x_axis == "wall_time":
        return series.wall_time
    if x_axis == "relative_time":
        return series.wall_time - series.wall_time.min()
    raise ValueError(f"Unsupported x_axis: {x_axis}")


def ema_smooth(y: np.ndarray, weight: float) -> np.ndarray:
    """
    Exponential moving average smoothing similar to common dashboard smoothing.
    weight in [0, 1). Larger means smoother.
    """
    if weight <= 0:
        return y.copy()

    y_s = np.empty_like(y, dtype=np.float64)
    last = y[0]
    y_s[0] = last
    for i in range(1, len(y)):
        last = weight * last + (1.0 - weight) * y[i]
        y_s[i] = last
    return y_s


def plot_scalar_tag_from_runs(
    runs: List[Tuple[str, str]],
    tag: str,
    x_axis: str = "step",
    smooth_weight: float = 0.0,
    show_raw: bool = True,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot the same scalar tag from multiple runs on one figure.
    """
    plt.figure(figsize=figsize)

    for label, path in runs:
        scalars = load_scalars_from_event_files(path, tags=[tag])
        if tag not in scalars:
            print(f"[Warn] tag '{tag}' not found in: {path}")
            continue

        series = scalars[tag]
        x = _make_x_axis(series, x_axis)
        y = series.values

        if smooth_weight > 0:
            y_s = ema_smooth(y, smooth_weight)
            if show_raw:
                plt.plot(x, y, alpha=0.25, label=f"{label} (raw)")
            plt.plot(x, y_s, label=f"{label} (smoothed={smooth_weight})")
        else:
            plt.plot(x, y, label=label)

    if x_axis == "step":
        plt.xlabel("step")
    elif x_axis == "wall_time":
        plt.xlabel("wall_time (unix seconds)")
    else:
        plt.xlabel("relative_time (seconds)")

    plt.ylabel(tag)
    plt.title(f"TensorBoard Scalars: {tag}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def list_scalar_tags(path: str) -> List[str]:
    """
    List all scalar tags under a logdir or event file.
    """
    event_files = _find_event_files(path)
    if not event_files:
        return []

    tags = set()
    for f in event_files:
        ea = EventAccumulator(f)
        ea.Reload()
        for t in ea.Tags().get("scalars", []):
            tags.add(t)

    return sorted(tags)


def _aggregate_mean_by_step(series_list: List[ScalarSeries]) -> ScalarSeries:
    """
    Aggregate multiple ScalarSeries by step and compute mean value for each step.
    """
    if not series_list:
        raise ValueError("series_list is empty")

    step_to_values: Dict[int, List[float]] = {}
    step_to_times: Dict[int, List[float]] = {}

    for s in series_list:
        for st, wt, val in zip(s.steps.tolist(), s.wall_time.tolist(), s.values.tolist()):
            if st not in step_to_values:
                step_to_values[st] = []
                step_to_times[st] = []
            step_to_values[st].append(float(val))
            step_to_times[st].append(float(wt))

    steps_sorted = sorted(step_to_values.keys())
    mean_vals = [float(np.mean(step_to_values[st])) for st in steps_sorted]
    mean_times = [float(np.mean(step_to_times[st])) for st in steps_sorted]

    return ScalarSeries(
        steps=np.array(steps_sorted, dtype=np.int64),
        wall_time=np.array(mean_times, dtype=np.float64),
        values=np.array(mean_vals, dtype=np.float64),
    )


def _downsample_xy(x: np.ndarray, y: np.ndarray, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample x and y by stride.
    """
    if stride <= 1:
        return x, y
    return x[::stride], y[::stride]


def plot_sub_curve(
    paths: List[str],
    tags: List[str],
    x_axis: str = "step",
    smooth_weight: float = 0.0,
    stride: int = 1,
    value_divisor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load scalar curves from multiple paths, average them by step, and return raw and smoothed curves.
    value_divisor: divide y values by this number.
    """
    if value_divisor == 0:
        raise ValueError("value_divisor must be non-zero.")

    series_list: List[ScalarSeries] = []

    for i in range(len(paths)):
        p = paths[i]
        tag_i = tags[i]
        scalars = load_scalars_from_event_files(p, tags=[tag_i])
        if tag_i not in scalars:
            print(f"[Warn] tag '{tag_i}' not found in: {p}")
            continue
        series_list.append(scalars[tag_i])

    if not series_list:
        raise ValueError(f"No valid series found in given paths/tags: {list(zip(paths, tags))}")

    mean_series = _aggregate_mean_by_step(series_list)

    x = _make_x_axis(mean_series, x_axis)
    y_raw = mean_series.values.astype(np.float64)

    # Scale values
    if value_divisor != 1.0:
        y_raw = y_raw / float(value_divisor)

    y_smooth = None
    if smooth_weight > 0:
        y_smooth = ema_smooth(y_raw, smooth_weight)

    if stride > 1:
        x = x[::stride]
        y_raw = y_raw[::stride]
        if y_smooth is not None:
            y_smooth = y_smooth[::stride]

    return x, y_raw, y_smooth


def plot_exp_plot(
    curves: List[Dict[str, Any]],
    x_axis: str = "step",
    xlabel: Optional[str] = None,
    ylabel: str = "value",
    title: str = "Experiment Plot",
    smooth_weight: float = 0.0,
    stride: int = 1,
    figsize: Tuple[int, int] = (10, 6),
    grid: bool = True,
    show_raw: bool = True,
    raw_alpha: float = 0.25,
    raw_in_legend: bool = False,
    smooth_linewidth: float = 2.0,
    x_major_tick: Optional[float] = 1000.0,
    value_divisor: float = 1.0,
    color_map: str = "tab10",
    colors: Optional[List[Any]] = None,
):
    """
    Plot multiple sub-curves on one figure.
    color_map: matplotlib colormap name, e.g., "tab10", "tab20", "Set1"
    colors: optional list of colors, override color_map if provided
    """
    if value_divisor == 0:
        raise ValueError("value_divisor must be non-zero.")

    plt.figure(figsize=figsize)
    ax = plt.gca()

    cmap = plt.get_cmap(color_map)

    for i, cfg in enumerate(curves):
        label = cfg["label"]
        paths = cfg["paths"]
        tags = cfg["tags"]

        cfg_smooth = cfg.get("smooth_weight", smooth_weight)
        cfg_divisor = cfg.get("value_divisor", value_divisor)

        # Pick a distinct color for each curve
        if "color" in cfg:
            color = cfg["color"]
        elif colors is not None and i < len(colors):
            color = colors[i]
        else:
            color = cmap(i % getattr(cmap, "N", 10))

        x, y_raw, y_smooth = plot_sub_curve(
            paths=paths,
            tags=tags,
            x_axis=x_axis,
            smooth_weight=cfg_smooth,
            stride=stride,
            value_divisor=cfg_divisor,
        )

        if show_raw:
            raw_label = f"{label} (raw)" if (raw_in_legend and cfg_smooth > 0) else (label if raw_in_legend else "_nolegend_")
            plt.plot(x, y_raw, alpha=raw_alpha, label=raw_label, color=color)

        if y_smooth is not None:
            plt.plot(x, y_smooth, linewidth=smooth_linewidth, label=label, color=color)

    if xlabel is None:
        if x_axis in ["step", "episode"]:
            xlabel = x_axis
        elif x_axis == "wall_time":
            xlabel = "wall_time (unix seconds)"
        else:
            xlabel = "relative_time (seconds)"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if grid:
        plt.grid(True, alpha=0.3)

    if x_axis in ["step", "episode"] and x_major_tick is not None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_major_tick))
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_dly():

    plot_paths = [
        r"./runs/evaluate/mappo_s_7878_t_2026-01-20-12-02-31-506743_d_all_queue_exp9",
        r"./runs/evaluate/mappo_s_2345_t_2026-01-20-10-59-17-016009_d_all_queue_exp9",    
    ]

    curves_cfg = [
        {
            "label": "mappo",
            "paths": [
                os.path.join(plot_paths[0], "detail_comp_dly_0_ep_20"),
                os.path.join(plot_paths[0], "detail_comp_dly_1_ep_20"),
            ],
            "tags": "detail/comp_dly_0",
            "marker": "o",
        },

        {
            "label": "maddpg",
            "paths": [
                os.path.join(plot_paths[1], "detail_comp_dly_0_ep_20"),
                os.path.join(plot_paths[1], "detail_comp_dly_1_ep_40"),
            ],
            "tags": "detail/comp_dly_0",
            "marker": "*",
        },
    ]

def plot_timeout():
    plot_paths = [
        r"plot_data/target/maddpg_s_7878_t_2026-02-02-11-03-56-032697_d_my_exp9",
        r"plot_data/target/maddpg_s_7878_t_2026-02-02-21-25-08-707599_d_rt_exp10",
        r"plot_data/target/maddpg_s_7878_t_2026-02-03-00-02-45-332873_d_no_queue_exp11",
    ]

    curves_cfg = [
        {
            "label": "LyaHDDPG",
            "paths": [
                plot_paths[0]
            ],
            "tags":  [
                "device_overtime_nums_0",
            ],
            "marker": "o",
            "color": "C0"
        },

        {
            "label": "RT-DDPG",
            "paths": [
                plot_paths[1]
            ],
            "tags":  [
                "device_overtime_nums_0",
            ],
            "marker": "*",
            "color": "C3"
        },

        {
            "label": "Navie",
            "paths": [
                plot_paths[2]
            ],
            "tags":  [
                "device_overtime_nums_0",
            ],
            "marker": "*",
            "color": "C2"
        },
    ]

    plot_exp_plot(
        curves=curves_cfg,
        x_axis="episode",
        xlabel="Episodes",
        ylabel="Timeout Rate(%)",
        title="Timeout rate Comparison Of type-0 tasks",
        smooth_weight=0.95,
        stride=50,
        show_raw=True,
        raw_alpha=0.2,
        x_major_tick=1000,
        value_divisor=600.0,
    )

    curves_cfg = [
        {
            "label": "LyaHDDPG",
            "paths": [
                plot_paths[0]
            ],
            "tags":  [
                "device_overtime_nums_1",
            ],
            "marker": "o",
            "color": "C0"
        },

        {
            "label": "RT-DDPG",
            "paths": [
                plot_paths[1]
            ],
            "tags":  [
                "device_overtime_nums_1",
            ],
            "marker": "*",
            "color": "C3"
        },

        {
            "label": "Navie",
            "paths": [
                plot_paths[2]
            ],
            "tags":  [
                "device_overtime_nums_1",
            ],
            "marker": "*",
            "color": "C2"
        },
    ]

    plot_exp_plot(
        curves=curves_cfg,
        x_axis="episode",
        xlabel="Episodes",
        ylabel="Timeout Rate(%)",
        title="Timeout rate Comparison Of type-1 tasks",
        smooth_weight=0.95,
        stride=50,
        show_raw=True,
        raw_alpha=0.2,
        x_major_tick=1000,
        value_divisor=600.0,
    )

    curves_cfg = [
        {
            "label": "LyaHDDPG",
            "paths": [
                plot_paths[0]
            ],
            "tags":  [
                "device_overtime_nums_2",
            ],
            "marker": "o",
            "color": "C0"
        },

        {
            "label": "RT-DDPG",
            "paths": [
                plot_paths[1]
            ],
            "tags":  [
                "device_overtime_nums_2",
            ],
            "marker": "*",
            "color": "C3"
        },

        {
            "label": "Navie",
            "paths": [
                plot_paths[2]
            ],
            "tags":  [
                "device_overtime_nums_2",
            ],
            "marker": "*",
            "color": "C2"
        },
    ]

    plot_exp_plot(
        curves=curves_cfg,
        x_axis="episode",
        xlabel="Episodes",
        ylabel="Timeout Rate(%)",
        title="Timeout rate Comparison Of type-2 tasks",
        smooth_weight=0.95,
        stride=50,
        show_raw=True,
        raw_alpha=0.2,
        x_major_tick=1000,
        value_divisor=600.0,
    )

def plot_engy():
    
    plot_paths = [
        r"plot_data/target/maddpg_s_7878_t_2026-02-02-11-03-56-032697_d_my_exp9",
        r"plot_data/target/maddpg_s_7878_t_2026-02-02-21-25-08-707599_d_rt_exp10",
        r"plot_data/target/maddpg_s_7878_t_2026-02-03-00-02-45-332873_d_no_queue_exp11",
    ]

    curves_cfg = [
        {
            "label": "LyaHDDPG",
            "paths": [
                plot_paths[0]
            ],
            "tags":  [
                "device_cost_0",
            ],
            "marker": "o",
        },

        {
            "label": "RT-DDPG",
            "paths": [
                plot_paths[1]
            ],
            "tags":  [
                "device_cost_0",
            ],
            "marker": "*",
        },

        {
            "label": "Navie",
            "paths": [
                plot_paths[2]
            ],
            "tags":  [
                "device_cost_0",
            ],
            "marker": "*",
        },
    ]

    plot_exp_plot(
        curves=curves_cfg,
        x_axis="step",
        xlabel="time step",
        ylabel="energy consumption",
        title="Energy Consumption Comparison Of type-0 tasks",
        smooth_weight=0.0,
        stride=50,
    )

    plot_paths = [
        r"plot_data/target/maddpg_s_7878_t_2026-02-02-11-03-56-032697_d_my_exp9",
        r"plot_data/target/maddpg_s_7878_t_2026-02-02-21-25-08-707599_d_rt_exp10",
        r"plot_data/target/maddpg_s_7878_t_2026-02-03-00-02-45-332873_d_no_queue_exp11",
    ]

    curves_cfg = [
        {
            "label": "LyaHDDPG",
            "paths": [
                plot_paths[0]
            ],
            "tags":  [
                "device_cost_1",
            ],
            "marker": "o",
        },

        {
            "label": "RT-DDPG",
            "paths": [
                plot_paths[1]
            ],
            "tags":  [
                "device_cost_1",
            ],
            "marker": "*",
        },

        {
            "label": "Navie",
            "paths": [
                plot_paths[2]
            ],
            "tags":  [
                "device_cost_1",
            ],
            "marker": "*",
        },
    ]

    plot_exp_plot(
        curves=curves_cfg,
        x_axis="step",
        xlabel="time step",
        ylabel="energy consumption",
        title="Energy Consumption Comparison Of type-1 tasks",
        smooth_weight=0.0,
        stride=50,
    )

    plot_paths = [
        r"plot_data/target/maddpg_s_7878_t_2026-02-02-11-03-56-032697_d_my_exp9",
        r"plot_data/target/maddpg_s_7878_t_2026-02-02-21-25-08-707599_d_rt_exp10",
        r"plot_data/target/maddpg_s_7878_t_2026-02-03-00-02-45-332873_d_no_queue_exp11",
    ]

    curves_cfg = [
        {
            "label": "LyaHDDPG",
            "paths": [
                plot_paths[0]
            ],
            "tags":  [
                "device_cost_2",
            ],
            "marker": "o",
        },

        {
            "label": "RT-DDPG",
            "paths": [
                plot_paths[1]
            ],
            "tags":  [
                "device_cost_2",
            ],
            "marker": "*",
        },

        {
            "label": "Navie",
            "paths": [
                plot_paths[2]
            ],
            "tags":  [
                "device_cost_2",
            ],
            "marker": "*",
        },
    ]

    plot_exp_plot(
        curves=curves_cfg,
        x_axis="step",
        xlabel="time step",
        ylabel="energy consumption",
        title="Energy Consumption Comparison Of type-2 tasks",
        smooth_weight=0.0,
        stride=50,
    )

def plot_device_queue():
    plot_paths = [
        r"./runs/evaluate/mappo_s_7878_t_2026-01-20-12-02-31-506743_d_all_queue_exp9",
        r"./runs/evaluate/mappo_s_2345_t_2026-01-20-10-59-17-016009_d_all_queue_exp9",    
    ]

    curves_cfg = [
        {
            "label": "mappo",
            "paths": [
                os.path.join(plot_paths[0], "detail_device_time_ql_2_ep_20_act"),
                os.path.join(plot_paths[0], "detail_device_time_ql_3_ep_20_act"),
                os.path.join(plot_paths[0], "detail_device_time_ql_4_ep_20_act"),
                os.path.join(plot_paths[0], "detail_device_time_ql_5_ep_20_act"),
            ],
            "tags":  [
                "detail/device_time_ql_2",
                "detail/device_time_ql_3",
                "detail/device_time_ql_4",
                "detail/device_time_ql_5",
            ],
            "marker": "o",
        },

        {
            "label": "maddpg",
            "paths": [
                os.path.join(plot_paths[1], "detail_device_time_ql_2_ep_20_act"),
                os.path.join(plot_paths[1], "detail_device_time_ql_3_ep_20_act"),
                os.path.join(plot_paths[1], "detail_device_time_ql_4_ep_20_act"),
                os.path.join(plot_paths[1], "detail_device_time_ql_5_ep_20_act"),
            ],
            "tags":  [
                "detail/device_time_ql_2",
                "detail/device_time_ql_3",
                "detail/device_time_ql_4",
                "detail/device_time_ql_5",
            ],
            "marker": "*",
        },
    ]

    plot_exp_plot(
        curves=curves_cfg,
        x_axis="step",
        xlabel="time step",
        ylabel="device queue length(s)",
        title="Device Queue Length Comparison",
        smooth_weight=0.0,
        stride=10,
    )

def plot_edge_queue():
    plot_paths = [
        r"./runs/evaluate/mappo_s_7878_t_2026-01-20-12-02-31-506743_d_all_queue_exp9",
        r"./runs/evaluate/mappo_s_2345_t_2026-01-20-10-59-17-016009_d_all_queue_exp9",    
    ]

    curves_cfg = [
        {
            "label": "mappo",
            "paths": [
                os.path.join(plot_paths[0], "detail_edge_time_ql_1_ep_20_act"),
            ],
            "tags":  [
                "detail/edge_time_ql_1",
            ],
            "marker": "o",
        },

        {
            "label": "maddpg",
            "paths": [
                os.path.join(plot_paths[1], "detail_edge_time_ql_1_ep_20_act"),
            ],
            "tags":  [
                "detail/edge_time_ql_1",
            ],
            "marker": "*",
        },
    ]

    plot_exp_plot(
        curves=curves_cfg,
        x_axis="step",
        xlabel="time step",
        ylabel="edge queue length(s)",
        title="Edge Queue Length Comparison",
        smooth_weight=0.0,
        stride=10,
    )


if __name__ == "__main__":
    
    # plot_paths = [
    #     r"./runs/evaluate/mappo_s_7878_t_2026-01-20-12-02-31-506743_d_all_queue_exp9",
    #     r"./runs/evaluate/mappo_s_2345_t_2026-01-20-10-59-17-016009_d_all_queue_exp9",    
    # ]
    # # List schalar tags
    # result = list_scalar_tags(os.path.join(plot_paths[0], "detail_edge_time_ql_2_ep_20_act"))
    # print("Scalar tags found:")
    # for t in result:
    #     print(f"{t}")

    # plot_dly()

    plot_timeout()

    # plot_engy()

    # plot_device_queue()

    # plot_edge_queue()

