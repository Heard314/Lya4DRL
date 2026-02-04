import os
import re
import glob
import shutil
from typing import Dict, List, Tuple, Optional, Set

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


def _has_event_file(logdir: str) -> bool:
    # Check whether there is any tfevents file in logdir
    return len(glob.glob(os.path.join(logdir, "events.out.tfevents.*"))) > 0


def _find_run_dirs(root: str) -> List[str]:
    # Find tensorboard run directories under root
    if _has_event_file(root):
        return [root]

    run_dirs = []
    for p in glob.glob(os.path.join(root, "*")):
        if os.path.isdir(p) and _has_event_file(p):
            run_dirs.append(p)

    return sorted(run_dirs)


def _load_scalar_series(run_dir: str) -> Dict[str, Dict[int, float]]:
    # Load all scalar tags from a tensorboard run directory
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()

    series: Dict[str, Dict[int, float]] = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        step_to_val: Dict[int, float] = {}
        for e in events:
            # Keep the last value if the same step appears multiple times
            step_to_val[int(e.step)] = float(e.value)
        series[tag] = step_to_val

    return series


def _split_tag_id(tag: str) -> Optional[Tuple[str, int]]:
    # Split "xxx_3" into ("xxx", 3), otherwise return None
    m = re.match(r"^(.*)_(\d+)$", tag)
    if not m:
        return None
    base = m.group(1)
    idx = int(m.group(2))
    return base, idx


def _merge_by_groups_with_passthrough(
    all_series: Dict[str, Dict[int, float]],
    merge_ids: List[List[int]],
    passthrough_bases: Set[str],
    require_all: bool = True,
) -> Dict[str, Dict[int, float]]:
    # Merge tags by merge_ids, and passthrough some bases without merging
    base_to_id_series: Dict[str, Dict[int, Dict[int, float]]] = {}
    output: Dict[str, Dict[int, float]] = {}

    for tag, series in all_series.items():
        sp = _split_tag_id(tag)
        if sp is None:
            # Non-id tags like "joint_cost" are written directly
            output[tag] = series
            continue

        base, idx = sp
        if base in passthrough_bases:
            # Tags like "edge_comp_ql_0" "joint_reward_1" are written directly
            output[tag] = series
            continue

        base_to_id_series.setdefault(base, {})[idx] = series

    # Merge remaining bases by groups
    for base, id_map in base_to_id_series.items():
        for group_idx, ids in enumerate(merge_ids):
            valid_ids = [i for i in ids if i in id_map]
            if not valid_ids:
                continue

            step_sets = [set(id_map[i].keys()) for i in valid_ids]
            steps = set.intersection(*step_sets) if require_all else set.union(*step_sets)

            out_series: Dict[int, float] = {}
            for step in sorted(steps):
                vals = []
                for i in valid_ids:
                    if step in id_map[i]:
                        vals.append(id_map[i][step])

                if require_all and len(vals) != len(valid_ids):
                    continue

                if vals:
                    out_series[step] = sum(vals) / len(vals)

            # Output tag uses group index: base_0 base_1 base_2
            output[f"{base}_{group_idx}"] = out_series

    return output


def merge_plot(plot_path, target_plot_path):

    merge_ids = [
        [0, 1],
        [2, 3, 4, 5],
        [6, 7, 8, 9],
    ]

    # These bases will be written as-is, no merging
    passthrough_bases = {
        "edge_comp_ql",
        "joint_reward",
    }

    require_all = True
    overwrite = True

    run_dirs = _find_run_dirs(plot_path)
    if not run_dirs:
        raise FileNotFoundError(f"No tensorboard event files found under: {plot_path}")

    if overwrite and os.path.exists(target_plot_path):
        shutil.rmtree(target_plot_path)
    os.makedirs(target_plot_path, exist_ok=True)

    # One writer -> one output events file in target_plot_path
    writer = SummaryWriter(log_dir=target_plot_path)

    # If multiple runs exist, prefix tags to avoid collisions
    use_prefix = (len(run_dirs) > 1)

    for run_dir in run_dirs:
        all_series = _load_scalar_series(run_dir)
        merged_and_passthrough = _merge_by_groups_with_passthrough(
            all_series=all_series,
            merge_ids=merge_ids,
            passthrough_bases=passthrough_bases,
            require_all=require_all,
        )

        prefix = os.path.basename(run_dir.rstrip("/")) + "/" if use_prefix else ""

        for tag, series in merged_and_passthrough.items():
            out_tag = prefix + tag
            for step, val in sorted(series.items()):
                writer.add_scalar(out_tag, val, step)

    writer.flush()
    writer.close()

if __name__ == "__main__":
    plot_paths = [
        "./plot_data/maddpg_s_7878_t_2026-02-02-11-03-56-032697_d_my_exp9/total",
        "./plot_data/maddpg_s_7878_t_2026-02-02-21-25-08-707599_d_rt_exp10/total",
        "./plot_data/maddpg_s_7878_t_2026-02-03-00-02-45-332873_d_no_queue_exp11/total",
    ]
    target_plot_path = [
        "./plot_data/target/maddpg_s_7878_t_2026-02-02-11-03-56-032697_d_my_exp9",
        "./plot_data/target/maddpg_s_7878_t_2026-02-02-21-25-08-707599_d_rt_exp10",
        "./plot_data/target/maddpg_s_7878_t_2026-02-03-00-02-45-332873_d_no_queue_exp11",
    ]
    merge_plot(plot_paths[0], target_plot_path[0])
    merge_plot(plot_paths[1], target_plot_path[1])
    merge_plot(plot_paths[2], target_plot_path[2])