"""
Full trace test script: runs MAPPO and MADDPG for 4 episodes each with
detailed per-slot variable tracing enabled. Trace output goes to:
  experiments/log/train/<algo>_s_<seed>_t_<timestamp>_d_trace_test_trace.log

Usage: python scripts/run_trace_tests.py
"""
import sys, os, traceback

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_SCRIPT_DIR)
os.chdir(_PROJ_ROOT)
sys.path.insert(0, _PROJ_ROOT)

import config.global_params as gp

# Override exp_result_dir for local Windows testing
RESULT_DIR = os.path.join(_PROJ_ROOT, "experiments") + "/"
gp.settings.exp_result_dir = RESULT_DIR

print(f"exp_result_dir set to: {RESULT_DIR}", flush=True)


def run_test(algo, seed):
    """Run one algorithm for 4 episodes with trace enabled."""
    sys.argv = [
        "test",
        "--train_mode", algo,
        "--run_desc", "trace_test",
        "--train_seed", str(seed),
        "--train_episodes", "4",
        "--train_time_slots", "3000",
        "--enable_trace",
    ]

    print(f"\n{'='*70}", flush=True)
    print(f"Starting {algo.upper()} trace test (4 episodes, seed={seed})", flush=True)
    print(f"{'='*70}", flush=True)

    from config.params import get_general_params
    gen_params = get_general_params()

    from controller import Controller
    ctr = Controller(gen_params)

    ctr.train()
    print(f"{algo.upper()} trace test completed.", flush=True)


if __name__ == "__main__":
    try:
        run_test("mappo", 42)
        run_test("maddpg", 7878)
        print("\nAll trace tests passed!", flush=True)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
