"""Edge server computation frequency ablation: 45 Gcycles/s"""
import sys, os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR))))
os.chdir(_PROJ_ROOT)
sys.path.insert(0, _PROJ_ROOT)

from config.params import get_general_params
from controller import Controller

sys.argv = [
    "main",
    "--train_mode", "maddpg",
    "--run_desc", "edge_freq_45",
    "--enable_virtual_queue_reward",
    "--device_vir_queue_reward_weight", "-700",
    "--device_act_queue_reward_weight", "-300",
    "--device_act_queue_reward_max_bound", "200",
    "--device_act_queue_reward_min_bound", "-300",
    "--device_vir_queue_reward_max_bound", "800",
    "--device_vir_queue_reward_min_bound", "-1200",
    "--device_act_queue_growth_rate", "1",
    "--device_vir_queue_growth_rate", "0.1",
    "--edge_act_queue_growth_rate", "1",
    "--edge_vir_queue_growth_rate", "0.1",
    "--timeout_reward_penalty", "-4000",
    "--target_reward_penalty", "-2000",
]

gen_params = get_general_params()
gen_params.edge_comp_freq = 45

ctr = Controller(gen_params)
ctr.train()
