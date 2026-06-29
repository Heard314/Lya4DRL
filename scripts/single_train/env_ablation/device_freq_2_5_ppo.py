"""1 server, 10 devices — MAPPO device_freq=2.5"""
import sys, os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
os.chdir(_PROJ_ROOT)
sys.path.insert(0, _PROJ_ROOT)

from config.params import get_general_params
from controller import Controller

sys.argv = [
    "main",
    "--train_mode", "mappo",
    "--run_desc", "single_1s10d_device_freq_2_5_ppo",
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
    "--target_reward_penalty", "-80",
    # single server
    "--edge_server_num", "1",
    # network dims (S=1, device_num=10)
    "--device_obs_dim", "6",
    "--edge_queue_obs_dim", "2",
    "--action_dim", "4",
    "--value_input_dims", "80",
    "--policy_input_dim", "8",
]

gen_params = get_general_params()
gen_params.device_types = [0]*2 + [1]*4 + [2]*4
gen_params.device_in_types = [
    list(range(0, 2)),
    list(range(2, 6)),
    list(range(6, 10)),
]
gen_params.device_num_per_type = [2, 4, 4]
gen_params.data_size_inls = [[0.0, 4.0], [0.0, 2.0], [0.0, 1.0]]

gen_params.device_comp_freqs = [2.5, 2.5, 2.5]

ctr = Controller(gen_params)
ctr.train()
