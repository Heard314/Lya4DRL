"""5 servers, 20 devices — proposed method (virtual queue reward)"""
import sys, os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
os.chdir(_PROJ_ROOT)
sys.path.insert(0, _PROJ_ROOT)

from config.params import get_general_params
from controller import Controller

sys.argv = [
    "main",
    "--train_mode", "maddpg",
    "--run_desc", "scale_5s20d_my_exp",
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
    "--edge_server_num", "5",
    "--device_num", "20",
    "--device_type_num", "3",
    # network dims (S=5, device_num=20)
    "--device_obs_dim", "10",
    "--edge_queue_obs_dim", "10",
    "--action_dim", "8",
    "--value_input_obs_dim", "400",
    "--value_input_act_dim", "160",
    "--value_input_dims", "560",
    "--policy_input_dim", "20",
]

gen_params = get_general_params()
gen_params.device_types = [0]*4 + [1]*8 + [2]*8
gen_params.device_in_types = [
    list(range(0, 4)),
    list(range(4, 12)),
    list(range(12, 20)),
]
gen_params.device_num_per_type = [4, 8, 8]

ctr = Controller(gen_params)
ctr.train()
