import argparse
from argparse import BooleanOptionalAction
from multiprocessing.connection import deliver_challenge

from torch import device

"""
general params
"""
device_num = 10
edge_queue_num = 3
gen_task_cycle = 5
ql_punish_fac = 100.0
def get_general_params():
    parser = argparse.ArgumentParser(description = "general params")

    parser.add_argument("--enable_actual_queue_reward", action="store_true",
                        help = "enable when the algorithm used is the RT-MADDPG method")

    parser.add_argument("--enable_virtual_queue_reward", action="store_true",
                        help = "enabled when the algorithm used is paper's proposed method")

    parser.add_argument("--run_desc", type = str, default = "",
                    help = "the description of the running experiment")

    parser.add_argument("--evaluate", action=BooleanOptionalAction, default=False,
                        help="evaluate or train")
    
    # choices: mappo or maddpg
    parser.add_argument("--train_mode", type = str, default = "maddpg",
                        help = "training mode")
    
    parser.add_argument("--eval_seed", type = int, default = 7878,
                        help = "evaluation random-seed")
    
    parser.add_argument("--train_seed", type = int, default = 7878,
                    help = "training random-seed")

    parser.add_argument("--eval_freq", type = int, default = 400,
                        help = "the evaluation frequency in the training process (unit: episodes)")

    # choices: mappo or maddpg or local_comp or edge_comp or random_comp
    parser.add_argument("--eval_mode", type = str, default = "mappo",
                        help = "evaluation mode")
    
    parser.add_argument("--eval_episodes", type = int, default = 800,
                        help = "the number of sample-episodes for evaluation")

    parser.add_argument("--eval_time_slots", type = int, default = 3000,
                        help = "the number of time-slots for evaluation")
    
    parser.add_argument("--load_weights", action="store_true",
                        help = "whether to load network parameters")
    
    parser.add_argument("--weights_dir", type = str, default = "weight/", 
                        help = "the directory for saving network parameters")
    
    parser.add_argument("--resume_episode", type = int, default = 0, 
                        help = "the episode to resume training from")

    # environment
    parser.add_argument("--delta", type = float, default = 0.1,
                        help = "the duration of each time-slot (s)")
    
    parser.add_argument("--device_num", type = int, default = device_num,
                        help = "the number of devices")

    parser.add_argument("--device_types", type = list, 
                        # default = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
                        # default = [0]*4 + [1]*2 + [2]*2 + [3]*1 + [4]*1,
                        default = [0]*2 + [1]*4 + [2]*4,
                        help = "the types of devices")

    parser.add_argument("--device_in_types", type = list, 
                        # default = [
                        #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                        #     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                        #     [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                        #     [40, 41, 42, 43, 44],
                        #     [45, 46, 47, 48, 49]
                        # ],
                        # default = [
                        #     [0, 1, 2, 3],
                        #     [4, 5],
                        #     [6, 7],
                        #     [8],
                        #     [9]
                        # ],
                        default = [
                            [0,1],
                            [2,3,4,5],
                            [6,7,8,9],
                        ],
                        help = "all devices nums for each types")

    parser.add_argument("--device_num_per_type", type = list, 
                        # default = [20,10,10,5,5],
                        # default = [4,2,2,1,1],
                        default = [2,4,4],
                        help = "the number of devices for each type")

    parser.add_argument("--device_type_num", type = int, 
                        default = edge_queue_num,
                        help = "the number of device types")
    
    # parser.add_argument("--task_arrival_prob", type = int, default = [1.0, 0.6, 0.2, 0.12, 0.1],
    #                     help = "the probability of task arrival at each time slot")

    parser.add_argument("--task_arrival_prob", type = int, 
                        # default = [1.0, 1.0, 1.0, 1.0, 1.0],
                        default = [1.0]*edge_queue_num,
                        help = "the probability of task arrival at each time slot")
    
    parser.add_argument("--max_task_num", type = int, default = 1,
                        help = "the maximum number of tasks at each time slot")

    parser.add_argument("--gen_task_cycle", type = int, default = gen_task_cycle,
                        help = "the cycle of generating tasks (unit: time slots)")

    parser.add_argument("--start_slot", type = int, default = 0,
                        help = "the cycle of generating tasks (unit: time slots)")

    # 修改为10-15Mbps
    parser.add_argument("--total_bandwidth", type = float, 
                        default = 30 * pow(10, 6),
                        help = "total bandwidth (Hz)")
    
    parser.add_argument("--device_trans_powers", type = list, 
                        # default = [251, 206, 126, 227, 186], 
                        default = [300]*edge_queue_num,
                        help = "the transmission powers of devices (mW)")
    
    # parser.add_argument("--device_path_loss", type = list, 
    #                     default = [2.4e-10, 7.6e-11, 8.0e-11, 4.0e-10, 5.3e-10], 
    #                     help = "the path loss of devices")
    
    parser.add_argument("--spec_dens", type = float, 
                        default = pow(10, -174 / 10),
                        help = "the spectral density of noise power (mW/Hz)")
    
    parser.add_argument("--speed_max", type = float, 
                    default = 5.0,
                    help = "the max speed of end device (m)")
    
    parser.add_argument("--speed_min", type = float, 
                        default = 0.0,
                        help = "the min speed of end device (m)")
    
    parser.add_argument("--accelerate_speed_max", type = float, 
                    default = 4.0,
                    help = "the max accelerate direction of end device (m/(s^2))")
    
    parser.add_argument("--accelerate_speed_min", type = float, 
                        default = 0.0,
                        help = "the min accelerate direction of end device (m/(s^2))")
    
    parser.add_argument("--direction_max", type = float, 
                    default = 360.0,
                    help = "the max direction of end device (degree/s)")
    
    parser.add_argument("--direction_min", type = float, 
                        default = 0.0,
                        help = "the min direction of end device (degree/s)")
    
    parser.add_argument("--accelerate_direction_max", type = float, 
                    default = 60.0,
                    help = "the max accelerate direction of end device (degree/(s^2))")
    
    parser.add_argument("--accelerate_direction_min", type = float, 
                        default = -60.0,
                        help = "the min accelerate direction of end device (degree/(s^2))")
    
    parser.add_argument("--max_distance_from_edge", type = float, 
                        default = 500,
                        help = "the max distance between end device and edge server. (m)")

    parser.add_argument("--min_distance_from_edge", type = float,
                        default = 100,
                        help = "the min distance between end device and edge server. (m)")
    
    parser.add_argument("--enable_device_comp_freq_chg", action="store_true",
                        help = "Used at the env general experiments, whether to enable the change of devices' computation frequencies during training and evaluation")

    parser.add_argument("--device_comp_freq", type = float, 
                        default = 2.5,
                        help = "the computation frequencies of single device, enabled when --enable_device_comp_freq_chg is True")

    parser.add_argument("--device_comp_freqs", type = list,
                        default = [2.5]*edge_queue_num,
                        help = "the computation frequencies of devices (Gcycles/s)")
    
    parser.add_argument("--std_comp_freq", type = float, default = 2, 
                        help = "standard computation frequency (Gcycles/s)")
    
    parser.add_argument("--device_engy_facs", type = list, 
                        default = [1]*edge_queue_num,
                        help = "the energy factors of devices (J/Gcycles)")
    
    parser.add_argument(
        "--data_size_inls",
        type=list,
        # default=[[0.9, 1.2], [0.9, 1.0], [0.5, 2.0],
        #         [1.4, 1.5], [0.6, 1.2]],
        # default=[[1.0, 2.0], [0.4, 1.0], [2.6, 3.2]],
        # default=[[1.0, 2.0], [0.6, 1.0], [2.8, 3.4]],
        default=[[1.0, 2.0], [0.6, 1.0], [2.7, 3.3]],
        help="the data-size intervals of tasks (Mbits)"
    )

    parser.add_argument(
        "--comp_dens_inls",
        type=list,
        # default=[[0.3, 0.4], [0.6, 0.8], [0.15, 0.2]],
        # default=[[2.0, 2.2], [4.2, 4.4], [1.0, 1.2]],
        default=[[2.0, 2.2], [3.8, 4.0], [1.0, 1.2]],
        help="the computation-density intervals of tasks (GFLOPs/Mbits)"
    )
    parser.add_argument("--comp_dly_thre", type = list, 
                        # default = [1, 5, 5, 10, 10], 
                        # default = [3, 5, 5, 10, 10],
                        default = [5, 10, 10],
                        help = "the timeout threshold for different task types (delta 0.1s)")

    parser.add_argument("--edge_comp_freq", type = float, default = 50,
                        help = "the computation frequency of MEC server (Gcycles/s)")
    
    # parser.add_argument("--service_price", type = float, default = 0.1, 
    #                     help = "the service price of MEC server ($/Gcycles)")
    
    parser.add_argument("--device_energy_weights", type = list, 
                        # default = [0.8, 0.8, 0.8, 0.8, 0.8],
                        # default = [1.0, 1.0, 1.0, 1.0, 1.0],
                        default = [1.0]*edge_queue_num,
                        help = "the weights of tasks' energy consumption")
    
    parser.add_argument("--edge_energy_weights", type = list, 
                        default = [0.2]*edge_queue_num,
                        help = "the weights of tasks' edge computation expense")
    
    parser.add_argument("--max_data_size", type = float, 
                        default = 3.5,
                        help = "maximum data-size (Mb)")

    parser.add_argument("--max_comp_dens", type = float,
                        default = 1.8,
                        help = "maximum computation density (Gcycles/Mb)")

    parser.add_argument("--edge_weight_w", type = float,
                        default = 1,
                        help = "the w parameter for edge weight linear function")

    parser.add_argument("--edge-weight_b", type = float,
                        default = 0.1,
                        help = "the b parameter for edge weight linear function")

    parser.add_argument("--device_dly_adj_fac", type = float,
                        default = 0.9,
                        # default = [0.8]*edge_queue_num,
                        help = "the weight of tasks' device computation queue overtime threshold factor.")

    parser.add_argument("--edge_dly_adj_fac", type = float, 
                        default = 0.8,
                        # default = [0.6]*edge_queue_num,
                        help = "the weight of tasks' edge computation queue overtime threshold factor.")

    # parser.add_argument("--vir_local_ql_growth_rate", type = float,
    #                     default = 0.6,
    #                     help = "The length growth rate of the virtual computing queue on the local device")

    # parser.add_argument("--vir_edge_ql_growth_rate", type = float,
    #                     default = 0.6,
    #                     help = "The length growth rate of the virtual computing queue on the edge server")

    # Hyperparameter
    # queue reward
    parser.add_argument("--device_act_queue_reward_max_bound", type = float, default = 800 * ql_punish_fac, 
                        help = "the max bound of actual device queue reward")

    parser.add_argument("--device_act_queue_reward_min_bound", type = float, default = -1000 * ql_punish_fac,
                        help = "the min bound of actual device queue reward")

    parser.add_argument("--device_vir_queue_reward_max_bound", type = float, default = 1600 * ql_punish_fac,
                        help = "the max bound of virtual device queue reward")

    parser.add_argument("--device_vir_queue_reward_min_bound", type = float, default = -2000 * ql_punish_fac, 
                        help = "the min bound of virtual device queue reward")

    parser.add_argument("--device_act_queue_reward_weight", type = float,
                        default = -1000 * ql_punish_fac,
                        help = "The Lyapunov Drift-Plus-Penalty weight for local queues")
    
    parser.add_argument("--device_vir_queue_reward_weight", type = float,
                        default = -2000 * ql_punish_fac,
                        help = "The Lyapunov Drift-Plus-Penalty weight for edge queues")

    parser.add_argument("--edge_queue_reward_bound_fac", type = float,
                        default = 0.95,
                        help = "The bound multi factor for edge queue reward")

    # navie reward
    parser.add_argument("--base_reward_penalty", type = float, default = 0,
                        help = "the base reward penalty for each task")

    parser.add_argument("--timeout_reward_penalty", type = float, default = -4000,
                        help = "the reward if the task is timeout")

    parser.add_argument("--lya_timeout_reward_penalty", type = float, default = -1500,
                        help = "the reward if the task is timeout and the train algorithm is Lya-MADDPG")

    parser.add_argument("--target_reward_penalty", type = float, default = -80,
                        help = "the reward if the task is completed within the threshold")

    # device queue growth rate
    parser.add_argument("--device_act_queue_growth_rate", type = float, default = 1, 
                        help = "the growth rate of actual device queue reward")

    parser.add_argument("--device_vir_queue_growth_rate", type = float, default = 1, 
                    help = "the growth rate of virtual device queue reward")

    # For average computation time
    parser.add_argument("--statSlotNum", type = int, default = 100,
                        help = "the number of slots to calculate average computation time")

    # edge queue growth rate
    parser.add_argument("--edge_act_queue_growth_rate", type = float, default = 1,
                        help = "the growth rate of actual edge queue reward")
    parser.add_argument("--edge_vir_queue_growth_rate", type = float, default = 1,
                        help = "the growth rate of virtual edge queue reward")

    # ou_noise settings
    parser.add_argument("--use_ou_noise", action="store_true",
                        help = "whether to use ou noise for action exploration")

    parser.add_argument("--ou_theta", type = float, default = 0.15,
                        help = "the theta parameter of ou noise")

    parser.add_argument("--ou_sigma", type = float, default = 0.20,
                        help = "the sigma parameter of ou noise")
    
    parser.add_argument("--ou_dt", type = float, default = 1.0,
                        help = "the dt parameter of ou noise")

    parser.add_argument("--ou_scale", type = float, default = 1.0,
                        help = "the scale parameter of ou noise")

    parser.add_argument("--results_dir", type = str, default = "result/",
                        help = "the directory for saving training results")
    
    parser.add_argument("--plot_dir", type = str, default = "runs/plot/",
                        help = "the directory for saving plot images")

    params = parser.parse_args()
    
    return params
"""
mappo params
"""
ppo_device_obs_dim = 6 # 3 + max_task_num * 3
ppo_edge_queue_obs_dim = 2 # comp_ql_length + vir_comp_ql_length
ppo_value_input_dims = [n * ppo_device_obs_dim + ppo_edge_queue_obs_dim for n in [2, 4, 4]]
ppo_policy_input_dim = ppo_device_obs_dim + ppo_edge_queue_obs_dim
def get_mappo_params():
    parser = argparse.ArgumentParser(description = "mappo params", add_help=False, allow_abbrev=False)

    #! 修改后，ObsScaling类的代码也需要改
    # env obs dim and networks input dim
    parser.add_argument("--device_obs_dim", type = int, default = ppo_device_obs_dim,
                        help = "the dimension of devices' observations")
    
    parser.add_argument("--edge_queue_obs_dim", type = int, default = ppo_edge_queue_obs_dim,
                        help = "the dimension of edge queues' observations")

    parser.add_argument("--value_input_dims", type = list, default = ppo_value_input_dims,
                        help = "the dimension of value network input")
    
    parser.add_argument("--policy_input_dim", type = int, default = ppo_policy_input_dim,
                        help = "the dimension of policy network input")

    # 包含：任务远程卸载率、传输能耗利用率、本地计算频率利用率
    parser.add_argument("--action_dim", type = int, default = 3,
                        help = "the dimension of agents' actions")

    parser.add_argument("--v_hid_dims", type = list, default = [200, 200],
                        help = "the dimension of value network's hidden layers")

    parser.add_argument("--p_hid_dims", type = list, default = [200, 200],
                        help = "the dimension of policy network's hidden layers")

    parser.add_argument("--use_orthogonal_init", type = bool, default = True,
                        help = "whether to use orthogonal-initialization")
    
    # training
    parser.add_argument("--train_episodes", type = int, default = 10000,
                        help = "the number of training episodes")

    parser.add_argument("--train_time_slots", type = int, default = 3000,
                        help = "the number of training time-slots")
    
    parser.add_argument("--buffer_train_time_slots", type = int, default = 600,
                        help = "the number of training time-slots storing in the replay buffer.")

    # parser.add_argument("--buffer_train_freq", type = int, default = 4,
    #                     help = "training frequency")

    parser.add_argument("--train_freq", type = int, default = 4,
                        help = "training frequency")
    
    parser.add_argument("--train_batch_size", type = int, default = 2400,
                        help = "training batch-size")
    
    parser.add_argument("--v_epochs", type = int, default = 4,
                        help = "the number of training epoches of value network")
    
    parser.add_argument("--p_epochs", type = int, default = 4,
                        help = "the number of training epoches of policy networks")
    
    parser.add_argument("--gamma", type = float, default = 0.99,
                        help = "the discount factor of rewards")
    
    parser.add_argument("--lamda", type = float, default = 0.95,
                        help = "the parameter about GAE")
    
    parser.add_argument("--v_lr", type = float, default = 1e-4,
                        help = "the learning-rate of value network")
    
    parser.add_argument("--p_lr", type = float, default = 1e-4,
                        help = "the learning-rate of policy networks")
    
    parser.add_argument("--use_lr_decay", type = bool, default = False,
                        help = "whether to use learning-rate decay")
    
    parser.add_argument("--min_v_lr", type = float, default = 1e-5, 
                        help = "the minimal learning-rate of value network")
    
    parser.add_argument("--min_p_lr", type = float, default = 1e-5,
                        help = "the minimal learning-rate of policy networks")
    
    parser.add_argument("--decay_fac", type = float, default = 0.999, 
                        help = "the parameter about learning-rate decay")
    
    parser.add_argument("--use_obs_scaling", type = bool, default = True, 
                        help = "whether to use observation scaling")
    
    parser.add_argument("--use_reward_scaling", type = bool, default = True, 
                        help = "whether to use reward scaling")
    
    parser.add_argument("--use_grad_clip", type = bool, default = True, 
                        help = "whether to use gradient clip")
    
    parser.add_argument("--v_grad_clip", type = float, default = 2,  
                        help = "the parameter about value network's gradient clip")
    
    parser.add_argument("--p_grad_clip", type = float, default = 2,  
                        help = "the parameter about policy networks' gradient clip")
    
    parser.add_argument("--p_clip", type = float, default = 0.1,
                        help = "the parameter about ppo clip")

    parser.add_argument("--enty_coef", type = float, default = 0.05,   
                        help = "the coefficient about policy's entropy")

    parser.add_argument("--save_freq", type = int, default = 1000, 
                        help = "the saving frequency of networks")
    
    params, unknown = parser.parse_known_args()
    
    return params

"""
maddpg params
"""
device_obs_dim = 6 # 3 + max_task_num * 3
edge_queue_obs_dim = 2 # comp_ql_length + vir_comp_ql_length
action_dim = 30 # 动作包含卸载率、传输功率利用率和本地计算频率利用率，每个值用10维表示
value_input_obs_dims = [n * (device_obs_dim) + edge_queue_obs_dim for n in [2, 4, 4]] # 包含整个集群的观测信息
value_input_act_dims = [n * action_dim for n in [2, 4, 4]] # 包含所有智能体的动作信息
value_input_dims = [n * (device_obs_dim + action_dim) + edge_queue_obs_dim for n in [2, 4, 4]] # 包含整个集群的观测信息和所有智能体的动作信息
policy_input_dim = device_obs_dim + edge_queue_obs_dim
maddpg_train_episodes = 15000
maddpg_time_slots = 3000
maddpg_lr_warm_episodes = maddpg_train_episodes * 0.15
# maddpg_train_freq = 4
# maddpg_update_freq = 8
# maddpg_buffer_episodes_num = 8
maddpg_lr_decay_freq = 100
# 260129 14:35 提高样本利用效率
maddpg_train_freq = 1
maddpg_update_freq = 1
maddpg_buffer_episodes_num = 20
maddpg_policy_delay_round = 2
maddpg_critic_updates_round = 1
maddpg_p_lr = 5e-5
maddpg_v_lr = 1e-4
maddpg_p_min_lr = 1e-7
maddpg_v_min_lr = 1e-7
def get_maddpg_params():
    parser = argparse.ArgumentParser(description = "maddpg params", add_help=False, allow_abbrev=False)
    
    # networks
    parser.add_argument("--device_obs_dim", type = int, default = device_obs_dim,
                        help = "the dimension of agents' observations")
    
    parser.add_argument("--edge_queue_obs_dim", type = int, default = edge_queue_obs_dim,
                        help = "the dimension of edge queues' observations")

    parser.add_argument("--value_input_obs_dims", type = list, default = value_input_obs_dims,
                        help = "the dimension of value network input's obs part")

    parser.add_argument("--value_input_act_dims", type = list, default = value_input_act_dims,
                        help = "the dimension of value network input's action part")

    parser.add_argument("--value_input_dims", type = list, default = value_input_dims,
                        help = "the dimension of value network input")
    
    parser.add_argument("--policy_input_dim", type = int, default = policy_input_dim,
                        help = "the dimension of policy network input")

    parser.add_argument("--action_dim", type = int, default = action_dim,
                        help = "the dimension of agents' actions")
    
    parser.add_argument("--v_hid_dims", type = list, default = [200, 200],
                        help = "the dimension of value network's hidden layers")
    
    parser.add_argument("--p_hid_dims", type = list, default = [200, 200], 
                        help = "the dimension of policy network's hidden layers")
    
    parser.add_argument("--use_orthogonal_init", type = bool, default = True,
                        help = "whether to use orthogonal-initialization")
    
    # training  
    parser.add_argument("--train_episodes", type = int, default = maddpg_train_episodes,
                        help = "the number of training episodes")
    
    parser.add_argument("--train_time_slots", type = int, default = maddpg_time_slots,
                        help = "the number of training time-slots")
    
    parser.add_argument("--lr_warm_time_slots", type = int, default = maddpg_lr_warm_episodes * maddpg_time_slots,
                        help = "the number of warming time-slots")

    parser.add_argument("--warm_time_slots", type = int, default = maddpg_time_slots * 4,
                        help = "the number of warming time-slots")
    
    parser.add_argument("--train_freq", type = int, default = maddpg_time_slots * maddpg_train_freq,
                        help = "training frequency")
    
    parser.add_argument("--target_update_freq", type = int, default = maddpg_time_slots * maddpg_update_freq,
                        help = "the updating frequency of target networks")
    
    parser.add_argument("--train_batch_size", type = int, default = int(maddpg_time_slots / gen_task_cycle * maddpg_train_freq),
                        help = "training batch-size")
    
    parser.add_argument("--v_epochs", type = int, default = 4,
                        help = "the number of training epochs of value network")
    
    parser.add_argument("--p_epochs", type = int, default = 1,
                        help = "the number of training epochs of policy networks")
    
    parser.add_argument("--buffer_size", type = int, default = maddpg_time_slots * maddpg_buffer_episodes_num, 
                        help = "the size of replay buffer")
    
    parser.add_argument("--gamma", type = float, default = 0.99,
                        help = "the discount factor of rewards")
    
    parser.add_argument("--tau", type = float, default = 0.005,
                        help = "the soft update factor of target networks")

    parser.add_argument("--peak_v_lr", type = float, default = maddpg_v_lr,           
                        help = "the learning-rate of value network")
    
    parser.add_argument("--peak_p_lr", type = float, default = maddpg_p_lr,          
                        help = "the learning-rate of policy networks")
    
    parser.add_argument("--use_lr_decay", type = bool, default = True,
                        help = "whether to use learning-rate decay")
    
    parser.add_argument("--min_v_lr", type = float, default = maddpg_v_min_lr,       
                        help = "the minimal learning-rate of value network")
    
    parser.add_argument("--min_p_lr", type = float, default = maddpg_p_min_lr,
                        help = "the minimal learning-rate of policy networks")
    
    parser.add_argument("--critic_updates_round", type = int, default = maddpg_critic_updates_round,
                        help = "the number of critic network updates per training")

    parser.add_argument("--policy_delay_round", type = int, default = maddpg_policy_delay_round,
                        help = "the number of policy network delays per training")

    parser.add_argument("--decay_intl", type = int, default = 300000,  
                        help = "the interval of learning-rate decay")
    
    # 训练到一半学习率达到最小值
    parser.add_argument("--p_decay_fac", type = float, default = (maddpg_p_lr - maddpg_p_min_lr) / (maddpg_train_episodes / maddpg_lr_decay_freq / 2),
                        help = "the parameter about policy networklearning-rate decay")

    parser.add_argument("--v_decay_fac", type = float, default = (maddpg_v_lr - maddpg_v_min_lr) / (maddpg_train_episodes / maddpg_lr_decay_freq / 2),
                        help = "the parameter about value network learning-rate decay")
    
    parser.add_argument("--use_obs_scaling", type = bool, default = True, 
                        help = "whether to use observation scaling")
    
    parser.add_argument("--use_reward_scaling", type = bool, default = True, 
                        help = "whether to use reward scaling")
    
    parser.add_argument("--use_action_noise", type = bool, default = True,
                        help = "whether to add noise in agents' actions")
    
    parser.add_argument("--noise_sigma_start", type = float, default = 0.4,
                        help = "noise sigma at the start of training")

    parser.add_argument("--noise_sigma_end", type = float, default = 0.02,
                        help = "noise sigma at the end of training")

    # 总训练回合数/训练频率
    parser.add_argument("--noise_decay_num", type = float, default = maddpg_train_episodes / maddpg_train_freq,
                        help = "the number of updates over which the noise sigma decays")

    parser.add_argument("--use_grad_clip", type = bool, default = True, 
                        help = "whether to use gradient clip")
    
    parser.add_argument("--v_grad_clip", type = float, default = 2,
                        help = "the parameter about value network's gradient clip")
    
    parser.add_argument("--p_grad_clip", type = float, default = 2,   
                        help = "the parameter about policy networks' gradient clip")
    
    parser.add_argument("--save_freq", type = int, default = 600000,
                        help = "the saving frequency of networks")
    
    parser.add_argument("--weights_dir", type = str, default = "weight/", 
                        help = "the directory for saving network parameters")
    
    params, unknown = parser.parse_known_args()
    
    return params