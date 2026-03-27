python ../../../../main.py --train_mode mappo --run_desc "ppo_exp1_e" --enable_actual_queue_reward
python ../../../../main.py --train_mode mappo --run_desc "ppo_exp1_e45" --enable_actual_queue_reward --edge_comp_freq 45
python ../../../../main.py --train_mode mappo --run_desc "ppo_exp1_e47.5" --enable_actual_queue_reward --edge_comp_freq 47.5
python ../../../../main.py --train_mode mappo --run_desc "ppo_exp1_e52.5" --enable_actual_queue_reward --edge_comp_freq 52.5
python ../../../../main.py --train_mode mappo --run_desc "ppo_exp1_e55" --enable_actual_queue_reward --edge_comp_freq 55
python ../../../../main.py --train_mode mappo --run_desc "ppo_exp1_l2" --enable_actual_queue_reward --enable_device_comp_freq_chg --device_comp_freq 2
python ../../../../main.py --train_mode mappo --run_desc "ppo_exp1_l2.25" --enable_actual_queue_reward --enable_device_comp_freq_chg --device_comp_freq 2.25
python ../../../../main.py --train_mode mappo --run_desc "ppo_exp1_l2.75" --enable_actual_queue_reward --enable_device_comp_freq_chg --device_comp_freq 2.75
python ../../../../main.py --train_mode mappo --run_desc "ppo_exp1_l3" --enable_actual_queue_reward --enable_device_comp_freq_chg --device_comp_freq 3