python ../../../main.py --train_mode maddpg --run_desc "my_exp1_e47.5" --enable_virtual_queue_reward --edge_comp_freq 47.5
python ../../../main.py --train_mode maddpg --run_desc "my_exp1_e52.5" --enable_virtual_queue_reward --edge_comp_freq 52.5
python ../../../main.py --train_mode maddpg --run_desc "my_exp1_e55" --enable_virtual_queue_reward --edge_comp_freq 55.0

python ../../../main.py --train_mode maddpg --run_desc "rt_exp_e47.5" --enable_actual_queue_reward --edge_comp_freq 47.5
python ../../../main.py --train_mode maddpg --run_desc "rt_exp_e52.5" --enable_actual_queue_reward --edge_comp_freq 52.5
python ../../../main.py --train_mode maddpg --run_desc "rt_exp_e55" --enable_actual_queue_reward --edge_comp_freq 55.0

python ../../../main.py --train_mode maddpg --run_desc "no_queue_e47.5" --edge_comp_freq 47.5
python ../../../main.py --train_mode maddpg --run_desc "no_queue_e52.5" --edge_comp_freq 52.5
python ../../../main.py --train_mode maddpg --run_desc "no_queue_e55" --edge_comp_freq 55.0