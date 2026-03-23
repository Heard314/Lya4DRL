python ../../../main.py --train_mode maddpg --run_desc "my_exp2_l2.25" --enable_virtual_queue_reward --enable_device_comp_freq_chg --device_comp_freq 2.25
python ../../../main.py --train_mode maddpg --run_desc "my_exp2_l2.75" --enable_virtual_queue_reward --enable_device_comp_freq_chg --device_comp_freq 2.75
python ../../../main.py --train_mode maddpg --run_desc "my_exp2_l3.0" --enable_virtual_queue_reward --enable_device_comp_freq_chg --device_comp_freq 3.0

python ../../../main.py --train_mode maddpg --run_desc "rt_exp_l2.25" --enable_actual_queue_reward --enable_device_comp_freq_chg --device_comp_freq 2.25
python ../../../main.py --train_mode maddpg --run_desc "rt_exp_l2.75" --enable_actual_queue_reward --enable_device_comp_freq_chg --device_comp_freq 2.75
python ../../../main.py --train_mode maddpg --run_desc "rt_exp_l3.0" --enable_actual_queue_reward --enable_device_comp_freq_chg --device_comp_freq 3.0

python ../../../main.py --train_mode maddpg --run_desc "no_queue_l2.25" --enable_device_comp_freq_chg --device_comp_freq 2.25
python ../../../main.py --train_mode maddpg --run_desc "no_queue_l2.75" --enable_device_comp_freq_chg --device_comp_freq 2.75
python ../../../main.py --train_mode maddpg --run_desc "no_queue_l3.0" --enable_device_comp_freq_chg --device_comp_freq 3.0