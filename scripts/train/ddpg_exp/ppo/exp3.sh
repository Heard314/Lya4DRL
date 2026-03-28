python ../../../main.py --train_mode maddpg --run_desc "my_exp" --enable_virtual_queue_reward
python ../../../main.py --train_mode maddpg --run_desc "rt_exp" --enable_actual_queue_reward
python ../../../main.py --train_mode maddpg --run_desc "no_queue_exp" 

