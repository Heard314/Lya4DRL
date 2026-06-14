python ../../../main.py --train_mode maddpg --run_desc "act_queue_exp7" --enable_actual_queue_reward \
    --device_vir_queue_reward_weight -1400 --device_act_queue_reward_weight -600 \
    --device_act_queue_reward_max_bound 400 --device_act_queue_reward_min_bound -600 \
    --device_vir_queue_reward_max_bound 800 --device_vir_queue_reward_min_bound -1200 \
    --device_act_queue_growth_rate 1 --device_vir_queue_growth_rate 0.1 \
    --edge_act_queue_growth_rate 1 --edge_vir_queue_growth_rate 0.1 \
    --timeout_reward_penalty -4000 --target_reward_penalty -1000