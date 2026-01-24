python ../../../main.py --train_mode maddpg --run_desc "all_queue_exp3" --enable_actual_queue_reward --enable_virtual_queue_reward \
    --edge_queue_reward_weight -700 --device_queue_reward_weight -300 \
    --device_act_queue_reward_max_bound 300 --device_act_queue_reward_min_bound -450 \
    --device_vir_queue_reward_max_bound 1200 --device_vir_queue_reward_min_bound -1800 \
    --device_act_queue_growth_rate 1 --device_vir_queue_growth_rate 0.1 \
    --edge_act_queue_growth_rate 1 --edge_vir_queue_growth_rate 0.1 \
    --timeout_reward_penalty -4000 --target_reward_penalty -2000