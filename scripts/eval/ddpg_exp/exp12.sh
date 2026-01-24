python ../../../main.py --train_mode maddpg --run_desc "no_queue_exp12" --use_ou_noise  \
    --edge_queue_reward_weight -700 --device_queue_reward_weight -300 \
    --device_act_queue_reward_max_bound 200 --device_act_queue_reward_min_bound -300 \
    --device_vir_queue_reward_max_bound 800 --device_vir_queue_reward_min_bound -1200 \
    --device_act_queue_growth_rate 1 --device_vir_queue_growth_rate 0.1 \
    --edge_act_queue_growth_rate 1 --edge_vir_queue_growth_rate 0.1 \
    --timeout_reward_penalty -2000 --target_reward_penalty -2000
