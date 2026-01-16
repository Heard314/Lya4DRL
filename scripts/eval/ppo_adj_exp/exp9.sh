python ../../../main.py --evaluate --eval_mode mappo --run_desc "all_queue_exp9" --enable_actual_queue_reward --enable_virtual_queue_reward \
    --load_weights --resume_episode 20000 --weights_dir "weight/mappo/train/mappo_s_7878_t_2026-01-15-21-42-37-279491_d_all_queue_exp9/" \
    --edge_queue_reward_weight -1400 --device_queue_reward_weight -600 \
    --device_act_queue_reward_max_bound 200 --device_act_queue_reward_min_bound -300 \
    --device_vir_queue_reward_max_bound 800 --device_vir_queue_reward_min_bound -1200 \
    --device_act_queue_growth_rate 1 --device_vir_queue_growth_rate 0.1 \
    --edge_act_queue_growth_rate 1 --edge_vir_queue_growth_rate 0.1 \
    --timeout_reward_penalty -4000 --target_reward_penalty -1000