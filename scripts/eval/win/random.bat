python ../../../main.py --evaluate --eval_mode random_comp --run_desc "random" \
    --device_act_queue_growth_rate 1 --device_vir_queue_growth_rate 0.1 \
    --edge_act_queue_growth_rate 1 --edge_vir_queue_growth_rate 0.1 \
    --timeout_reward_penalty -4000 --target_reward_penalty -40