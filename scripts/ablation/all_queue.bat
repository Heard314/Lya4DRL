echo "Add all queue punishment to reward"
python ..\..\main.py --train_mode mappo --run_desc "all_queue_mappo" --enable_actual_queue_reward --enable_virtual_queue_reward
echo "All queue punishment run finished!"

