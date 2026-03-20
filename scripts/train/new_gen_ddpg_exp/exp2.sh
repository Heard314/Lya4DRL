python ../../../main.py --train_mode maddpg --run_desc "gen0.95_qe" --enable_virtual_queue_reward --edge_dly_adj_fac 0.95
python ../../../main.py --train_mode maddpg --run_desc "gen1.0_qe" --enable_virtual_queue_reward --edge_dly_adj_fac 1.0
python ../../../main.py --train_mode maddpg --run_desc "gen1.05_qe" --enable_virtual_queue_reward --edge_dly_adj_fac 1.05
python ../../../main.py --train_mode maddpg --run_desc "gen1.1_qe" --enable_virtual_queue_reward --edge_dly_adj_fac 1.1
python ../../../main.py --train_mode maddpg --run_desc "gen1.15_qe" --enable_virtual_queue_reward --edge_dly_adj_fac 1.15
python ../../../main.py --train_mode maddpg --run_desc "gen1.2_qe" --enable_virtual_queue_reward --edge_dly_adj_fac 1.2
python ../../../main.py --train_mode maddpg --run_desc "gen1.25_qe" --enable_virtual_queue_reward --edge_dly_adj_fac 1.25

