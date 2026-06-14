@echo off
REM ============================================================
REM MAPPO trace test - 4 episodes with per-slot variable tracing
REM Output: experiments/log/train/mappo_s_*_t_*_d_trace_test_trace.log
REM ============================================================
cd /d D:\Desktop\work\yanjiushengbishe\mypaper\Lya4DRL
call conda run -n marl python main.py --train_mode mappo --run_desc trace_test --seed 42 --train_seed 42 --train_episodes 4 --train_time_slots 3000 --enable_trace
echo MAPPO trace test completed (exit code: %ERRORLEVEL%)
