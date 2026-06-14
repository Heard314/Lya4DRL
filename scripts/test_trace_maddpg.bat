@echo off
REM ============================================================
REM MADDPG trace test - 4 episodes with per-slot variable tracing
REM Output: experiments/log/train/maddpg_s_*_t_*_d_trace_test_trace.log
REM ============================================================
cd /d D:\Desktop\work\yanjiushengbishe\mypaper\Lya4DRL
call conda run -n marl python main.py --train_mode maddpg --run_desc trace_test --seed 7878 --train_seed 7878 --train_episodes 4 --train_time_slots 3000 --enable_trace
echo MADDPG trace test completed (exit code: %ERRORLEVEL%)
