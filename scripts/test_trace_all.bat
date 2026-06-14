@echo off
REM ============================================================
REM Run both MAPPO and MADDPG trace tests sequentially
REM ============================================================
echo ===== Starting MAPPO trace test (4 episodes) =====
call D:\Desktop\work\yanjiushengbishe\mypaper\Lya4DRL\scripts\test_trace_mappo.bat
echo.
echo ===== Starting MADDPG trace test (4 episodes) =====
call D:\Desktop\work\yanjiushengbishe\mypaper\Lya4DRL\scripts\test_trace_maddpg.bat
echo.
echo ===== All trace tests completed =====
