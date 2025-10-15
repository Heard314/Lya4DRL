@REM conda activate marl
@echo off
echo start training...
echo.

for /L %%i in (1,1,1) do (
    echo ========================================
    echo run %%i times
    echo ========================================
    python main.py --train_mode maddpg
    python main.py --train_mode mappo
    echo run %%i times finished
    echo.
    if %%i LSS 5 (
        echo wait 3 seconds...
        timeout /t 3 /nobreak >nul
    )
)

echo ========================================
echo all runs finished!
echo ========================================
pause