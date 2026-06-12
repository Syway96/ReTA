@echo off
chcp 65001 >nul
echo.
echo ===============================================
echo   AI 课程智能问答系统 - 一键部署
echo ===============================================
echo.
pause

python deploy.py

if errorlevel 1 (
    echo.
    echo 部署失败，请检查上述错误信息
    echo.
    pause
) else (
    echo.
    echo 部署成功!
)

pause
