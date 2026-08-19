@echo off
REM ============================================================
REM Script: cleanup-old-checkpoints.bat
REM Mục đích: Xóa các checkpoint cũ không còn dùng trong engine/models
REM Cách chạy: Chạy trực tiếp từ thư mục fixbug hoặc root project
REM Cảnh báo: Không thể hoàn tác sau khi xóa!
REM ============================================================

echo.
echo === DON DEP CHECKPOINT CU ===
echo.

set MODELS_DIR=e:\source_code\project\stock\engine\models

echo [1/4] Xoa checkpoint_epoch_*.pt (3 files, ~9 MB)...
del /Q "%MODELS_DIR%\checkpoint_epoch_*.pt" 2>nul
if %errorlevel%==0 (echo     OK) else (echo     Khong tim thay hoac da xoa)

echo [2/4] Xoa checkpoint_incremental_epoch_*.pt (10 files, ~30 MB)...
del /Q "%MODELS_DIR%\checkpoint_incremental_epoch_*.pt" 2>nul
if %errorlevel%==0 (echo     OK) else (echo     Khong tim thay hoac da xoa)

echo [3/4] Xoa hard_example_finetuned.pt (~3 MB)...
del /Q "%MODELS_DIR%\hard_example_finetuned.pt" 2>nul
if %errorlevel%==0 (echo     OK) else (echo     Khong tim thay hoac da xoa)

echo [4/4] Xoa backup cu (giu lai pre_confidence)...
del /Q "%MODELS_DIR%\stock_eval_net_backup_pre_label_change.pt" 2>nul
del /Q "%MODELS_DIR%\stock_eval_net_backup_66features.pt" 2>nul
del /Q "%MODELS_DIR%\stock_eval_net_backup_wyckoff_v1.pt" 2>nul
del /Q "%MODELS_DIR%\stock_eval_net_backup_72features.pt" 2>nul
del /Q "%MODELS_DIR%\rl_policy_backup_72features.pt" 2>nul
del /Q "%MODELS_DIR%\rl_policy_backup_before_reward_v2.pt" 2>nul
echo     OK

echo.
echo === HOAN TAT ===
echo Da giai phong khoang 45 MB
echo.
echo Files con lai:
dir /B "%MODELS_DIR%\*.pt" "%MODELS_DIR%\*.json"
echo.
