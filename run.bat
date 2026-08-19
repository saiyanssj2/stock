@echo off
set PYTHONIOENCODING=utf-8

:: Xóa log cũ mỗi lần start
if exist pipeline_debug.log del pipeline_debug.log

:: Log thời điểm khởi động app
echo [%date% %time%] APP START (run.bat) >> pipeline_debug.log

python -m streamlit run app.py

:: Log khi app dừng (Ctrl+C hoặc đóng cửa sổ)
echo [%date% %time%] APP STOP (run.bat) >> pipeline_debug.log

:: Reset status nếu đang running (tránh kẹt)
python -c "import json,os; f='data/engine/status/auto_pipeline_status.json'; d=json.load(open(f,'r',encoding='utf-8')) if os.path.exists(f) else {}; exec(\"if d.get('state')=='running': open(f,'w',encoding='utf-8').write(json.dumps({'state':'done','timestamp':'2026-01-01T00:00:00','message':'app stopped'}))\") if d else None" 2>nul
