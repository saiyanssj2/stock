# -*- coding: utf-8 -*-
"""
Pipeline Worker — chạy trong background thread.

Thực hiện: update data → Walk-Forward cycle (SL + RL + backtest).
Được trigger từ UI (nút Start) hoặc gọi trực tiếp.
"""

import json
import traceback
from datetime import datetime
from pathlib import Path

from config.preferences import load_preferences_from_file

# Status file path
STATUS_FILE = Path("data/engine/status/auto_pipeline_status.json")
DEBUG_LOG = Path("pipeline_debug.log")


def _log(msg: str) -> None:
    """Ghi log debug ra file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


def _write_status(data: dict) -> None:
    """Ghi status ra file JSON (thread-safe ghi đè)."""
    try:
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        _log(f"  status: {data.get('step', data.get('state'))} - {data.get('message', '')}")
    except Exception as e:
        _log(f"  ERROR write status: {e}")


def is_running() -> bool:
    """Kiểm tra pipeline có đang chạy không."""
    if not STATUS_FILE.exists():
        return False
    try:
        data = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        return data.get("state") == "running"
    except Exception:
        return False


def request_stop() -> None:
    """Yêu cầu dừng pipeline (ghi flag để worker check)."""
    stop_file = Path("data/engine/status/pipeline_stop_requested")
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.write_text("1", encoding="utf-8")


def _should_stop() -> bool:
    """Kiểm tra xem có yêu cầu dừng không (không xóa file, để caller quyết định)."""
    stop_file = Path("data/engine/status/pipeline_stop_requested")
    return stop_file.exists()


def _clear_stop_flag() -> None:
    """Xóa flag stop sau khi đã xử lý xong."""
    stop_file = Path("data/engine/status/pipeline_stop_requested")
    stop_file.unlink(missing_ok=True)


def run_pipeline() -> None:
    """
    Chạy pipeline loop: update data → Walk-Forward cycle → lặp lại.
    
    Dừng khi user nhấn Stop (ghi flag).
    """
    _log("=" * 60)
    _log("PIPELINE START (loop mode)")
    _log("=" * 60)

    _write_status({
        "state": "running",
        "step": "starting",
        "message": "Khởi động pipeline...",
        "timestamp": datetime.now().isoformat(),
    })

    prefs = load_preferences_from_file()
    if prefs is None:
        _log("ERROR: prefs is None")
        _write_status({"state": "done", "message": "No preferences", "timestamp": datetime.now().isoformat()})
        return

    symbols = prefs.get("selected_symbols", [])
    if not symbols:
        from pathlib import Path as _P
        csv_files = list(_P("data").glob("*.csv"))
        symbols = sorted([f.stem for f in csv_files if f.stem.isalpha()])
        if not symbols:
            from config.vn_market_rules import VN30_SYMBOLS
            symbols = VN30_SYMBOLS
    _log(f"Preferences loaded: {len(symbols)} symbols")

    # === LOOP: chạy liên tục cho đến khi Stop ===
    while True:
        if _should_stop():
            _clear_stop_flag()
            _log("STOPPED by user")
            _write_status({"state": "done", "message": "Đã dừng bởi user", "timestamp": datetime.now().isoformat()})
            return

        # --- BƯỚC 1: Update Data ---
        _log("--- BƯỚC 1: Update Data ---")
        _write_status({"state": "running", "step": "update_data", "message": "Đang cập nhật data...", "timestamp": datetime.now().isoformat()})

        try:
            _do_update_data()
        except Exception as e:
            _log(f"  EXCEPTION update data: {e}")

        if _should_stop():
            _clear_stop_flag()
            _log("STOPPED by user")
            _write_status({"state": "done", "message": "Đã dừng bởi user", "timestamp": datetime.now().isoformat()})
            return

        # --- BƯỚC 2: Walk-Forward Cycle ---
        _log("--- BƯỚC 2: Walk-Forward Cycle ---")
        _write_status({"state": "running", "step": "wf_cycle", "message": "Đang chạy Walk-Forward Cycle...", "timestamp": datetime.now().isoformat()})

        try:
            from engine.wf_trainer import WalkForwardCycle
            from engine.wf_trainer.config import WFConfig

            wf_config = WFConfig()
            wf = WalkForwardCycle(config=wf_config)

            _log(f"  Running WF cycle #{wf.cycle_count + 1} with symbols: {len(symbols)} mã")

            report = wf.run(symbols=symbols, debug_log_path=str(DEBUG_LOG))
            _log(f"  DONE: cycle #{report.cycle_number}, "
                 f"SL val_loss={report.sl_val_loss:.4f}, "
                 f"RL avg_return={report.rl_avg_return:.2%}, "
                 f"Backtest return={report.backtest_return:.2%}, "
                 f"WR={report.backtest_win_rate:.1%}, "
                 f"trades={report.backtest_trades}")
            _write_status({
                "state": "running", "step": "wf_cycle_done",
                "message": f"Cycle #{report.cycle_number}: "
                           f"return={report.backtest_return:.1%}, "
                           f"WR={report.backtest_win_rate:.0%}, "
                           f"{report.backtest_trades} trades",
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            _log(f"  EXCEPTION WF: {e}")
            import traceback
            _log(traceback.format_exc())

        _log("--- CYCLE DONE, chờ 60s rồi tiếp ---")
        _log("")

        # Chờ 60s trước cycle tiếp (check stop mỗi 5s)
        import time
        for _ in range(12):
            if _should_stop():
                _clear_stop_flag()
                _log("STOPPED by user")
                _write_status({"state": "done", "message": "Đã dừng bởi user", "timestamp": datetime.now().isoformat()})
                return
            time.sleep(5)


def _do_update_data() -> None:
    """Logic update data — tách ra để gọn. Có thể dừng giữa chừng."""
    from engine.data_update_wiring import (
        load_last_update_timestamp,
        save_last_update_timestamp,
    )

    last_update = load_last_update_timestamp()
    need_update = True
    if last_update and (last_update.get("trading_weekday") is not None or last_update.get("timestamp")):
        try:
            now = datetime.now()

            if last_update.get("trading_weekday") is not None:
                a_weekday = last_update["trading_weekday"]
                last_slot = last_update.get("slot", "P")
                last_ts = datetime.fromisoformat(last_update["timestamp"])
                a_date = last_ts.date()
            else:
                last_ts = datetime.fromisoformat(last_update["timestamp"])
                a_date = last_ts.date()
                a_weekday = a_date.weekday()
                h, m = last_ts.hour, last_ts.minute
                if h < 9 or (h == 9 and m < 15):
                    last_slot = "X"
                elif (h == 9 and m >= 15) or (h == 10) or (h == 11 and m < 30):
                    last_slot = "M"
                elif (h == 11 and m >= 30) or (h >= 12 and h < 15):
                    last_slot = "N"
                else:
                    last_slot = "P"

            h, m = now.hour, now.minute
            if h < 9 or (h == 9 and m < 15):
                q_slot = "X"
            elif (h == 9 and m >= 15) or (h == 10) or (h == 11 and m < 30):
                q_slot = "M"
            elif (h == 11 and m >= 30) or (h >= 12 and h < 15):
                q_slot = "N"
            else:
                q_slot = "P"

            b_date = now.date()
            diff_days = (b_date - a_date).days
            slot_order = {"X": 0, "M": 1, "N": 2, "P": 3}
            need_update = False

            if a_weekday == 4 and diff_days <= 2 and diff_days > 0:
                if last_slot == "P":
                    _log(f"  SKIP: thứ 6 đã hoàn tất (slot P)")
                else:
                    need_update = True
                    _log(f"  Thứ 6 chưa hoàn tất (slot '{last_slot}'), update")

            elif diff_days == 0:
                if slot_order.get(q_slot, 0) > slot_order.get(last_slot, 0):
                    need_update = True
                    _log(f"  Cùng ngày, slot mới '{q_slot}' (cuối: '{last_slot}')")
                else:
                    _log(f"  SKIP: đã update slot '{last_slot}' hôm nay")

            elif diff_days == 1 and a_weekday <= 3 and last_slot == "P":
                if slot_order.get(q_slot, 0) > 0:
                    need_update = True
                    _log(f"  Ngày mới (A=T{a_weekday+2} P→X), update slot '{q_slot}'")
                else:
                    _log(f"  SKIP: ngày mới nhưng slot X (sàn chưa mở)")

            elif diff_days >= 2 and last_slot == "P":
                if slot_order.get(q_slot, 0) > 0:
                    need_update = True
                    _log(f"  Data cũ {diff_days} ngày (A hoàn tất), update slot '{q_slot}'")
                else:
                    need_update = False
                    from models.data_models import UpdateResult as _UR
                    _dummy_result = _UR(total_symbols=0, success_count=0, failed_symbols=[], errors={}, duration_seconds=0)
                    save_last_update_timestamp(_dummy_result, trading_weekday=now.weekday(), slot="X")
                    _log(f"  Cập nhật trạng thái: T{now.weekday()+2} X (sàn chưa mở, data T6 đã đủ)")

            elif diff_days >= 1 and last_slot != "P":
                need_update = True
                _log(f"  Ngày cũ chưa hoàn tất (slot '{last_slot}', diff={diff_days}), update")

            else:
                _log(f"  SKIP: diff={diff_days}, slot={last_slot}/{q_slot}")

        except (ValueError, TypeError):
            pass

    if need_update:
        _log("  Bắt đầu update data...")
        from engine.data_pipeline import DataPipeline

        update_start_time = datetime.now()
        pipeline = DataPipeline(data_dir="data")
        # Truyền callback để có thể dừng giữa chừng
        result = pipeline.update_all(should_stop=_should_stop)
        save_last_update_timestamp(result, override_timestamp=update_start_time)

        _log(f"  DONE: {result.success_count}/{result.total_symbols} OK, "
             f"{len(result.failed_symbols)} fail, {result.duration_seconds:.1f}s")
        _write_status({
            "state": "running", "step": "update_data_done",
            "message": f"Data: {result.success_count}/{result.total_symbols} OK",
            "timestamp": datetime.now().isoformat(),
        })
