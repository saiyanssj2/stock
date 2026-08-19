"""
Training Trigger Monitor — Theo dõi và ghi log mọi lần training được khởi tạo.

Ghi nhận nguồn trigger (user/auto/scheduled/hook), append-only JSONL storage,
FIFO rotation khi vượt 10.000 entries.

Requirements:
- 1.1: Log nguồn trigger kèm timestamp ISO-8601 và session ID duy nhất
- 1.2: Ghi nhận "auto_cycle_triggered" với cycle_number và elapsed_since_last_cycle
- 1.3: Ghi nhận caller context (tên component, trigger_source)
- 1.4: Append-only JSONL, 10.000 entries max, FIFO rotation
- 1.5: Tổng hợp trong khoảng thời gian, trả kết quả trong 5 giây
- 1.6: I/O error → log warning, không chặn training
- 1.7: Pre-training health check → block training nếu I/O lỗi
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

from engine.diagnostics.models import TriggerEvent, TriggerSource

logger = logging.getLogger(__name__)


class DiagnosticsIOError(Exception):
    """Lỗi I/O trong diagnostics module — dùng để block training (Req 1.7)."""

    pass


class TrainingTriggerMonitor:
    """Monitor append-only log cho training triggers.

    Giới hạn 10.000 entries, FIFO rotation.
    Khi I/O lỗi trong quá trình ghi log → warning, không chặn training (Req 1.6).
    Khi I/O lỗi trước khi training bắt đầu → block training (Req 1.7).
    """

    MAX_ENTRIES: int = 10_000

    def __init__(self, log_path: str = "data/engine/diagnostics/trigger_log.jsonl"):
        """Khởi tạo monitor với đường dẫn JSONL log file.

        Args:
            log_path: Đường dẫn tới file JSONL lưu trigger events.
        """
        self._log_path = Path(log_path)
        self._io_healthy = True

        # Đảm bảo thư mục output tồn tại
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"Không thể tạo thư mục diagnostics: {e}")
            self._io_healthy = False

    def log_trigger(self, event: TriggerEvent) -> bool:
        """Ghi trigger event vào JSONL log file.

        Returns True nếu ghi thành công, False nếu gặp lỗi I/O.
        Khi lỗi I/O: ghi cảnh báo vào app logger, không chặn training (Req 1.6).

        Args:
            event: TriggerEvent cần ghi.

        Returns:
            True nếu ghi thành công, False nếu lỗi.
        """
        try:
            # Serialize event thành JSON line
            entry = self._serialize_event(event)
            line = json.dumps(entry, ensure_ascii=False) + "\n"

            # Append vào file
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line)

            # Kiểm tra và rotate nếu cần (Req 1.4)
            self._rotate_if_needed()

            self._io_healthy = True
            return True

        except OSError as e:
            # Req 1.6: I/O error → log warning, không chặn training
            logger.warning(
                f"Không thể ghi trigger log do lỗi I/O: {e}"
            )
            self._io_healthy = False
            return False

    def is_healthy(self) -> bool:
        """Kiểm tra I/O có đang hoạt động bình thường không.

        Thực hiện probe write để verify khả năng ghi (Req 1.7).

        Returns:
            True nếu I/O hoạt động bình thường, False nếu có vấn đề.
        """
        try:
            # Probe: thử tạo file tạm trong cùng thư mục để verify write access
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            probe_path = self._log_path.parent / ".health_probe"
            probe_path.write_text("ok", encoding="utf-8")
            probe_path.unlink(missing_ok=True)
            self._io_healthy = True
            return True
        except OSError:
            self._io_healthy = False
            return False

    def get_triggers_in_range(
        self, start_time: datetime, end_time: datetime
    ) -> List[TriggerEvent]:
        """Lấy triggers trong khoảng thời gian [start_time, end_time].

        Args:
            start_time: Thời điểm bắt đầu (inclusive).
            end_time: Thời điểm kết thúc (inclusive).

        Returns:
            Danh sách TriggerEvent trong khoảng thời gian.
        """
        results: List[TriggerEvent] = []

        if not self._log_path.exists():
            return results

        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        event_time = datetime.fromisoformat(entry["timestamp"])
                        if start_time <= event_time <= end_time:
                            event = self._deserialize_event(entry)
                            results.append(event)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # Bỏ qua dòng bị lỗi
                        continue
        except OSError as e:
            logger.warning(f"Không thể đọc trigger log: {e}")

        return results

    def get_summary(
        self, start_time: datetime, end_time: datetime
    ) -> dict:
        """Tổng hợp số lần trigger theo từng nguồn trong khoảng thời gian (Req 1.5).

        Args:
            start_time: Thời điểm bắt đầu (inclusive).
            end_time: Thời điểm kết thúc (inclusive).

        Returns:
            Dict với key là trigger_source (str), value là số lần trigger.
        """
        triggers = self.get_triggers_in_range(start_time, end_time)

        # Khởi tạo summary với tất cả sources = 0
        summary: dict = {source.value: 0 for source in TriggerSource}

        for event in triggers:
            source_value = event.trigger_source.value
            summary[source_value] = summary.get(source_value, 0) + 1

        return {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_triggers": len(triggers),
            "by_source": summary,
        }

    def _rotate_if_needed(self) -> None:
        """Xóa entries cũ nhất nếu vượt MAX_ENTRIES (Req 1.4 - FIFO).

        Đọc tất cả entries, giữ lại MAX_ENTRIES entries mới nhất,
        ghi lại file.
        """
        try:
            if not self._log_path.exists():
                return

            # Đọc tất cả dòng
            with open(self._log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Nếu chưa vượt giới hạn → không cần rotate
            if len(lines) <= self.MAX_ENTRIES:
                return

            # Giữ lại MAX_ENTRIES dòng mới nhất (cuối file = mới nhất)
            lines_to_keep = lines[-self.MAX_ENTRIES:]

            # Ghi lại file atomic: write to temp rồi rename
            temp_fd, temp_path = tempfile.mkstemp(
                dir=str(self._log_path.parent),
                suffix=".jsonl.tmp",
            )
            try:
                with os.fdopen(temp_fd, "w", encoding="utf-8") as tmp_f:
                    tmp_f.writelines(lines_to_keep)

                # Atomic replace (Windows: cần xóa trước nếu tồn tại)
                temp_path_obj = Path(temp_path)
                if os.name == "nt":
                    self._log_path.unlink(missing_ok=True)
                temp_path_obj.replace(self._log_path)
            except Exception:
                # Cleanup temp file nếu có lỗi
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise

        except OSError as e:
            logger.warning(f"Không thể rotate trigger log: {e}")

    def _serialize_event(self, event: TriggerEvent) -> dict:
        """Chuyển TriggerEvent thành dict để serialize JSON.

        Args:
            event: TriggerEvent cần serialize.

        Returns:
            Dict phù hợp để ghi JSON.
        """
        return {
            "session_id": event.session_id,
            "timestamp": event.timestamp,
            "trigger_source": event.trigger_source.value,
            "caller_component": event.caller_component,
            "metadata": event.metadata,
        }

    def _deserialize_event(self, entry: dict) -> TriggerEvent:
        """Chuyển dict từ JSON thành TriggerEvent.

        Args:
            entry: Dict đọc từ JSON line.

        Returns:
            TriggerEvent instance.
        """
        return TriggerEvent(
            session_id=entry["session_id"],
            timestamp=entry["timestamp"],
            trigger_source=TriggerSource(entry["trigger_source"]),
            caller_component=entry["caller_component"],
            metadata=entry.get("metadata", {}),
        )

    def _count_entries(self) -> int:
        """Đếm số entries hiện tại trong log file.

        Returns:
            Số dòng (entries) trong file, 0 nếu file không tồn tại.
        """
        if not self._log_path.exists():
            return 0

        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return 0
