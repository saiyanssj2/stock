"""
Session checkpoint manager for training resume capability.

Provides:
- SessionCheckpoint: dataclass representing persisted session state
- SessionCheckpointManager: handles atomic save/load/delete of checkpoint files

The checkpoint enables resuming interrupted training sessions from the last
completed symbol, preventing loss of hours of training work.

Key behaviors:
- Atomic writes via temp file + os.replace to prevent corruption on crash
- Graceful handling of corrupt/missing checkpoint files (returns None, deletes)
- JSON-based storage at engine/models/training_session_checkpoint.json

Requirements: 5.1, 5.2, 5.3, 5.6, 5.7, 5.8
"""

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ==============================================================================
# SessionCheckpoint Dataclass
# ==============================================================================


@dataclass
class SessionCheckpoint:
    """Persisted session state for resume capability.

    Contains all information needed to resume a training session from
    the last completed symbol after an interruption.

    Attributes:
        completed_symbols: List of symbol names that finished training.
        pending_symbols: List of symbol names still awaiting training.
        mode: Training mode ("full" or "incremental").
        session_start_time: ISO format timestamp of when the session started.
        symbol_durations: Mapping of symbol name to training duration in seconds.
        total_epochs_per_symbol: Number of epochs configured per symbol.
        checkpoint_version: Schema version for forward compatibility.
    """

    completed_symbols: List[str]
    pending_symbols: List[str]
    mode: str
    session_start_time: str
    symbol_durations: Dict[str, float]
    total_epochs_per_symbol: int
    checkpoint_version: int = 1


# ==============================================================================
# SessionCheckpointManager
# ==============================================================================


# Required fields that must be present in a valid checkpoint JSON
_REQUIRED_FIELDS = (
    "completed_symbols",
    "pending_symbols",
    "mode",
    "session_start_time",
    "symbol_durations",
)


class SessionCheckpointManager:
    """Manages session checkpoint file with atomic writes.

    Handles persistence of training session state to disk for crash recovery.
    Uses atomic writes (temp file + os.replace) to prevent corruption if the
    process crashes during a write operation.

    Requirements: 5.1, 5.2, 5.3, 5.6, 5.7, 5.8
    """

    CHECKPOINT_PATH = "engine/models/training_session_checkpoint.json"

    def __init__(self, base_dir: str = "."):
        """Initialize SessionCheckpointManager.

        Args:
            base_dir: Project root directory. The checkpoint file path is
                resolved relative to this directory.
        """
        self._base_dir = Path(base_dir)
        self._checkpoint_path = self._base_dir / self.CHECKPOINT_PATH

    @property
    def checkpoint_path(self) -> Path:
        """Resolved absolute path to the checkpoint file."""
        return self._checkpoint_path

    def save(self, checkpoint: SessionCheckpoint) -> None:
        """Atomically save checkpoint to disk.

        Writes to a temporary file first, then uses os.replace for an
        atomic rename. This ensures a crash during writing does not
        corrupt the existing checkpoint.

        Args:
            checkpoint: SessionCheckpoint instance to persist.

        Requirement 5.3: Atomic write via temp file + os.replace.
        """
        # Ensure the parent directory exists
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(checkpoint)

        # Write to a temp file in the same directory, then atomically replace
        dir_path = str(self._checkpoint_path.parent)
        try:
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp",
                prefix="checkpoint_",
                dir=dir_path,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception:
                # Close fd if os.fdopen failed (unlikely but safe)
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise

            # Atomic replace
            os.replace(tmp_path, str(self._checkpoint_path))
            logger.debug(f"Checkpoint saved: {self._checkpoint_path}")

        except Exception as e:
            # Clean up temp file if it still exists
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except (OSError, UnboundLocalError):
                pass
            logger.warning(f"Failed to save checkpoint: {e}")

    def load(self) -> Optional[SessionCheckpoint]:
        """Load checkpoint from disk if it exists and is valid.

        Validates that:
        1. The file exists and can be read
        2. The content is valid JSON
        3. All required fields are present

        If the file is corrupt or missing required fields, it is deleted
        and None is returned.

        Returns:
            SessionCheckpoint if valid file exists, None otherwise.

        Requirement 5.7: Return None and delete file if corrupt.
        """
        if not self._checkpoint_path.exists():
            return None

        try:
            with open(self._checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            logger.warning(
                f"Checkpoint file corrupt (cannot parse): {e}. Deleting."
            )
            self._safe_delete()
            return None

        # Validate required fields
        if not isinstance(data, dict):
            logger.warning("Checkpoint file corrupt (not a JSON object). Deleting.")
            self._safe_delete()
            return None

        for field_name in _REQUIRED_FIELDS:
            if field_name not in data:
                logger.warning(
                    f"Checkpoint file missing required field '{field_name}'. Deleting."
                )
                self._safe_delete()
                return None

        # Construct SessionCheckpoint from validated data
        try:
            checkpoint = SessionCheckpoint(
                completed_symbols=data["completed_symbols"],
                pending_symbols=data["pending_symbols"],
                mode=data["mode"],
                session_start_time=data["session_start_time"],
                symbol_durations=data["symbol_durations"],
                total_epochs_per_symbol=data.get("total_epochs_per_symbol", 100),
                checkpoint_version=data.get("checkpoint_version", 1),
            )
            return checkpoint
        except (TypeError, KeyError, ValueError) as e:
            logger.warning(f"Checkpoint file has invalid field values: {e}. Deleting.")
            self._safe_delete()
            return None

    def delete(self) -> None:
        """Delete the checkpoint file.

        Called when a training session completes successfully or when
        the user chooses to start fresh. Handles the case where the
        file doesn't exist or cannot be deleted gracefully.

        Requirement 5.6: Delete checkpoint on completion.
        Requirement 5.8: Log warning if deletion fails.
        """
        self._safe_delete()

    def exists(self) -> bool:
        """Check if a valid checkpoint file exists.

        Returns:
            True if the checkpoint file exists on disk, False otherwise.
        """
        return self._checkpoint_path.exists()

    def _safe_delete(self) -> None:
        """Delete the checkpoint file with graceful error handling.

        Logs a warning if the file cannot be deleted but does not
        raise an exception.
        """
        try:
            if self._checkpoint_path.exists():
                self._checkpoint_path.unlink()
                logger.debug(f"Checkpoint file deleted: {self._checkpoint_path}")
        except OSError as e:
            logger.warning(
                f"Could not delete checkpoint file {self._checkpoint_path}: {e}"
            )
