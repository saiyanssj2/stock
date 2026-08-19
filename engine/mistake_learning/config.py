"""
Cấu hình cho Mistake-Driven Learning feature.

Định nghĩa MistakeLearningConfig dataclass với validation cho tất cả parameters.
Sử dụng ConfigError từ engine.config khi giá trị nằm ngoài valid range.
"""

from dataclasses import dataclass

from engine.config import ConfigError


@dataclass
class MistakeLearningConfig:
    """Cấu hình cho toàn bộ Mistake-Driven Learning feature.

    Mỗi parameter có default value và valid range. Nếu giá trị nằm ngoài
    valid range, __post_init__ sẽ raise ConfigError.
    """

    # Phân loại trade: ngưỡng PnL để coi là bad trade
    bad_trade_threshold: float = -0.05  # Valid range: [-0.50, -0.01]

    # Ngưỡng nhận diện missed opportunity
    missed_opportunity_threshold: float = 0.05

    # Số ngày nhìn trước để phát hiện missed opportunities
    lookforward_days: int = 5

    # Hệ số weight cho hard examples khi retrain
    weight_multiplier: float = 3.0  # Valid range: [2.0, 5.0]

    # Tỷ lệ tối đa hard examples trong training set
    max_hard_examples_ratio: float = 0.3  # Valid range: [0.05, 0.50]

    # Ngưỡng loss tối thiểu để trigger label correction
    correction_threshold: float = -0.05

    # Ngưỡng confidence tối thiểu để sửa label
    min_confidence_for_correction: float = 0.5

    # Số lượng tối đa anti-patterns lưu trữ
    max_patterns: int = 100

    # Ngưỡng cosine similarity để merge patterns
    pattern_similarity_threshold: float = 0.8  # Valid range: [0.5, 1.0]

    # Số cycles trước khi pattern hết hạn
    pattern_expiry_cycles: int = 20  # Valid range: [1, 100]

    # Bật/tắt toàn bộ feedback loop
    enabled: bool = True

    # Số trades tối thiểu để chạy analysis
    min_trades_for_analysis: int = 10  # Valid range: [1, 1000]

    # Số epochs tối đa cho feedback retrain
    max_retrain_epochs: int = 5  # Valid range: [1, 50]

    def __post_init__(self) -> None:
        """Validate tất cả parameters sau khi khởi tạo."""
        self._validate_bad_trade_threshold()
        self._validate_weight_multiplier()
        self._validate_max_hard_examples_ratio()
        self._validate_pattern_similarity_threshold()
        self._validate_pattern_expiry_cycles()
        self._validate_min_trades_for_analysis()
        self._validate_max_retrain_epochs()

    def _validate_bad_trade_threshold(self) -> None:
        """Kiểm tra bad_trade_threshold trong [-0.50, -0.01]."""
        if not (-0.50 <= self.bad_trade_threshold <= -0.01):
            raise ConfigError(
                f"bad_trade_threshold={self.bad_trade_threshold} nằm ngoài "
                f"valid range [-0.50, -0.01]",
                error_code="CONFIG_ERROR",
                details={
                    "parameter": "bad_trade_threshold",
                    "value": self.bad_trade_threshold,
                    "valid_range": [-0.50, -0.01],
                },
            )

    def _validate_weight_multiplier(self) -> None:
        """Kiểm tra weight_multiplier trong [2.0, 5.0]."""
        if not (2.0 <= self.weight_multiplier <= 5.0):
            raise ConfigError(
                f"weight_multiplier={self.weight_multiplier} nằm ngoài "
                f"valid range [2.0, 5.0]",
                error_code="CONFIG_ERROR",
                details={
                    "parameter": "weight_multiplier",
                    "value": self.weight_multiplier,
                    "valid_range": [2.0, 5.0],
                },
            )

    def _validate_max_hard_examples_ratio(self) -> None:
        """Kiểm tra max_hard_examples_ratio trong [0.05, 0.50]."""
        if not (0.05 <= self.max_hard_examples_ratio <= 0.50):
            raise ConfigError(
                f"max_hard_examples_ratio={self.max_hard_examples_ratio} nằm ngoài "
                f"valid range [0.05, 0.50]",
                error_code="CONFIG_ERROR",
                details={
                    "parameter": "max_hard_examples_ratio",
                    "value": self.max_hard_examples_ratio,
                    "valid_range": [0.05, 0.50],
                },
            )

    def _validate_pattern_similarity_threshold(self) -> None:
        """Kiểm tra pattern_similarity_threshold trong [0.5, 1.0]."""
        if not (0.5 <= self.pattern_similarity_threshold <= 1.0):
            raise ConfigError(
                f"pattern_similarity_threshold={self.pattern_similarity_threshold} "
                f"nằm ngoài valid range [0.5, 1.0]",
                error_code="CONFIG_ERROR",
                details={
                    "parameter": "pattern_similarity_threshold",
                    "value": self.pattern_similarity_threshold,
                    "valid_range": [0.5, 1.0],
                },
            )

    def _validate_pattern_expiry_cycles(self) -> None:
        """Kiểm tra pattern_expiry_cycles trong [1, 100]."""
        if not (1 <= self.pattern_expiry_cycles <= 100):
            raise ConfigError(
                f"pattern_expiry_cycles={self.pattern_expiry_cycles} nằm ngoài "
                f"valid range [1, 100]",
                error_code="CONFIG_ERROR",
                details={
                    "parameter": "pattern_expiry_cycles",
                    "value": self.pattern_expiry_cycles,
                    "valid_range": [1, 100],
                },
            )

    def _validate_min_trades_for_analysis(self) -> None:
        """Kiểm tra min_trades_for_analysis trong [1, 1000]."""
        if not (1 <= self.min_trades_for_analysis <= 1000):
            raise ConfigError(
                f"min_trades_for_analysis={self.min_trades_for_analysis} nằm ngoài "
                f"valid range [1, 1000]",
                error_code="CONFIG_ERROR",
                details={
                    "parameter": "min_trades_for_analysis",
                    "value": self.min_trades_for_analysis,
                    "valid_range": [1, 1000],
                },
            )

    def _validate_max_retrain_epochs(self) -> None:
        """Kiểm tra max_retrain_epochs trong [1, 50]."""
        if not (1 <= self.max_retrain_epochs <= 50):
            raise ConfigError(
                f"max_retrain_epochs={self.max_retrain_epochs} nằm ngoài "
                f"valid range [1, 50]",
                error_code="CONFIG_ERROR",
                details={
                    "parameter": "max_retrain_epochs",
                    "value": self.max_retrain_epochs,
                    "valid_range": [1, 50],
                },
            )
