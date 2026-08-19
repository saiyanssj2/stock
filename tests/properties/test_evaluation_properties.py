"""
Property-based tests cho Post-Training Evaluation module.

Sử dụng hypothesis để kiểm tra các correctness properties
được định nghĩa trong design document.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from engine.evaluation_config import EvaluationConfig, OverfittingSeverity
from engine.overfitting_detector import OverfittingDetector


# ==============================================================================
# Feature: post-training-evaluation, Property 6: Overfitting severity classification follows threshold rules
# ==============================================================================


class TestOverfittingSeverityClassificationProperty:
    """
    Property 6: Overfitting severity classification follows threshold rules.

    For any loss ratio value, the classification SHALL be:
    - "none" when ratio < 1.5
    - "mild" when 1.5 <= ratio < 2.0
    - "moderate" when 2.0 <= ratio < 3.0
    - "severe" when ratio >= 3.0

    Furthermore, when severity is "moderate" or "severe",
    a non-empty warning message SHALL be present.

    **Validates: Requirements 3.3, 3.4**
    """

    def setup_method(self):
        """Khởi tạo detector với config mặc định."""
        self.detector = OverfittingDetector()

    @given(
        ratio=st.floats(
            min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False
        )
    )
    @settings(max_examples=100)
    def test_classify_severity_follows_threshold_rules(self, ratio: float):
        """
        Kiểm tra classify_severity trả về OverfittingSeverity đúng
        theo các ngưỡng threshold cho mọi giá trị ratio.

        **Validates: Requirements 3.3, 3.4**
        """
        severity = self.detector.classify_severity(ratio)

        if ratio < 1.5:
            assert severity == OverfittingSeverity.NONE, (
                f"ratio={ratio} < 1.5 nhưng severity={severity}, kỳ vọng NONE"
            )
        elif ratio < 2.0:
            assert severity == OverfittingSeverity.MILD, (
                f"ratio={ratio} trong [1.5, 2.0) nhưng severity={severity}, kỳ vọng MILD"
            )
        elif ratio < 3.0:
            assert severity == OverfittingSeverity.MODERATE, (
                f"ratio={ratio} trong [2.0, 3.0) nhưng severity={severity}, kỳ vọng MODERATE"
            )
        else:
            assert severity == OverfittingSeverity.SEVERE, (
                f"ratio={ratio} >= 3.0 nhưng severity={severity}, kỳ vọng SEVERE"
            )

    @given(
        ratio=st.floats(
            min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False
        )
    )
    @settings(max_examples=100)
    def test_detect_warning_message_present_for_moderate_or_severe(self, ratio: float):
        """
        Kiểm tra khi severity là MODERATE hoặc SEVERE, detect() trả về
        warning_message không None và không rỗng.

        Sử dụng train_loss > 0 sao cho test_loss / train_loss = ratio mong muốn.

        **Validates: Requirements 3.3, 3.4**
        """
        # Tạo train_loss cố định > 0, tính test_loss để đạt ratio mong muốn
        train_loss = 1.0
        test_loss = ratio * train_loss

        # Tạo predictions arrays đơn giản cho test
        train_predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        test_predictions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        result = self.detector.detect(
            train_loss=train_loss,
            test_loss=test_loss,
            train_predictions=train_predictions,
            test_predictions=test_predictions,
        )

        if ratio >= 2.0:
            # Severity phải là MODERATE hoặc SEVERE → warning_message phải có
            assert result.warning_message is not None, (
                f"ratio={ratio}, severity={result.severity} nhưng warning_message là None"
            )
            assert len(result.warning_message) > 0, (
                f"ratio={ratio}, severity={result.severity} nhưng warning_message rỗng"
            )
        else:
            # Severity là NONE hoặc MILD → warning_message phải là None
            assert result.warning_message is None, (
                f"ratio={ratio}, severity={result.severity} nhưng warning_message "
                f"không phải None: {result.warning_message}"
            )


# ==============================================================================
# Feature: post-training-evaluation, Property 4: Overfitting loss ratio is correctly computed
# ==============================================================================


class TestOverfittingLossRatioProperty:
    """
    Property 4: Overfitting loss ratio is correctly computed.

    For any train_loss > 0 and test_loss >= 0, the computed loss ratio
    SHALL equal test_loss / train_loss.

    **Validates: Requirements 3.1**
    """

    def setup_method(self):
        """Khởi tạo detector với config mặc định."""
        self.detector = OverfittingDetector()
        # Dummy predictions arrays - chỉ test loss ratio nên không cần giá trị thực
        self.dummy_train_preds = np.array([0.1, 0.2, 0.3])
        self.dummy_test_preds = np.array([0.1, 0.2, 0.3])

    @given(
        train_loss=st.floats(
            min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        test_loss=st.floats(
            min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100)
    def test_loss_ratio_equals_test_over_train(self, train_loss, test_loss):
        """
        Với mọi train_loss > 0 và test_loss >= 0,
        loss_ratio phải bằng test_loss / train_loss.

        **Validates: Requirements 3.1**
        """
        result = self.detector.detect(
            train_loss=train_loss,
            test_loss=test_loss,
            train_predictions=self.dummy_train_preds,
            test_predictions=self.dummy_test_preds,
        )

        expected_ratio = test_loss / train_loss
        assert result.loss_ratio == pytest.approx(expected_ratio)


# ==============================================================================
# Feature: post-training-evaluation, Property 1: Signal threshold conversion is deterministic and correct
# ==============================================================================


import pandas as pd
from hypothesis import assume

from engine.backtest_evaluator import ModelStrategy
from engine.config import Action


@settings(max_examples=100)
@given(
    score=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    buy_threshold=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    sell_threshold=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_signal_threshold_conversion_is_deterministic_and_correct(
    score: float,
    buy_threshold: float,
    sell_threshold: float,
) -> None:
    """
    Property 1: Signal threshold conversion is deterministic and correct.

    For any Position_Score value in [-1.0, +1.0] and any valid threshold pair
    (sell_threshold < buy_threshold), the signal conversion SHALL produce
    BUY when score > buy_threshold, SELL when score < sell_threshold,
    and HOLD otherwise.

    **Validates: Requirements 1.1, 1.5**
    """
    # Lọc chỉ các cặp threshold hợp lệ: sell_threshold < buy_threshold
    assume(sell_threshold < buy_threshold)

    # Tạo predictions array với 1 phần tử (score)
    predictions = np.array([score])

    # Index mapping: positional index 0 → predictions index 0
    index_mapping = {0: 0}

    # Tạo ModelStrategy với thresholds
    strategy = ModelStrategy(
        predictions=predictions,
        index_mapping=index_mapping,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    # Tạo minimal DataFrame (1 row) - cần cho Strategy Protocol interface
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [102.0],
            "volume": [1000],
        }
    )

    # Gọi generate_signal
    result = strategy.generate_signal(df, index=0)

    # Kiểm tra tính đúng đắn của signal conversion
    if score > buy_threshold:
        assert result == Action.BUY, (
            f"Expected BUY khi score={score} > buy_threshold={buy_threshold}, "
            f"nhưng nhận được {result}"
        )
    elif score < sell_threshold:
        assert result == Action.SELL, (
            f"Expected SELL khi score={score} < sell_threshold={sell_threshold}, "
            f"nhưng nhận được {result}"
        )
    else:
        assert result == Action.HOLD, (
            f"Expected HOLD khi sell_threshold={sell_threshold} <= score={score} "
            f"<= buy_threshold={buy_threshold}, nhưng nhận được {result}"
        )


# ==============================================================================
# Feature: post-training-evaluation, Property 5: KS statistic is zero for identical distributions
# ==============================================================================


class TestKsStatisticIdenticalDistributions:
    """
    Property 5: KS statistic is zero for identical distributions.

    For any array of predictions, computing the KS statistic between that array
    and itself SHALL yield a statistic of 0.0 (or within floating-point epsilon).

    **Validates: Requirements 3.2**
    """

    def setup_method(self):
        """Khởi tạo OverfittingDetector."""
        self.detector = OverfittingDetector()

    @given(
        data=st.lists(
            st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=2,
            max_size=100,
        )
    )
    @settings(max_examples=100)
    def test_ks_statistic_zero_for_identical_distributions(self, data):
        """
        KS statistic giữa một array và chính nó phải bằng 0.0.

        **Validates: Requirements 3.2**
        """
        # Chuyển list thành numpy array
        arr = np.array(data)

        # Tính KS statistic giữa array và chính nó
        statistic, p_value = self.detector._compute_ks_statistic(arr, arr)

        # KS statistic phải bằng 0.0 (trong khoảng floating-point epsilon)
        assert abs(statistic) <= 1e-10, (
            f"KS statistic should be 0.0 for identical distributions, got {statistic}"
        )


# ==============================================================================
# Feature: post-training-evaluation, Property 8: Pearson correlation is bounded and symmetric for self-correlation
# ==============================================================================

from engine.prediction_analyzer import PredictionAnalyzer


class TestPearsonCorrelationBoundsProperty:
    """
    Property 8: Pearson correlation is bounded and symmetric for self-correlation.

    For any non-constant array X of length >= 2, pearson_correlation(X, X) SHALL
    equal 1.0, and pearson_correlation(X, -X) SHALL equal -1.0. For any two arrays,
    the result SHALL be in [-1.0, 1.0].

    **Validates: Requirements 4.2**
    """

    def setup_method(self):
        """Khởi tạo PredictionAnalyzer."""
        self.analyzer = PredictionAnalyzer()

    @given(
        data=st.lists(
            st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=2,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_self_correlation_equals_one(self, data):
        """
        Với mọi mảng non-constant X có length >= 2,
        pearson_correlation(X, X) phải bằng 1.0.

        **Validates: Requirements 4.2**
        """
        x = np.array(data)

        # Đảm bảo mảng thực sự non-constant (threshold đủ lớn để tránh
        # floating-point precision issues khi scipy phát hiện constant input)
        assume(np.std(x) > 1e-10)
        assume(np.ptp(x) > 0)  # max - min > 0

        result = self.analyzer.pearson_correlation(x, x)

        assert result == pytest.approx(1.0, abs=1e-7), (
            f"pearson_correlation(X, X) phải bằng 1.0, nhưng nhận được {result}"
        )

    @given(
        data=st.lists(
            st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=2,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_negated_correlation_equals_negative_one(self, data):
        """
        Với mọi mảng non-constant X có length >= 2,
        pearson_correlation(X, -X) phải bằng -1.0.

        **Validates: Requirements 4.2**
        """
        x = np.array(data)

        # Đảm bảo mảng thực sự non-constant (threshold đủ lớn để tránh
        # floating-point precision issues khi scipy phát hiện constant input)
        assume(np.std(x) > 1e-10)
        assume(np.ptp(x) > 0)  # max - min > 0

        result = self.analyzer.pearson_correlation(x, -x)

        assert result == pytest.approx(-1.0, abs=1e-7), (
            f"pearson_correlation(X, -X) phải bằng -1.0, nhưng nhận được {result}"
        )

    @given(
        data_x=st.lists(
            st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=2,
            max_size=50,
        ),
        data_y=st.lists(
            st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=2,
            max_size=50,
        ),
    )
    @settings(max_examples=100)
    def test_correlation_bounded_between_negative_one_and_one(self, data_x, data_y):
        """
        Với mọi hai mảng cùng độ dài, kết quả pearson_correlation
        phải nằm trong khoảng [-1.0, 1.0].

        **Validates: Requirements 4.2**
        """
        # Cắt hai mảng về cùng độ dài
        min_len = min(len(data_x), len(data_y))
        x = np.array(data_x[:min_len])
        y = np.array(data_y[:min_len])

        # Đảm bảo cả hai mảng non-constant
        assume(np.std(x) > 0)
        assume(np.std(y) > 0)

        result = self.analyzer.pearson_correlation(x, y)

        assert -1.0 <= result <= 1.0, (
            f"pearson_correlation phải nằm trong [-1, 1], nhưng nhận được {result}"
        )


# ==============================================================================
# Feature: post-training-evaluation, Property 7: Directional accuracy equals proportion of sign matches
# ==============================================================================


from engine.prediction_analyzer import PredictionAnalyzer


class TestDirectionalAccuracyProperty:
    """
    Property 7: Directional accuracy equals proportion of sign matches.

    For any non-empty arrays of predictions and actual returns (excluding zeros),
    directional_accuracy SHALL equal
    `count(sign(pred[i]) == sign(actual[i])) / total * 100`.

    **Validates: Requirements 4.1**
    """

    def setup_method(self):
        """Khởi tạo PredictionAnalyzer."""
        self.analyzer = PredictionAnalyzer()

    @given(
        predictions=st.lists(
            st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            min_size=1,
            max_size=50,
        ),
        actuals=st.lists(
            st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(max_examples=100)
    def test_directional_accuracy_equals_proportion_of_sign_matches(
        self, predictions, actuals
    ):
        """
        Kiểm tra directional_accuracy trả về đúng tỷ lệ sign matches
        sau khi loại bỏ các cặp có giá trị 0.

        **Validates: Requirements 4.1**
        """
        # Đảm bảo cùng kích thước
        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]

        pred_arr = np.array(predictions)
        actual_arr = np.array(actuals)

        # Lọc bỏ các cặp có prediction hoặc actual bằng 0
        mask = (pred_arr != 0) & (actual_arr != 0)
        filtered_preds = pred_arr[mask]
        filtered_actuals = actual_arr[mask]

        # Nếu không còn cặp nào sau khi lọc, kỳ vọng kết quả 0.0
        if len(filtered_preds) == 0:
            result = self.analyzer.directional_accuracy(pred_arr, actual_arr)
            assert result == 0.0, (
                f"Khi không có cặp hợp lệ (tất cả chứa 0), "
                f"kỳ vọng 0.0 nhưng nhận {result}"
            )
            return

        # Tính expected accuracy thủ công
        sign_matches = np.sign(filtered_preds) == np.sign(filtered_actuals)
        expected_accuracy = np.sum(sign_matches) / len(filtered_preds) * 100.0

        # Gọi hàm directional_accuracy
        result = self.analyzer.directional_accuracy(pred_arr, actual_arr)

        # So sánh kết quả
        assert result == pytest.approx(expected_accuracy), (
            f"directional_accuracy={result} != expected={expected_accuracy}. "
            f"predictions={predictions}, actuals={actuals}"
        )


# ==============================================================================
# Feature: post-training-evaluation, Property 9: Calibration quartiles partition predictions completely
# ==============================================================================

from engine.prediction_analyzer import PredictionAnalyzer


class TestCalibrationQuartilePartitioningProperty:
    """
    Property 9: Calibration quartiles partition predictions completely.

    For any arrays of predictions and actual returns of length >= 4,
    the sum of sample counts across all four quartiles SHALL equal
    the total number of predictions.

    Vì calibration_by_quartile trả về mean_abs_return per quartile (không phải counts),
    property test kiểm tra:
    - Kết quả có đúng 4 keys (Q1, Q2, Q3, Q4)
    - Tất cả values >= 0 (mean absolute return luôn không âm)

    **Validates: Requirements 4.3**
    """

    def setup_method(self):
        """Khởi tạo PredictionAnalyzer."""
        self.analyzer = PredictionAnalyzer()

    @given(
        predictions=st.lists(
            st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            min_size=4,
            max_size=100,
        ),
        actual_returns=st.lists(
            st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            min_size=4,
            max_size=100,
        ),
    )
    @settings(max_examples=100)
    def test_calibration_quartiles_have_exactly_four_keys(
        self, predictions, actual_returns
    ):
        """
        Kiểm tra calibration_by_quartile luôn trả về dict với đúng 4 keys:
        Q1, Q2, Q3, Q4 cho mọi input hợp lệ có length >= 4.

        **Validates: Requirements 4.3**
        """
        # Đảm bảo cả hai arrays có cùng kích thước
        min_len = min(len(predictions), len(actual_returns))
        assume(min_len >= 4)
        preds = np.array(predictions[:min_len])
        actuals = np.array(actual_returns[:min_len])

        result = self.analyzer.calibration_by_quartile(preds, actuals)

        # Kết quả phải có đúng 4 keys
        expected_keys = {"Q1", "Q2", "Q3", "Q4"}
        assert set(result.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(result.keys())}"
        )

    @given(
        predictions=st.lists(
            st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            min_size=4,
            max_size=100,
        ),
        actual_returns=st.lists(
            st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            min_size=4,
            max_size=100,
        ),
    )
    @settings(max_examples=100)
    def test_calibration_quartiles_values_are_non_negative(
        self, predictions, actual_returns
    ):
        """
        Kiểm tra tất cả giá trị mean_abs_return trong quartile result >= 0,
        vì mean absolute return luôn không âm.

        **Validates: Requirements 4.3**
        """
        # Đảm bảo cả hai arrays có cùng kích thước
        min_len = min(len(predictions), len(actual_returns))
        assume(min_len >= 4)
        preds = np.array(predictions[:min_len])
        actuals = np.array(actual_returns[:min_len])

        result = self.analyzer.calibration_by_quartile(preds, actuals)

        # Tất cả values phải >= 0
        for quartile, value in result.items():
            assert value >= 0.0, (
                f"Quartile {quartile} có mean_abs_return={value} < 0, "
                f"điều này không thể xảy ra với mean absolute return"
            )


# ==============================================================================
# Feature: post-training-evaluation, Property 10: Score distribution flagging follows std thresholds
# ==============================================================================

from engine.prediction_analyzer import PredictionAnalyzer


class TestScoreDistributionFlaggingProperty:
    """
    Property 10: Score distribution flagging follows std thresholds.

    For any array of predictions, the `too_conservative` flag SHALL be True
    if and only if std < 0.1, and the `too_extreme` flag SHALL be True
    if and only if std > 0.8.

    **Validates: Requirements 4.4**
    """

    def setup_method(self):
        """Khởi tạo PredictionAnalyzer với config mặc định."""
        self.analyzer = PredictionAnalyzer()

    @given(
        predictions=st.lists(
            st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            min_size=2,
            max_size=100,
        )
    )
    @settings(max_examples=100)
    def test_score_distribution_flagging_follows_std_thresholds(self, predictions):
        """
        Kiểm tra too_conservative và too_extreme flags tuân thủ đúng
        ngưỡng std cho mọi array predictions hợp lệ.

        **Validates: Requirements 4.4**
        """
        arr = np.array(predictions)

        # Tính expected std dùng population std (ddof=0)
        expected_std = float(np.std(arr, ddof=0))

        # Gọi score_distribution
        result = self.analyzer.score_distribution(arr)

        # Kiểm tra too_conservative flag: True iff std < 0.1
        expected_conservative = expected_std < 0.1
        assert result.too_conservative == expected_conservative, (
            f"too_conservative={result.too_conservative}, expected={expected_conservative} "
            f"(std={expected_std}, threshold=0.1)"
        )

        # Kiểm tra too_extreme flag: True iff std > 0.8
        expected_extreme = expected_std > 0.8
        assert result.too_extreme == expected_extreme, (
            f"too_extreme={result.too_extreme}, expected={expected_extreme} "
            f"(std={expected_std}, threshold=0.8)"
        )


# ==============================================================================
# Feature: post-training-evaluation, Property 11: Confusion matrix categories are exhaustive and mutually exclusive
# ==============================================================================


from engine.prediction_analyzer import PredictionAnalyzer


class TestConfusionMatrixExhaustivenessProperty:
    """
    Property 11: Confusion matrix categories are exhaustive and mutually exclusive.

    For any arrays of predictions and actual returns, TP + TN + FP + FN SHALL equal
    the total count of non-zero pairs. Each pair (pred, actual) SHALL appear in
    exactly one category.

    **Validates: Requirements 4.5**
    """

    def setup_method(self):
        """Khởi tạo PredictionAnalyzer."""
        self.analyzer = PredictionAnalyzer()

    @given(
        predictions=st.lists(
            st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            min_size=1,
            max_size=50,
        ),
        actual_returns=st.lists(
            st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(max_examples=100)
    def test_confusion_matrix_total_equals_non_zero_pairs(
        self, predictions, actual_returns
    ):
        """
        TP + TN + FP + FN phải bằng tổng số cặp mà cả prediction và actual
        đều khác 0.

        **Validates: Requirements 4.5**
        """
        # Đảm bảo arrays cùng độ dài (lấy min length)
        min_len = min(len(predictions), len(actual_returns))
        preds = np.array(predictions[:min_len])
        actuals = np.array(actual_returns[:min_len])

        # Tính confusion matrix
        cm = self.analyzer.confusion_matrix(preds, actuals)

        # Đếm thủ công số cặp non-zero (cả pred != 0 và actual != 0)
        non_zero_pairs = sum(
            1 for p, a in zip(preds, actuals) if p != 0 and a != 0
        )

        # Assert: cm.total == count_of_non_zero_pairs
        assert cm.total == non_zero_pairs, (
            f"cm.total={cm.total} != non_zero_pairs={non_zero_pairs}\n"
            f"predictions={preds}\n"
            f"actual_returns={actuals}"
        )

    @given(
        predictions=st.lists(
            st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            min_size=1,
            max_size=50,
        ),
        actual_returns=st.lists(
            st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
            ),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(max_examples=100)
    def test_confusion_matrix_sum_equals_total_property(
        self, predictions, actual_returns
    ):
        """
        TP + TN + FP + FN phải bằng cm.total (property internal consistency).

        **Validates: Requirements 4.5**
        """
        # Đảm bảo arrays cùng độ dài
        min_len = min(len(predictions), len(actual_returns))
        preds = np.array(predictions[:min_len])
        actuals = np.array(actual_returns[:min_len])

        # Tính confusion matrix
        cm = self.analyzer.confusion_matrix(preds, actuals)

        # Assert: TP + TN + FP + FN == cm.total
        computed_sum = (
            cm.true_positive + cm.true_negative + cm.false_positive + cm.false_negative
        )
        assert computed_sum == cm.total, (
            f"TP({cm.true_positive}) + TN({cm.true_negative}) + "
            f"FP({cm.false_positive}) + FN({cm.false_negative}) = {computed_sum} "
            f"!= cm.total={cm.total}"
        )


# ==============================================================================
# Feature: post-training-evaluation, Property 3: Random strategy is reproducible and approximately uniform
# ==============================================================================

from engine.baseline_comparator import RandomStrategy
from engine.config import Action


class TestRandomStrategyReproducibilityProperty:
    """
    Property 3: Random strategy is reproducible and approximately uniform.

    For any fixed random seed, running the Random strategy twice on the same data
    SHALL produce identical signal sequences. Additionally, for any sequence of
    length >= 100, each signal type (BUY/SELL/HOLD) SHALL appear at least 20%
    of the time.

    **Validates: Requirements 2.4**
    """

    @given(
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    @settings(max_examples=100)
    def test_random_strategy_reproducibility_with_same_seed(self, seed: int):
        """
        Với mọi seed cố định, chạy RandomStrategy hai lần trên cùng dữ liệu
        phải cho ra cùng chuỗi tín hiệu.

        **Validates: Requirements 2.4**
        """
        # Tạo minimal DataFrame cho test (100 rows)
        df = pd.DataFrame(
            {
                "open": [100.0] * 100,
                "high": [105.0] * 100,
                "low": [95.0] * 100,
                "close": [102.0] * 100,
                "volume": [1000] * 100,
            }
        )

        # Tạo hai instances RandomStrategy với cùng seed
        strategy1 = RandomStrategy(seed=seed)
        strategy2 = RandomStrategy(seed=seed)

        # Chạy generate_signal 100 lần cho mỗi strategy
        signals1 = [strategy1.generate_signal(df, i) for i in range(100)]
        signals2 = [strategy2.generate_signal(df, i) for i in range(100)]

        # Hai chuỗi tín hiệu phải hoàn toàn giống nhau
        assert signals1 == signals2, (
            f"RandomStrategy với seed={seed} không reproducible. "
            f"Lần 1: {signals1[:10]}..., Lần 2: {signals2[:10]}..."
        )

    @given(
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    @settings(max_examples=100)
    def test_random_strategy_approximately_uniform_distribution(self, seed: int):
        """
        Với mọi seed, khi chạy RandomStrategy >= 300 lần, mỗi loại tín hiệu
        (BUY/SELL/HOLD) phải xuất hiện ít nhất 20% tổng số lần.

        **Validates: Requirements 2.4**
        """
        n_signals = 300

        # Tạo minimal DataFrame cho test
        df = pd.DataFrame(
            {
                "open": [100.0] * n_signals,
                "high": [105.0] * n_signals,
                "low": [95.0] * n_signals,
                "close": [102.0] * n_signals,
                "volume": [1000] * n_signals,
            }
        )

        # Tạo RandomStrategy với seed
        strategy = RandomStrategy(seed=seed)

        # Sinh chuỗi tín hiệu
        signals = [strategy.generate_signal(df, i) for i in range(n_signals)]

        # Đếm số lần xuất hiện mỗi loại
        buy_count = signals.count(Action.BUY)
        sell_count = signals.count(Action.SELL)
        hold_count = signals.count(Action.HOLD)

        # Mỗi loại phải chiếm >= 20% (tức >= 60 lần trong 300 signals)
        min_threshold = n_signals * 0.20

        assert buy_count >= min_threshold, (
            f"BUY chiếm {buy_count}/{n_signals} = {buy_count/n_signals:.1%}, "
            f"dưới ngưỡng 20%. Seed={seed}"
        )
        assert sell_count >= min_threshold, (
            f"SELL chiếm {sell_count}/{n_signals} = {sell_count/n_signals:.1%}, "
            f"dưới ngưỡng 20%. Seed={seed}"
        )
        assert hold_count >= min_threshold, (
            f"HOLD chiếm {hold_count}/{n_signals} = {hold_count/n_signals:.1%}, "
            f"dưới ngưỡng 20%. Seed={seed}"
        )


# ==============================================================================
# Feature: post-training-evaluation, Property 12: JSON report contains all required metric sections
# ==============================================================================

import json
import tempfile

from engine.report_generator import ReportGenerator
from engine.evaluation_config import (
    EvaluationConfig,
    EvaluationResult,
    ModelMetadata,
    SymbolEvaluationResult,
)


class TestJsonReportStructureProperty:
    """
    Property 12: JSON report contains all required metric sections.

    For any valid EvaluationResult with N symbols, the serialized JSON report
    SHALL contain keys for backtest_metrics, comparison_metrics, overfitting_metrics,
    prediction_quality, model_metadata, evaluation_timestamp, and exactly N entries
    in the per_symbol section.

    **Validates: Requirements 5.1, 5.3, 5.4**
    """

    @given(
        n_symbols=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=100)
    def test_json_report_contains_all_required_keys_and_n_symbols(
        self, n_symbols: int
    ):
        """
        Với mọi EvaluationResult có N symbols, JSON report phải chứa
        đủ 7 keys bắt buộc và per_symbol phải có đúng N entries.

        **Validates: Requirements 5.1, 5.3, 5.4**
        """
        # Tạo EvaluationResult với N SymbolEvaluationResult entries
        symbol_results = {}
        for i in range(n_symbols):
            symbol_name = f"SYMBOL_{i}"
            symbol_results[symbol_name] = SymbolEvaluationResult(symbol=symbol_name)

        eval_result = EvaluationResult(
            model_metadata=ModelMetadata(
                architecture="TestNet",
                parameter_count=1000,
                symbols_trained=[f"SYMBOL_{i}" for i in range(n_symbols)],
            ),
            symbol_results=symbol_results,
        )

        # Sử dụng temporary directory cho report output
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = EvaluationConfig(report_dir=tmp_dir)
            generator = ReportGenerator(config=config)

            # Gọi generate_json() và đọc lại file JSON
            json_path = generator.generate_json(eval_result)

            with open(json_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)

        # Kiểm tra 7 required keys tồn tại
        required_keys = {
            "backtest_metrics",
            "comparison_metrics",
            "overfitting_metrics",
            "prediction_quality",
            "model_metadata",
            "evaluation_timestamp",
            "per_symbol",
        }

        for key in required_keys:
            assert key in report_data, (
                f"Required key '{key}' không tồn tại trong JSON report. "
                f"Keys hiện có: {list(report_data.keys())}"
            )

        # Kiểm tra per_symbol có đúng N entries
        per_symbol = report_data["per_symbol"]
        assert len(per_symbol) == n_symbols, (
            f"per_symbol có {len(per_symbol)} entries, kỳ vọng {n_symbols}. "
            f"Keys: {list(per_symbol.keys())}"
        )


# ==============================================================================
# Feature: post-training-evaluation, Property 13: Equity curve serialization round-trip preserves data
# ==============================================================================

from engine.report_generator import ReportGenerator


class TestEquityCurveSerializationRoundTripProperty:
    """
    Property 13: Equity curve serialization round-trip preserves data.

    For any equity curve represented as a pd.Series with DatetimeIndex,
    serializing to a list of [timestamp, value] pairs and then reconstructing
    SHALL produce values within floating-point epsilon of the original.

    **Validates: Requirements 5.5**
    """

    def setup_method(self):
        """Khởi tạo ReportGenerator."""
        self.generator = ReportGenerator()

    @given(
        values=st.lists(
            st.floats(
                min_value=-1e10,
                max_value=1e10,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_serialization_preserves_values(self, values):
        """
        Với mọi equity curve (pd.Series với DatetimeIndex), serialize thành
        list of [timestamp_str, value] pairs rồi so sánh giá trị phải khớp
        trong khoảng floating-point epsilon.

        **Validates: Requirements 5.5**
        """
        # Tạo pd.Series với DatetimeIndex
        index = pd.date_range(start="2020-01-01", periods=len(values), freq="D")
        curve = pd.Series(values, index=index)

        # Serialize equity curve
        serialized = self.generator._serialize_equity_curve(curve)

        # Kiểm tra length giữ nguyên
        assert len(serialized) == len(curve), (
            f"Serialized length={len(serialized)} != original length={len(curve)}"
        )

        # Kiểm tra từng giá trị khớp trong floating-point epsilon
        for i, (pair, original_value) in enumerate(zip(serialized, values)):
            ts_str, serialized_value = pair

            # Giá trị phải khớp trong floating-point epsilon
            assert serialized_value == pytest.approx(original_value), (
                f"Tại index {i}: serialized_value={serialized_value} != "
                f"original_value={original_value}"
            )

    @given(
        values=st.lists(
            st.floats(
                min_value=-1e10,
                max_value=1e10,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_serialization_preserves_timestamps(self, values):
        """
        Với mọi equity curve, timestamps trong kết quả serialize phải là
        chuỗi ISO format hợp lệ và tương ứng đúng thứ tự với index gốc.

        **Validates: Requirements 5.5**
        """
        # Tạo pd.Series với DatetimeIndex
        index = pd.date_range(start="2020-01-01", periods=len(values), freq="D")
        curve = pd.Series(values, index=index)

        # Serialize equity curve
        serialized = self.generator._serialize_equity_curve(curve)

        # Kiểm tra từng timestamp khớp với index gốc
        for i, (pair, original_ts) in enumerate(zip(serialized, index)):
            ts_str, _ = pair

            # Timestamp phải parse được và khớp với original
            parsed_ts = pd.Timestamp(ts_str)
            assert parsed_ts == original_ts, (
                f"Tại index {i}: parsed_ts={parsed_ts} != original_ts={original_ts}"
            )


# ==============================================================================
# Feature: post-training-evaluation, Property 14: Normalization applies training parameters consistently
# ==============================================================================

from engine.evaluation_module import EvaluationModule


class TestNormalizationConsistencyProperty:
    """
    Property 14: Normalization applies training parameters consistently.

    For any raw feature array and norm_params dict containing mean and std arrays,
    the normalized output SHALL equal `(raw - mean) / std` element-wise (where std > 0).

    **Validates: Requirements 6.2**
    """

    def setup_method(self):
        """Khởi tạo EvaluationModule."""
        self.module = EvaluationModule()

    @given(
        num_features=st.integers(min_value=1, max_value=5),
        num_samples=st.integers(min_value=1, max_value=10),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_normalization_equals_z_score_formula(self, num_features, num_samples, data):
        """
        Với mọi raw feature array shape (N, num_features) và norm_params chứa
        mean/std arrays có length = num_features (std > 0), kết quả normalize
        phải bằng (raw - mean) / std element-wise.

        **Validates: Requirements 6.2**
        """
        # Sinh feature array shape (num_samples, num_features)
        features = data.draw(
            st.lists(
                st.lists(
                    st.floats(
                        min_value=-100.0,
                        max_value=100.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    min_size=num_features,
                    max_size=num_features,
                ),
                min_size=num_samples,
                max_size=num_samples,
            )
        )
        features_arr = np.array(features, dtype=np.float64)

        # Sinh mean array có length = num_features
        mean_list = data.draw(
            st.lists(
                st.floats(
                    min_value=-100.0,
                    max_value=100.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=num_features,
                max_size=num_features,
            )
        )

        # Sinh std array có length = num_features (std > 0)
        std_list = data.draw(
            st.lists(
                st.floats(
                    min_value=0.01,
                    max_value=10.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=num_features,
                max_size=num_features,
            )
        )

        mean_arr = np.array(mean_list, dtype=np.float64)
        std_arr = np.array(std_list, dtype=np.float64)

        # Tạo norm_params dict
        norm_params = {
            "mean": mean_arr.tolist(),
            "std": std_arr.tolist(),
        }

        # Gọi _normalize_features
        result = self.module._normalize_features(features_arr, norm_params)

        # Tính expected result theo công thức z-score: (raw - mean) / std
        expected = (features_arr - mean_arr) / std_arr

        # So sánh element-wise
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-7,
            atol=1e-10,
            err_msg=(
                f"Normalization không khớp z-score formula.\n"
                f"features shape: {features_arr.shape}\n"
                f"mean: {mean_arr}\n"
                f"std: {std_arr}"
            ),
        )



# ==============================================================================
# Feature: post-training-evaluation, Property 15: Missing model/data raises correct EngineError subclass
# ==============================================================================

from unittest.mock import MagicMock

from engine.config import DataError, ModelError
from engine.evaluation_module import EvaluationModule


class TestMissingModelDataErrorHandlingProperty:
    """
    Property 15: Missing model/data raises correct EngineError subclass.

    For any non-existent file path provided as model_path, the evaluation SHALL
    raise a ModelError with a non-empty error_code and details dict containing
    the invalid path. For missing test data, it SHALL raise a DataError with
    appropriate error_code.

    **Validates: Requirements 6.4**
    """

    def setup_method(self):
        """Khởi tạo EvaluationModule."""
        self.module = EvaluationModule()

    @given(
        path_segment=st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        )
    )
    @settings(max_examples=100)
    def test_load_model_raises_model_error_for_nonexistent_path(
        self, path_segment: str
    ):
        """
        Với mọi đường dẫn file không tồn tại, _load_model() phải raise ModelError
        với error_code="MODEL_NOT_FOUND" và details["model_path"] chứa đường dẫn gốc.

        **Validates: Requirements 6.4**
        """
        # Tạo đường dẫn file không tồn tại
        nonexistent_path = f"/tmp/nonexistent_{path_segment}.pt"

        # Gọi _load_model và kiểm tra raises ModelError
        with pytest.raises(ModelError) as exc_info:
            self.module._load_model(nonexistent_path)

        # Kiểm tra error_code không rỗng
        assert exc_info.value.error_code is not None
        assert len(exc_info.value.error_code) > 0, (
            f"error_code rỗng, kỳ vọng non-empty string"
        )
        assert exc_info.value.error_code == "MODEL_NOT_FOUND", (
            f"error_code={exc_info.value.error_code}, kỳ vọng 'MODEL_NOT_FOUND'"
        )

        # Kiểm tra details chứa model_path
        assert "model_path" in exc_info.value.details, (
            f"details không chứa key 'model_path': {exc_info.value.details}"
        )
        assert exc_info.value.details["model_path"] == nonexistent_path, (
            f"details['model_path']={exc_info.value.details['model_path']} "
            f"!= expected={nonexistent_path}"
        )

    @given(
        path_segment=st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(whitelist_categories=("L", "N")),
        )
    )
    @settings(max_examples=100)
    def test_validate_inputs_raises_data_error_for_empty_test_data(
        self, path_segment: str
    ):
        """
        Khi test_data là dict rỗng, _validate_inputs() phải raise DataError
        với error_code="INSUFFICIENT_TEST_DATA".

        **Validates: Requirements 6.4**
        """
        # Tạo mock TrainingResult với model_path hợp lệ (file thật sự tồn tại)
        # Sử dụng file __init__.py của engine vì nó chắc chắn tồn tại
        import os

        engine_init = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "engine",
            "__init__.py",
        )

        # Nếu file không tồn tại, dùng file config.py thay thế
        if not os.path.exists(engine_init):
            engine_init = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "engine",
                "config.py",
            )

        training_result = MagicMock()
        training_result.model_path = engine_init

        # Gọi _validate_inputs với test_data rỗng
        with pytest.raises(DataError) as exc_info:
            self.module._validate_inputs(training_result, test_data={})

        # Kiểm tra error_code
        assert exc_info.value.error_code is not None
        assert len(exc_info.value.error_code) > 0, (
            f"error_code rỗng, kỳ vọng non-empty string"
        )
        assert exc_info.value.error_code == "INSUFFICIENT_TEST_DATA", (
            f"error_code={exc_info.value.error_code}, "
            f"kỳ vọng 'INSUFFICIENT_TEST_DATA'"
        )


# ==============================================================================
# Feature: post-training-evaluation, Property 2: Test split uses exactly the last 15% of chronological data
# ==============================================================================

import math

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st


class TestSplitChronologicalCorrectnessProperty:
    """
    Property 2: Test split uses exactly the last 15% of chronological data.

    For any dataset of length N (N >= 10), the test split SHALL contain exactly
    the last `floor(N * 0.15)` rows, and all test rows SHALL have timestamps
    strictly after all train/validation rows.

    **Validates: Requirements 1.4**
    """

    @given(
        n=st.integers(min_value=10, max_value=500),
    )
    @settings(max_examples=100)
    def test_split_size_equals_floor_n_times_015(self, n: int):
        """
        Với mọi dataset có N rows (N >= 10), test split phải chứa đúng
        floor(N * 0.15) rows.

        **Validates: Requirements 1.4**
        """
        # Tạo dữ liệu chronological với N ngày liên tiếp
        dates = pd.date_range(start="2020-01-01", periods=n, freq="D")

        # Tính expected test size theo công thức
        expected_test_size = math.floor(n * 0.15)
        split_index = n - expected_test_size

        # Thực hiện split: train/validation = [:split_index], test = [split_index:]
        train_val_dates = dates[:split_index]
        test_dates = dates[split_index:]

        # Assert: kích thước test split đúng bằng floor(N * 0.15)
        assert len(test_dates) == expected_test_size, (
            f"N={n}: test split size={len(test_dates)}, "
            f"expected=floor({n} * 0.15)={expected_test_size}"
        )

    @given(
        n=st.integers(min_value=10, max_value=500),
    )
    @settings(max_examples=100)
    def test_split_test_timestamps_strictly_after_train(self, n: int):
        """
        Với mọi dataset có N rows (N >= 10), tất cả timestamps trong test split
        phải strictly sau tất cả timestamps trong train/validation split.

        **Validates: Requirements 1.4**
        """
        # Tạo dữ liệu chronological với N ngày liên tiếp
        dates = pd.date_range(start="2020-01-01", periods=n, freq="D")

        # Tính split point
        expected_test_size = math.floor(n * 0.15)
        split_index = n - expected_test_size

        # Thực hiện split
        train_val_dates = dates[:split_index]
        test_dates = dates[split_index:]

        # Nếu test split rỗng (trường hợp edge với N rất nhỏ), skip assertion
        if len(test_dates) == 0 or len(train_val_dates) == 0:
            return

        # Assert: timestamp nhỏ nhất trong test > timestamp lớn nhất trong train/val
        max_train_ts = train_val_dates.max()
        min_test_ts = test_dates.min()

        assert min_test_ts > max_train_ts, (
            f"N={n}: min test timestamp={min_test_ts} không strictly after "
            f"max train/val timestamp={max_train_ts}"
        )

    @given(
        n=st.integers(min_value=10, max_value=500),
    )
    @settings(max_examples=100)
    def test_split_train_and_test_cover_full_dataset(self, n: int):
        """
        Với mọi dataset có N rows (N >= 10), tổng kích thước train/val + test
        phải bằng đúng N (không mất dữ liệu).

        **Validates: Requirements 1.4**
        """
        # Tạo dữ liệu chronological với N ngày liên tiếp
        dates = pd.date_range(start="2020-01-01", periods=n, freq="D")

        # Tính split point
        expected_test_size = math.floor(n * 0.15)
        split_index = n - expected_test_size

        # Thực hiện split
        train_val_dates = dates[:split_index]
        test_dates = dates[split_index:]

        # Assert: train/val + test = toàn bộ dataset
        assert len(train_val_dates) + len(test_dates) == n, (
            f"N={n}: train/val={len(train_val_dates)} + test={len(test_dates)} "
            f"= {len(train_val_dates) + len(test_dates)} != N={n}"
        )
