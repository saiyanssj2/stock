# Implementation Plan: Stock Decision Engine

## Overview

This plan implements a Stock Decision Engine that combines Minimax/Alpha-Beta search with a TCN+Attention neural network evaluation model for Vietnamese stock market trading decisions. The engine integrates into the existing Streamlit application as a new tab, runs entirely offline on local CSV data, and supports concurrent training and inference on RTX 2060.

## Tasks

- [x] 1. Set up project structure, configuration, and core types
  - [x] 1.1 Create engine package directory structure and config module
    - Create `engine/` package with `__init__.py`
    - Create `engine/strategies/` subpackage with `__init__.py`
    - Create `engine/models/` directory (for saved model weights)
    - Create `engine/config.py` with all configuration constants (ModelConfig, TrainingConfig, EngineConfig, SearchConfig)
    - Define Action enum (BUY, HOLD, SELL), DecisionReport, ScenarioResult, IndicatorContribution, Trade, BacktestResult, ComparisonResult dataclasses
    - Define error hierarchy: EngineError, DataError, ModelError, ResourceError, ConfigError
    - _Requirements: 1.1, 1.4, 1.5, 2.2, 3.2, 4.3, 4.7, 5.1, 9.1_

- [x] 2. Implement MarketState and FeatureVectorBuilder
  - [x] 2.1 Implement MarketState dataclass and construction
    - Create `engine/market_state.py`
    - Implement MarketState dataclass with fields: symbol, timestamp, ohlcv (lookback, 5), indicators (lookback, num_indicators), lookback
    - Implement `from_dataframe()` classmethod that loads CSV data via pandas and computes indicators using `analysis.add_indicators()`
    - Implement validation: reject if data rows < lookback, reject if lookback not in [20, 200]
    - Implement `serialize()` / `deserialize()` methods for caching
    - _Requirements: 2.1, 2.2, 2.3, 2.5, 2.6_

  - [x] 2.2 Implement FeatureVectorBuilder with normalization
    - Create FeatureVectorBuilder class in `engine/market_state.py`
    - Define INDICATOR_COLUMNS list matching exact ordering from `analysis.add_indicators()`
    - Implement `build()`: forward-fill NaN along time axis, zero-fill remaining, min-max normalize to [0, 1]
    - Implement `denormalize()` for inverse normalization
    - Implement `save_params()` / load normalization params from JSON
    - Implement `from_training_data()` classmethod to compute min/max from training DataFrames
    - _Requirements: 2.4, 2.7, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [x] 2.3 Write property tests for MarketState (Properties 2, 9)
    - **Property 2: MarketState serialization round-trip** - serialize then deserialize produces values within 1e-9 tolerance
    - **Property 9: Lookback window configuration and enforcement** - valid range [20, 200] succeeds, outside fails
    - **Validates: Requirements 2.2, 2.5, 2.6**

  - [x] 2.4 Write property tests for FeatureVectorBuilder (Properties 1, 3, 4, 6)
    - **Property 1: Feature vector normalization bounds** - all values in [0.0, 1.0] for non-degenerate data
    - **Property 3: Normalize/denormalize round-trip** - within 0.01% relative tolerance
    - **Property 4: Normalization parameters serialization round-trip** - save/load JSON produces identical params within 1e-15
    - **Property 6: NaN handling produces clean feature vectors** - no NaN or Inf in output
    - **Validates: Requirements 2.4, 2.7, 8.2, 8.3, 8.4, 8.6**

- [x] 3. Implement EvaluationModel (TCN + Attention)
  - [x] 3.1 Implement TCNBlock and StockEvalNet architecture
    - Create `engine/evaluation_model.py`
    - Implement TCNBlock with dilated causal convolutions, BatchNorm, GELU, residual connections
    - Implement StockEvalNet: 3 TCN blocks (63→128→128→64), MultiheadAttention (4 heads, d=64), LayerNorm, AdaptiveAvgPool1d, Linear head with Tanh output
    - Verify parameter count ~180K
    - _Requirements: 3.1, 3.2, 3.4, 3.5_

  - [x] 3.2 Implement ModelManager with hot-swap and validation
    - Implement ModelManager class: load with checksum validation, hot-swap (atomic model replacement), version tracking
    - Implement GPU/CPU device management and fallback logic
    - Implement `predict()` high-level API that handles batching, device placement, and memory cleanup
    - Implement batch size retry on OOM (halve 3x, then CPU fallback)
    - _Requirements: 3.3, 3.6, 7.3, 12.1, 12.3, 12.5, 12.6, 13.2, 13.3, 13.4, 13.5, 13.6_

  - [x] 3.3 Write property tests for EvaluationModel (Properties 5, 7)
    - **Property 5: Model save/load inference consistency** - outputs within ±1e-6 after save/load on same device
    - **Property 7: Evaluation model output bounded for any batch size** - shape (batch, 1) with values in [-1.0, +1.0] for batch 1-50
    - **Validates: Requirements 3.1, 3.5, 13.2**

- [~] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement ScenarioGenerator
  - [x] 5.1 Implement scenario generation with percentile-based sampling
    - Create `engine/scenario_generator.py`
    - Implement `_estimate_distribution()`: compute daily returns from last N sessions (N ≥ 30), detect volatility regime (ATR-based)
    - Implement `generate()`: produce 3-7 scenarios at strategic percentiles (P10, P30, P50, P70, P90 for 5 scenarios)
    - Implement `_apply_action_effect()`: project price movement, cap at ±7% daily limit, recompute indicators
    - Validate minimum 30 days history, reject otherwise
    - _Requirements: 4.6, 4.7, 4.8, 4.9_

  - [x] 5.2 Write property test for ScenarioGenerator (Property 14)
    - **Property 14: Scenario generation validity** - produces 3-7 valid scenarios with ≤±7% price change for states with ≥30 days history; rejects states with <30 days
    - **Validates: Requirements 4.6, 4.7, 4.8**

- [x] 6. Implement SearchModule (Minimax + Alpha-Beta)
  - [x] 6.1 Implement Minimax search with Alpha-Beta pruning
    - Create `engine/search_module.py`
    - Implement SearchNode dataclass
    - Implement `_minimax()` with alpha-beta pruning, depth control (default 3, max 5)
    - Implement `search()`: iterate BUY/HOLD/SELL actions, generate scenarios, evaluate leaf nodes, apply timeout (5s)
    - Implement adaptive branching factor reduction when depth > 3 to maintain time constraint
    - On timeout, return best action from deepest fully evaluated level
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.9_

  - [x] 6.2 Implement DecisionReport generation with confidence scoring
    - Compute confidence from scenario agreement level
    - Apply low-confidence HOLD override (confidence < 0.3 → HOLD)
    - Generate top 3 scenarios with action sequences and leaf scores
    - Compute top 5 indicator contributions (sorted by absolute contribution)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [x] 6.3 Write property tests for SearchModule (Properties 13, 15, 16)
    - **Property 13: Minimax optimality** - recommended action has score ≥ all other root-level actions
    - **Property 15: Decision report structural completeness** - all required fields present and valid
    - **Property 16: Low confidence HOLD override** - confidence < 0.3 always produces HOLD
    - **Validates: Requirements 4.2, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**

- [x] 7. Implement TrainingPipeline
  - [x] 7.1 Implement label generation and data splitting
    - Create `engine/training_pipeline.py`
    - Implement `_generate_labels()`: compute future returns over configurable horizon (default 5), apply tanh scaling with sensitivity=10
    - Implement `_chronological_split()`: 70/15/15 chronological train/val/test split
    - Validate minimum 250 sessions per symbol, skip with warning if insufficient
    - Always include VNINDEX data for market context
    - _Requirements: 6.1, 6.2, 6.3, 6.12, 6.13_

  - [x] 7.2 Implement full training and incremental training
    - Implement `train_full()`: train from scratch on VN30 + VNINDEX, target <4 hours
    - Implement `train_incremental()`: fine-tune with new data, max 10 epochs, target <10 minutes
    - Implement epoch logging (loss, MAE, validation loss) to local file
    - Implement CheckpointManager: save after each epoch (model, optimizer, epoch), resume from last checkpoint
    - Save complete model state (architecture, weights, optimizer, epoch, norm params) in PyTorch format
    - _Requirements: 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10, 6.11, 13.1_

  - [x] 7.3 Write property tests for TrainingPipeline (Properties 10, 11, 12)
    - **Property 10: Training label generation bounded** - labels in [-1.0, +1.0], monotonically increasing with return
    - **Property 11: Chronological split ordering** - all train timestamps < all val timestamps < all test timestamps
    - **Property 12: Checkpoint save/resume consistency** - resume produces same parameter updates within 1e-6
    - **Validates: Requirements 6.2, 6.3, 6.8**

- [~] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement BacktestEngine
  - [x] 9.1 Implement backtesting framework with Vietnamese market rules
    - Create `engine/backtest_engine.py`
    - Implement `run()`: simulate trades with T+2.5 settlement, ±7% daily limits, 100-share lot size, 20% max position
    - Track equity curve, compute total return %, annualized return, win rate, max drawdown, Sharpe ratio (risk-free=0%)
    - Record all trades with entry/exit prices, shares, PnL
    - Handle invalid date ranges with error messages
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

  - [x] 9.2 Implement strategy comparison framework
    - Implement `compare_strategies()`: run identical backtest across all strategies
    - Enforce same date range, initial capital, universe, position sizing for fairness
    - Exclude strategies with insufficient signals (no BUY or no SELL) with notification
    - _Requirements: 10.5, 10.6, 10.7, 10.9_

  - [x] 9.3 Write property tests for BacktestEngine (Properties 17, 19)
    - **Property 17: Backtest trade constraints** - lot size multiple of 100, ±7% price limit, T+2.5 settlement, ≤20% position
    - **Property 19: Strategy comparison fairness** - all strategies evaluated on identical date range, capital, universe, and sizing rules
    - **Validates: Requirements 9.2, 9.3, 9.4, 10.5**

- [x] 10. Implement Philosophy-Based Strategies
  - [x] 10.1 Implement BaseStrategy ABC and Wyckoff strategy
    - Create `engine/strategies/base.py` with BaseStrategy abstract class (generate_signal method)
    - Create `engine/strategies/wyckoff.py`: BUY if composite score ≥ +3, SELL if ≤ -3, HOLD otherwise
    - Reuse existing Wyckoff accumulation/distribution detection logic from scanner
    - _Requirements: 10.1_

  - [x] 10.2 Implement Technical, Momentum, and Mean Reversion strategies
    - Create `engine/strategies/technical.py`: composite score from RSI, MACD, EMA crossover, Bollinger Bands; BUY ≥ +5, SELL ≤ -4
    - Create `engine/strategies/momentum.py`: BUY when ADX > 25 AND ROC_14 > 0 AND volume > 1.5x avg; SELL when ADX > 25 AND ROC < 0
    - Create `engine/strategies/mean_reversion.py`: BUY when price < BB_lower OR RSI < 30 OR price < SMA_20 - 2σ; SELL when opposite
    - _Requirements: 10.2, 10.3, 10.4_

  - [x] 10.3 Write property test for philosophy strategies (Property 18)
    - **Property 18: Philosophy strategy signal validity** - each strategy produces exactly one Action from {BUY, HOLD, SELL} consistent with documented threshold rules
    - **Validates: Requirements 10.1, 10.2, 10.3, 10.4**

- [x] 11. Implement DecisionEngine orchestrator
  - [x] 11.1 Implement DecisionEngine main class
    - Create `engine/decision_engine.py`
    - Implement `__init__()`: load config, initialize ModelManager, SearchModule, ScenarioGenerator, FeatureVectorBuilder
    - Implement `analyze()`: load CSV → build MarketState → run search → generate DecisionReport
    - Implement `backtest()` and `compare()` delegation to BacktestEngine
    - Implement data validation (Property 8): check CSV existence, required columns, sufficient rows
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 11.2, 11.3_

  - [x] 11.2 Write property test for data validation (Property 8)
    - **Property 8: Data validation error reporting** - missing/malformed files correctly identified in error response
    - **Validates: Requirements 1.2, 1.4, 1.5**

- [~] 12. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Implement Streamlit UI integration
  - [x] 13.1 Create Decision Engine UI tab
    - Create `ui_decision_engine.py` with `show_decision_engine(base_dir)` function
    - Implement symbol selection dropdown (VN30 stocks)
    - Display DecisionReport: signal badge (BUY/HOLD/SELL), confidence score, position score
    - Display top 5 indicator contributions as bar chart
    - Display top 3 scenarios as ranked list with scores
    - Handle errors with Vietnamese st.error() messages
    - _Requirements: 11.1, 11.4, 11.5, 11.6_

  - [-] 13.2 Create Backtest sub-tab in Decision Engine UI
    - Add backtest sub-tab within the Decision Engine tab
    - Implement date range selector (default 1 year) and initial capital input
    - Display strategy comparison table (total return, annualized return, win rate, max drawdown, Sharpe)
    - Display overlaid equity curves chart with distinct colors per strategy
    - Display trade list with entry/exit prices, shares, PnL
    - _Requirements: 11.7, 9.6, 10.6, 10.7_

  - [-] 13.3 Add training status display and model info to UI
    - Display model version and last training timestamp
    - Display training progress when active (epoch, loss, ETA) updated every 10s
    - Display notification when training paused due to GPU constraints
    - _Requirements: 7.4, 7.5, 7.6_

  - [~] 13.4 Integrate Decision Engine tab into app.py
    - Import `show_decision_engine` in `app.py`
    - Add "🧠 Decision Engine" to the existing `st.tabs()` list
    - Wire the tab to call `show_decision_engine(BASE_DIR)`
    - _Requirements: 11.1_

- [ ] 14. Implement concurrent training and inference
  - [~] 14.1 Implement background training with hot-swap
    - Add threading/async support for background training in DecisionEngine
    - Implement GPU memory monitoring: pause training if >90% utilization, resume at <80%
    - Implement model hot-swap within 3 seconds without dropping in-flight requests
    - Implement symbol queue for incremental training additions from UI
    - Ensure inference completes within 5 seconds regardless of training activity
    - _Requirements: 7.1, 7.2, 7.3, 7.5, 7.6, 6.11, 12.4_

- [ ] 15. Create test infrastructure and shared fixtures
  - [x] 15.1 Set up test directories and conftest with Hypothesis strategies
    - Create `tests/` directory with `property/`, `unit/`, `integration/` subdirectories
    - Create `tests/conftest.py` with shared Hypothesis strategies: `market_state_strategy()`, `feature_vector_strategy()`, `ohlcv_dataframe_strategy()`, `normalization_params_strategy()`, `trade_sequence_strategy()`
    - Configure Hypothesis settings (max_examples=100, suppress_health_check=[too_slow])
    - _Requirements: All property tests depend on this infrastructure_

  - [~] 15.2 Write unit tests for core components
    - Test model architecture smoke tests (parameter count, device placement)
    - Test specific signal generation scenarios for each philosophy strategy
    - Test error handling for specific invalid inputs
    - Test configuration validation edge cases
    - _Requirements: 3.2, 10.1, 10.2, 10.3, 10.4, 1.4, 1.5_

  - [~] 15.3 Write integration tests
    - Test end-to-end analysis pipeline (CSV → DecisionReport)
    - Test concurrent training + inference
    - Test model hot-swap under load
    - Test performance benchmarks (50ms inference, 5s search)
    - _Requirements: 7.1, 7.3, 11.4, 3.3, 4.4_

- [~] 16. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate the 19 universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The engine uses Python with PyTorch, matching the design specification
- All code runs on target hardware: i5-10400F + RTX 2060 (6GB VRAM)
- Vietnamese market rules (T+2.5 settlement, ±7% limits, 100-share lots) are enforced throughout

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "15.1"] },
    { "id": 1, "tasks": ["2.1", "2.2"] },
    { "id": 2, "tasks": ["2.3", "2.4", "3.1"] },
    { "id": 3, "tasks": ["3.2", "3.3"] },
    { "id": 4, "tasks": ["5.1", "7.1"] },
    { "id": 5, "tasks": ["5.2", "6.1", "7.2"] },
    { "id": 6, "tasks": ["6.2", "6.3", "7.3"] },
    { "id": 7, "tasks": ["9.1", "10.1"] },
    { "id": 8, "tasks": ["9.2", "9.3", "10.2"] },
    { "id": 9, "tasks": ["10.3", "11.1"] },
    { "id": 10, "tasks": ["11.2", "13.1"] },
    { "id": 11, "tasks": ["13.2", "13.3"] },
    { "id": 12, "tasks": ["13.4", "14.1"] },
    { "id": 13, "tasks": ["15.2", "15.3"] }
  ]
}
```
