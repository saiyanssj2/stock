"""
DecisionEngine: Main orchestrator for the Stock Decision Engine.

Ties together all components:
- ModelManager: loads and manages the evaluation neural network
- SearchModule: runs Minimax/Alpha-Beta search over action-scenario trees
- ScenarioGenerator: produces plausible future market states
- FeatureVectorBuilder: normalizes market data into model-ready tensors
- DecisionReportGenerator: builds structured DecisionReport from search results
- BacktestEngine: simulates historical trading
- TrainingPipeline: trains the evaluation model

The analyze() pipeline:
1. Validate CSV file (existence, required columns, sufficient rows)
2. Load data into DataFrame
3. Build MarketState using MarketState.from_dataframe()
4. Run SearchModule.search() to explore action tree
5. Generate DecisionReport using DecisionReportGenerator
6. Return the DecisionReport

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 11.2, 11.3
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from engine.backtest_engine import BacktestEngine
from engine.config import (
    Action,
    BacktestResult,
    ComparisonResult,
    ConfigError,
    DataError,
    DecisionReport,
    EngineConfig,
    ModelError,
    SearchConfig,
)
from engine.evaluation_model import ModelManager
from engine.market_state import FeatureVectorBuilder, MarketState
from engine.scenario_generator import ScenarioGenerator
from engine.search_module import DecisionReportGenerator, SearchModule
from engine.strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    TechnicalStrategy,
    WyckoffStrategy,
)

logger = logging.getLogger(__name__)


class DecisionEngine:
    """Main entry point coordinating all decision engine components.

    Provides high-level APIs for:
    - analyze(symbol): Full analysis pipeline returning a DecisionReport
    - backtest(symbol, start, end): Run backtest for a symbol
    - compare(symbol, start, end): Compare AI vs philosophy strategies

    Parameters
    ----------
    base_dir : str
        Base directory where CSV data files are stored.
    config : EngineConfig, optional
        Engine configuration. Uses defaults if not provided.
    search_config : SearchConfig, optional
        Search configuration. Uses defaults if not provided.
    """

    def __init__(
        self,
        base_dir: str,
        config: Optional[EngineConfig] = None,
        search_config: Optional[SearchConfig] = None,
    ):
        self._config = config or EngineConfig()
        self._config.base_dir = base_dir
        self._search_config = search_config or SearchConfig()
        self._base_dir = Path(base_dir)

        # Initialize ModelManager
        self._model_manager = ModelManager()

        # Initialize ScenarioGenerator
        self._scenario_generator = ScenarioGenerator(
            min_history=self._search_config.min_history_days,
            config=self._config,
            search_config=self._search_config,
        )

        # Initialize SearchModule
        self._search_module = SearchModule(
            model_manager=self._model_manager,
            scenario_generator=self._scenario_generator,
            config=self._search_config,
        )

        # Initialize DecisionReportGenerator
        self._report_generator = DecisionReportGenerator(
            config=self._search_config,
            engine_config=self._config,
        )

        # Initialize FeatureVectorBuilder (loaded lazily when norm params available)
        self._feature_builder: Optional[FeatureVectorBuilder] = None

        # Initialize BacktestEngine
        self._backtest_engine = BacktestEngine(
            initial_capital=self._config.default_capital,
            max_position_pct=self._config.max_position_pct,
            config=self._config,
        )

        # Attempt to load model and normalization params
        self._try_load_model()
        self._try_load_norm_params()

    # --------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------

    @property
    def config(self) -> EngineConfig:
        """Current engine configuration."""
        return self._config

    @property
    def model_manager(self) -> ModelManager:
        """Access to the ModelManager for status queries."""
        return self._model_manager

    @property
    def is_model_loaded(self) -> bool:
        """Whether the evaluation model is loaded and ready."""
        return self._model_manager.is_loaded

    # --------------------------------------------------------------------------
    # Initialization Helpers
    # --------------------------------------------------------------------------

    def _try_load_model(self) -> None:
        """Attempt to load the evaluation model from the configured path.

        Logs a warning if the model file is not found but does not raise
        an error - the model can be loaded later or training can be run.
        """
        model_path = self._base_dir / self._config.model_path
        if model_path.exists():
            try:
                self._model_manager.load(str(model_path))
                logger.info(f"Evaluation model loaded from {model_path}")
            except ModelError as e:
                logger.warning(f"Failed to load evaluation model: {e}")
        else:
            logger.info(
                f"Model file not found at {model_path}. "
                "Run training pipeline to create a model."
            )

    def _try_load_norm_params(self) -> None:
        """Attempt to load normalization parameters from the configured path.

        Logs a warning if the file is not found.
        """
        norm_path = self._base_dir / self._config.norm_params_path
        if norm_path.exists():
            try:
                self._feature_builder = FeatureVectorBuilder(norm_params_path=str(norm_path))
                logger.info(f"Normalization parameters loaded from {norm_path}")
            except DataError as e:
                logger.warning(f"Failed to load normalization parameters: {e}")
                self._feature_builder = None
        else:
            logger.info(
                f"Normalization params not found at {norm_path}. "
                "Using default normalization."
            )
            # Use default builder (identity normalization) as fallback
            self._feature_builder = FeatureVectorBuilder()

    # --------------------------------------------------------------------------
    # Data Validation (Property 8)
    # --------------------------------------------------------------------------

    def validate_csv(self, file_path: Path) -> None:
        """Validate a CSV file for the decision engine.

        Checks:
        1. File exists
        2. File contains required columns
        3. File has sufficient rows

        Parameters
        ----------
        file_path : Path
            Path to the CSV file to validate.

        Raises
        ------
        DataError
            If the file is missing, has missing columns, or insufficient rows.
        """
        # Check existence
        if not file_path.exists():
            raise DataError(
                f"CSV data file not found: {file_path}",
                error_code="FILE_NOT_FOUND",
                details={
                    "path": str(file_path),
                    "expected_location": str(file_path.parent),
                    "suggestion": "Run update_data.py to download data.",
                },
            )

        # Try to read the file and check columns/rows
        try:
            df = pd.read_csv(file_path, nrows=0)  # Read just headers first
        except Exception as e:
            raise DataError(
                f"Cannot read CSV file {file_path.name}: {e}",
                error_code="FILE_UNREADABLE",
                details={
                    "path": str(file_path),
                    "error": str(e),
                },
            )

        # Check required columns
        required_columns = self._config.csv_required_columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataError(
                f"CSV file '{file_path.name}' is missing required columns: {missing_columns}",
                error_code="MISSING_COLUMNS",
                details={
                    "path": str(file_path),
                    "missing_columns": missing_columns,
                    "available_columns": list(df.columns),
                    "required_columns": required_columns,
                },
            )

        # Check sufficient rows
        try:
            df_full = pd.read_csv(file_path)
        except Exception as e:
            raise DataError(
                f"Failed to fully read CSV file {file_path.name}: {e}",
                error_code="FILE_UNREADABLE",
                details={"path": str(file_path), "error": str(e)},
            )

        if len(df_full) < self._config.min_data_rows:
            raise DataError(
                f"CSV file '{file_path.name}' has insufficient data: "
                f"{len(df_full)} rows, minimum {self._config.min_data_rows} required",
                error_code="INSUFFICIENT_DATA",
                details={
                    "path": str(file_path),
                    "rows": len(df_full),
                    "min_required": self._config.min_data_rows,
                },
            )

    def validate_multiple_csvs(self, file_paths: List[Path]) -> Dict[str, DataError]:
        """Validate multiple CSV files and collect all errors.

        Parameters
        ----------
        file_paths : List[Path]
            List of CSV file paths to validate.

        Returns
        -------
        Dict[str, DataError]
            Mapping of file path string to the DataError for that file.
            Empty dict if all files are valid.
        """
        errors: Dict[str, DataError] = {}
        for path in file_paths:
            try:
                self.validate_csv(path)
            except DataError as e:
                errors[str(path)] = e
        return errors

    # --------------------------------------------------------------------------
    # Core Analysis Pipeline
    # --------------------------------------------------------------------------

    def analyze(self, symbol: str) -> DecisionReport:
        """Run the full analysis pipeline for a stock symbol.

        Pipeline steps:
        1. Validate CSV file (existence, columns, rows)
        2. Load data into DataFrame
        3. Build MarketState from DataFrame
        4. Run SearchModule to explore action tree
        5. Generate DecisionReport from search results

        Parameters
        ----------
        symbol : str
            Stock ticker symbol (e.g., "VNM", "FPT", "VNINDEX").

        Returns
        -------
        DecisionReport
            Complete analysis report with recommendation, confidence,
            scenarios, and indicator contributions.

        Raises
        ------
        DataError
            If CSV data is missing, malformed, or insufficient.
        ModelError
            If the evaluation model is not loaded.
        """
        # Step 1: Validate CSV file
        csv_path = self._resolve_csv_path(symbol)
        self.validate_csv(csv_path)

        # Step 2: Load data
        df = self._load_csv(csv_path)

        # Step 3: Build MarketState
        lookback = self._config.lookback
        state = MarketState.from_dataframe(df, symbol, lookback=lookback, config=self._config)

        # Step 4: Ensure model is loaded
        if not self._model_manager.is_loaded:
            # Try to create a new model for inference (untrained - returns neutral scores)
            logger.warning(
                "No trained model loaded. Creating untrained model for analysis. "
                "Results will be neutral until a model is trained."
            )
            self._model_manager.create_new_model()

        # Step 5: Run search
        search_result = self._search_module.search(state)

        # Step 6: Generate report
        report = self._report_generator.generate(search_result, state)

        return report

    # --------------------------------------------------------------------------
    # Backtest Delegation
    # --------------------------------------------------------------------------

    def backtest(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_capital: Optional[float] = None,
    ) -> BacktestResult:
        """Run a backtest for a symbol using the AI decision engine strategy.

        Delegates to BacktestEngine with an AI-based strategy that uses
        the SearchModule for signal generation.

        Parameters
        ----------
        symbol : str
            Stock ticker symbol.
        start_date : str
            Start date (format: 'YYYY-MM-DD').
        end_date : str
            End date (format: 'YYYY-MM-DD').
        initial_capital : float, optional
            Starting capital in VND. Uses config default if not provided.

        Returns
        -------
        BacktestResult
            Backtest results with metrics and trade list.

        Raises
        ------
        DataError
            If CSV data is unavailable or malformed.
        ConfigError
            If date range is invalid.
        """
        csv_path = self._resolve_csv_path(symbol)
        self.validate_csv(csv_path)
        df = self._load_csv(csv_path)

        # Create AI strategy wrapper
        ai_strategy = _AIStrategy(self, symbol, df)

        # Configure backtest engine with capital
        if initial_capital is not None:
            engine = BacktestEngine(
                initial_capital=initial_capital,
                max_position_pct=self._config.max_position_pct,
                config=self._config,
            )
        else:
            engine = self._backtest_engine

        return engine.run(ai_strategy, df, start_date, end_date)

    def compare(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_capital: Optional[float] = None,
    ) -> ComparisonResult:
        """Compare AI strategy against philosophy-based strategies.

        Runs identical backtests across all strategies:
        - AI Decision Engine
        - Wyckoff
        - Technical Indicators
        - Momentum
        - Mean Reversion

        Parameters
        ----------
        symbol : str
            Stock ticker symbol.
        start_date : str
            Start date (format: 'YYYY-MM-DD').
        end_date : str
            End date (format: 'YYYY-MM-DD').
        initial_capital : float, optional
            Starting capital in VND. Uses config default if not provided.

        Returns
        -------
        ComparisonResult
            Comparison results with all strategy metrics.

        Raises
        ------
        DataError
            If CSV data is unavailable or malformed.
        ConfigError
            If date range is invalid.
        """
        csv_path = self._resolve_csv_path(symbol)
        self.validate_csv(csv_path)
        df = self._load_csv(csv_path)

        # Build strategy dict
        ai_strategy = _AIStrategy(self, symbol, df)
        strategies: Dict[str, object] = {
            "AI Engine": ai_strategy,
            "Wyckoff": WyckoffStrategy(),
            "Technical": TechnicalStrategy(),
            "Momentum": MomentumStrategy(),
            "Mean Reversion": MeanReversionStrategy(),
        }

        # Configure backtest engine with capital
        if initial_capital is not None:
            engine = BacktestEngine(
                initial_capital=initial_capital,
                max_position_pct=self._config.max_position_pct,
                config=self._config,
            )
        else:
            engine = self._backtest_engine

        return engine.compare_strategies(strategies, df, start_date, end_date)

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------

    def _resolve_csv_path(self, symbol: str) -> Path:
        """Resolve the CSV file path for a given symbol.

        Looks for the file as:
        1. {base_dir}/{symbol}.csv
        2. {base_dir}/data/{symbol}.csv

        Parameters
        ----------
        symbol : str
            Stock ticker symbol.

        Returns
        -------
        Path
            Resolved path to the CSV file.
        """
        # Try direct path first
        direct_path = self._base_dir / f"{symbol}.csv"
        if direct_path.exists():
            return direct_path

        # Try data subdirectory
        data_path = self._base_dir / "data" / f"{symbol}.csv"
        if data_path.exists():
            return data_path

        # Return the direct path (will fail validation with clear error)
        return direct_path

    def _load_csv(self, csv_path: Path) -> pd.DataFrame:
        """Load a CSV file into a DataFrame.

        Parameters
        ----------
        csv_path : Path
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame.

        Raises
        ------
        DataError
            If the file cannot be read.
        """
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            raise DataError(
                f"Failed to read CSV file: {csv_path.name}: {e}",
                error_code="FILE_UNREADABLE",
                details={"path": str(csv_path), "error": str(e)},
            )


# ==============================================================================
# AI Strategy Wrapper (for backtest integration)
# ==============================================================================


class _AIStrategy:
    """Wraps the DecisionEngine as a backtest-compatible strategy.

    Uses a simplified approach for backtesting: evaluates the market state
    at each point using the evaluation model directly, without running the
    full search tree (which would be prohibitively slow over many days).

    For backtesting, the AI strategy uses the model's position score:
    - score > 0.3 → BUY
    - score < -0.3 → SELL
    - otherwise → HOLD
    """

    def __init__(self, engine: DecisionEngine, symbol: str, df: pd.DataFrame):
        self._engine = engine
        self._symbol = symbol
        self._df = df
        self._lookback = engine.config.lookback

    def generate_signal(self, df: pd.DataFrame, index: int) -> Action:
        """Generate a trading signal using the AI model.

        Parameters
        ----------
        df : pd.DataFrame
            Full DataFrame with OHLCV and indicators.
        index : int
            Current position index.

        Returns
        -------
        Action
            BUY, HOLD, or SELL.
        """
        import numpy as np

        # Need enough history for lookback window
        if index < self._lookback:
            return Action.HOLD

        try:
            # Extract lookback window
            window_df = df.iloc[index - self._lookback + 1 : index + 1].copy()

            if len(window_df) < self._lookback:
                return Action.HOLD

            # Build a quick feature vector and evaluate
            state = MarketState.from_dataframe(
                df.iloc[: index + 1],
                self._symbol,
                lookback=self._lookback,
                config=self._engine.config,
            )

            # Build feature array for model
            if state.indicators.shape[1] > 0:
                features = np.concatenate(
                    [state.ohlcv, state.indicators], axis=1
                ).astype(np.float32)
            else:
                features = state.ohlcv.astype(np.float32)

            # Handle NaN
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Get model prediction
            scores = self._engine.model_manager.predict(features)
            score = float(scores.flatten()[0])

            # Simple threshold-based decision
            if score > 0.3:
                return Action.BUY
            elif score < -0.3:
                return Action.SELL
            else:
                return Action.HOLD

        except Exception as e:
            logger.debug(f"AI strategy signal generation failed at index {index}: {e}")
            return Action.HOLD

    @property
    def name(self) -> str:
        """Strategy name."""
        return "AI Engine"
