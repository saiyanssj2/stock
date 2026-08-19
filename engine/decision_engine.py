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
- BackgroundTrainingManager: concurrent training with hot-swap

The analyze() pipeline:
1. Validate CSV file (existence, required columns, sufficient rows)
2. Load data into DataFrame
3. Build MarketState using MarketState.from_dataframe()
4. Run SearchModule.search() to explore action tree
5. Generate DecisionReport using DecisionReportGenerator
6. Return the DecisionReport

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 7.1, 7.2, 7.3, 7.5, 7.6, 6.11, 11.2, 11.3, 12.4
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from engine.backtest_engine import BacktestEngine
from engine.background_training import (
    BackgroundTrainingManager,
    GPUMemoryMonitor,
    SymbolQueue,
    TrainingState,
    TrainingStatus,
)
from engine.training_controller import GracefulTrainingStatus, TrainingController
from engine.training_cycle import TrainingCycleManager
from engine.progressive_trainer import ProgressiveTrainer, TrainingPhase
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
        # Vietnamese stock CSV data stores prices in units of 1000 VND
        # (e.g., 76.5 in CSV = 76,500 VND actual). Set price_scale for backtest.
        if self._config.price_scale == 1.0:
            self._config.price_scale = 1000.0
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

        # Initialize BackgroundTrainingManager for concurrent training/inference
        # (Requirement 7.1, 7.2, 7.3, 7.5, 7.6, 6.11, 12.4)
        self._gpu_monitor = GPUMemoryMonitor()
        self._background_training = BackgroundTrainingManager(
            model_manager=self._model_manager,
            gpu_monitor=self._gpu_monitor,
        )

        # Initialize TrainingController for graceful stop support (Req 14)
        self._training_controller = TrainingController()

        # Initialize ProgressiveTrainer for phase C→B→A management (Req 16)
        self._progressive_trainer = ProgressiveTrainer()

        # Initialize TrainingCycleManager for cycle-based training
        self._cycle_manager = TrainingCycleManager(self, str(self._base_dir))

        # Attempt to load model and normalization params
        self._try_load_model()
        self._try_load_norm_params()

        # Sync initial model version into training status so UI shows it immediately
        if self._model_manager.version > 0:
            with self._background_training._status_lock:
                self._background_training._status.model_version = self._model_manager.version

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

    @property
    def background_training(self) -> BackgroundTrainingManager:
        """Access to the background training manager."""
        return self._background_training

    @property
    def training_status(self) -> TrainingStatus:
        """Current training status for UI display.

        Updated at least every 10 seconds during training (Req 7.5).
        """
        return self._background_training.status

    @property
    def is_training(self) -> bool:
        """Whether background training is active."""
        return self._background_training.is_training

    @property
    def gpu_monitor(self) -> GPUMemoryMonitor:
        """Access to GPU memory monitor for resource info."""
        return self._gpu_monitor

    @property
    def training_controller(self) -> TrainingController:
        """Access to the TrainingController for graceful stop support.

        The TrainingController provides request_stop() and get_status()
        methods for UI integration.

        Requirements: 14.1, 14.4, 14.7
        """
        return self._training_controller

    @property
    def progressive_trainer(self) -> ProgressiveTrainer:
        """Access to the ProgressiveTrainer for phase management.

        The ProgressiveTrainer provides current_phase, transition_criteria,
        validation_history, confirm_transition(), and revert_phase() for
        UI integration.

        Requirements: 16.7, 16.9, 16.10
        """
        return self._progressive_trainer

    @property
    def cycle_manager(self) -> TrainingCycleManager:
        """Access to the TrainingCycleManager for cycle-based training.

        Provides run_cycle(), history, status_summary for the
        "càng train nhiều càng tránh sai lầm" concept.
        """
        return self._cycle_manager

    # --------------------------------------------------------------------------
    # Initialization Helpers
    # --------------------------------------------------------------------------

    def _try_load_model(self) -> None:
        """Attempt to load the evaluation model from the configured path.

        Logs a warning if the model file is not found but does not raise
        an error - the model can be loaded later or training can be run.
        """
        model_path = self._base_dir / self._config.model_path
        # Fallback: if base_dir is a subdirectory (e.g. "data/"), try parent
        if not model_path.exists():
            model_path = self._base_dir.parent / self._config.model_path
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
        # Fallback: if base_dir is a subdirectory (e.g. "data/"), try parent
        if not norm_path.exists():
            norm_path = self._base_dir.parent / self._config.norm_params_path
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
                    "suggestion": "Run DataPipeline.update_symbol() to download data.",
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
        from engine.decision_engine_impl import run_backtest

        return run_backtest(self, symbol, start_date, end_date, initial_capital)

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
        from engine.decision_engine_impl import run_compare

        return run_compare(self, symbol, start_date, end_date, initial_capital)

    def compare_all_symbols(
        self,
        start_date: str,
        end_date: str,
        initial_capital: Optional[float] = None,
        data_dir: Optional[str] = None,
    ) -> Dict:
        """Compare strategies across ALL available symbols.

        Each strategy runs on every symbol and the results are aggregated.
        This shows which strategy performs best overall across the market,
        and which symbols each strategy selects as winners.

        Parameters
        ----------
        start_date : str
            Start date (format: 'YYYY-MM-DD').
        end_date : str
            End date (format: 'YYYY-MM-DD').
        initial_capital : float, optional
            Starting capital per backtest. Uses config default if not provided.
        data_dir : str, optional
            Data directory. Uses engine base_dir if not provided.

        Returns
        -------
        Dict with keys: per_strategy, per_symbol, best_symbols, metadata
        """
        from engine.decision_engine_impl import run_compare_multi_symbol

        return run_compare_multi_symbol(
            self, start_date, end_date, initial_capital, data_dir
        )

    # --------------------------------------------------------------------------
    # Background Training (Concurrent Training/Inference)
    # --------------------------------------------------------------------------

    def start_background_training(
        self,
        symbol_data: Optional[Dict[str, pd.DataFrame]] = None,
        mode: str = "incremental",
        symbols: Optional[List[str]] = None,
        data_dir: Optional[str] = None,
        resume_from_checkpoint: bool = False,
        session_state_dict: Optional[Dict] = None,
    ) -> bool:
        """Start background training without blocking inference.

        Inference continues to use the current model during training.
        When training completes, the model is hot-swapped within 3 seconds
        without dropping in-flight requests.

        If no symbol_data is provided, loads from base_dir for the given symbols
        (default: VN30 + VNINDEX).

        Requirements: 7.1, 7.2, 7.3, 6.11, 5.4, 5.5

        Parameters
        ----------
        symbol_data : Dict[str, pd.DataFrame], optional
            Pre-loaded DataFrames. If None, loads from CSV files.
        mode : str
            "full" for full retraining, "incremental" for fine-tuning.
        symbols : List[str], optional
            Symbols to train on. If None, uses VN30 + VNINDEX defaults.
        data_dir : str, optional
            Base directory for data loading. If None, uses engine's base_dir.
        resume_from_checkpoint : bool
            If True, resume from existing checkpoint.
        session_state_dict : Dict, optional
            Reference to st.session_state for progress updates.

        Returns
        -------
        bool
            True if training started, False if already running.
        """
        from engine.decision_engine_impl import start_background_training_impl

        return start_background_training_impl(
            self, symbol_data, mode, symbols,
            data_dir=data_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            session_state_dict=session_state_dict,
        )

    def stop_background_training(self) -> None:
        """Stop background training gracefully.

        Training stops after the current epoch completes.
        Also signals the TrainingController for graceful stop (Req 14).
        """
        # Signal the TrainingController (if training is active via controller)
        if self._training_controller.is_training:
            try:
                self._training_controller.request_stop()
            except Exception:
                pass  # Controller may not have active training

        self._background_training.stop_training()

    def queue_training_symbol(self, symbol: str) -> None:
        """Queue a symbol for the next incremental training session.

        Called from UI when user adds new stock symbols.
        The symbol will be included in the next training session
        without restarting the application.

        Requirement 6.11

        Parameters
        ----------
        symbol : str
            Stock ticker symbol to queue.
        """
        self._background_training.queue_symbol(symbol)

    def queue_training_symbols(self, symbols: List[str]) -> None:
        """Queue multiple symbols for incremental training.

        Parameters
        ----------
        symbols : List[str]
            Stock ticker symbols to queue.
        """
        self._background_training.queue_symbols(symbols)

    def set_training_status_callback(
        self, callback: Callable[[TrainingStatus], None]
    ) -> None:
        """Set callback for training status updates (for UI).

        The callback is invoked after each epoch with current status
        including epoch, loss, and ETA.

        Requirement 7.5: Updated at least every 10 seconds.

        Parameters
        ----------
        callback : Callable[[TrainingStatus], None]
            Function called with TrainingStatus updates.
        """
        self._background_training.set_status_callback(callback)

    def set_training_paused_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for training pause notifications.

        Called when training is paused due to GPU memory pressure.

        Requirement 7.6

        Parameters
        ----------
        callback : Callable[[str], None]
            Function called with pause reason string.
        """
        self._background_training.set_paused_callback(callback)

    def _get_default_training_symbols(self) -> List[str]:
        """Get default list of training symbols (VN30 + VNINDEX).

        Returns
        -------
        List[str]
            Default symbol list for training.
        """
        vn30_symbols = [
            "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR",
            "HDB", "HPG", "MBB", "MSN", "MWG", "PLX", "POW", "SAB",
            "SHB", "SSB", "SSI", "STB", "TCB", "TPB", "VCB", "VHM",
            "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
        ]
        return ["VNINDEX"] + vn30_symbols

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
