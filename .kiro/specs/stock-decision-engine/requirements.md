# Requirements Document

## Introduction

Stock Decision Engine là một module mới cho ứng dụng phân tích kỹ thuật chứng khoán Việt Nam hiện tại, kết hợp hai mô hình ra quyết định:

1. **Search-based (Minimax/Alpha-Beta):** Khám phá cây quyết định các kịch bản giá tương lai bằng phương pháp tìm kiếm kiểu engine cờ vua, trong đó "nước đi" là các hành động MUA/GIỮ/BÁN và "đối thủ" là thị trường.
2. **ML-based Evaluation (Deep Learning):** Mạng neural đánh giá trạng thái thị trường (position evaluation) tương tự cách AlphaZero đánh giá thế cờ, chạy local trên GPU RTX 2060.

Toàn bộ hệ thống hoạt động offline, không gọi API bên ngoài khi inference, chỉ sử dụng dữ liệu OHLCV đã được tải về dưới dạng file CSV và các chỉ báo kỹ thuật đã có sẵn trong `analysis.py`.

## Glossary

- **Decision_Engine**: Module chính điều phối quá trình ra quyết định, kết hợp Search Module và Evaluation Model
- **Search_Module**: Component thực hiện thuật toán Minimax với cắt tỉa Alpha-Beta để khám phá cây quyết định
- **Evaluation_Model**: Mạng neural network chạy local, đánh giá điểm số cho một trạng thái thị trường
- **Market_State**: Đối tượng đại diện cho trạng thái thị trường tại một thời điểm, bao gồm chuỗi OHLCV và giá trị các chỉ báo kỹ thuật
- **Action**: Hành động giao dịch có thể thực hiện: BUY (mua), HOLD (giữ), SELL (bán)
- **Search_Depth**: Số bước tương lai mà Search_Module khám phá trong cây quyết định
- **Position_Score**: Điểm số từ -1.0 đến +1.0 do Evaluation_Model trả về, đánh giá mức độ thuận lợi của Market_State
- **Training_Pipeline**: Quy trình huấn luyện Evaluation_Model từ dữ liệu lịch sử
- **Technical_Indicator**: Các chỉ báo kỹ thuật được tính từ dữ liệu OHLCV (EMA, RSI, MACD, Bollinger Bands, ADX, OBV, v.v.)
- **Feature_Vector**: Vector đặc trưng đầu vào cho Evaluation_Model, được tạo từ Technical_Indicator và dữ liệu OHLCV
- **Scenario_Generator**: Component tạo ra các kịch bản giá tương lai hợp lý dựa trên phân bố thống kê từ dữ liệu lịch sử
- **Decision_Report**: Báo cáo kết quả ra quyết định bao gồm Action khuyến nghị, Position_Score, và phân tích chi tiết
- **Backtest_Engine**: Component chạy mô phỏng giao dịch trên dữ liệu lịch sử để đánh giá hiệu quả chiến lược
- **Philosophy_Strategy**: Một chiến lược giao dịch dựa trên triết lý cụ thể (Wyckoff, Technical Indicators, Momentum, Mean Reversion)
- **Equity_Curve**: Biểu đồ thể hiện giá trị danh mục đầu tư theo thời gian trong quá trình backtest
- **Win_Rate**: Tỷ lệ phần trăm giao dịch có lời trên tổng số giao dịch
- **Max_Drawdown**: Mức sụt giảm lớn nhất từ đỉnh đến đáy của equity curve
- **Sharpe_Ratio**: Tỷ số đo lường lợi nhuận điều chỉnh theo rủi ro

## Requirements

### Requirement 1: Offline Operation

**User Story:** As a trader, I want the decision engine to run entirely on my local machine, so that I do not depend on external services or internet connectivity during analysis.

#### Acceptance Criteria

1. WHILE performing technical analysis and generating trading signals, THE Decision_Engine SHALL operate without any external API calls or network connections
2. THE Decision_Engine SHALL load all required data from local CSV files stored in the project directory, where each CSV file contains at minimum the columns: time, open, high, low, close, and volume, with at least 5 rows of data
3. THE Decision_Engine SHALL load Evaluation_Model weights and analysis parameters from local files without network access, completing the load within 30 seconds
4. IF one or more required local CSV files are missing, THEN THE Decision_Engine SHALL return an error message indicating the specific file names that are missing and their expected location in the project directory
5. IF a required local CSV file is present but contains malformed data or is missing required columns, THEN THE Decision_Engine SHALL return an error message identifying the file and the specific data issue detected
6. WHEN the Decision_Engine is started without network connectivity, THE Decision_Engine SHALL complete initialization and be ready to accept analysis requests without failure

### Requirement 2: Market State Representation

**User Story:** As a trader, I want the system to represent market conditions as a structured state object, so that both the search and evaluation modules can process it consistently.

#### Acceptance Criteria

1. THE Decision_Engine SHALL construct a Market_State from OHLCV data and Technical_Indicator values computed by the existing `analysis.py` module
2. THE Market_State SHALL include a configurable lookback window of historical data with a default of 60 trading sessions, configurable between 20 and 200 trading sessions inclusive
3. THE Market_State SHALL include all Technical_Indicator values currently computed by `add_indicators()` in `analysis.py`
4. THE Feature_Vector SHALL normalize all numerical values to the range [0, 1] using min-max scaling computed over the lookback window
5. WHEN a Market_State is serialized and deserialized, THE Decision_Engine SHALL produce a Market_State whose numerical values differ from the original by no more than 1e-9 per field (round-trip integrity)
6. IF the available OHLCV data contains fewer rows than the configured lookback window, THEN THE Decision_Engine SHALL reject Market_State construction and return an error indicating insufficient historical data
7. IF a Technical_Indicator value is NaN for any session within the lookback window, THEN THE Feature_Vector SHALL replace that value with 0.0 after normalization

### Requirement 3: Evaluation Model Architecture

**User Story:** As a trader, I want a neural network model that evaluates market conditions and outputs a position score, so that the search module can assess different scenarios.

#### Acceptance Criteria

1. THE Evaluation_Model SHALL accept a Feature_Vector as input and output a single Position_Score between -1.0 and +1.0, where -1.0 indicates maximally unfavorable and +1.0 indicates maximally favorable market conditions
2. THE Evaluation_Model SHALL have a total parameter count such that model weights plus intermediate activations during inference consume no more than 4GB VRAM on the RTX 2060
3. THE Evaluation_Model SHALL complete inference for a single Market_State within 50 milliseconds on the RTX 2060
4. THE Evaluation_Model SHALL use PyTorch as the deep learning framework for compatibility with CUDA on RTX 2060
5. THE Evaluation_Model SHALL support batch inference of between 1 and 50 Market_State inputs simultaneously, processing a batch of 7 inputs within 100 milliseconds on the RTX 2060
6. IF CUDA is unavailable, THEN THE Evaluation_Model SHALL fall back to CPU inference on the i5-10400F and complete single-input inference within 500 milliseconds
7. IF the Feature_Vector contains only NaN or zero values after normalization, THEN THE Evaluation_Model SHALL return a Position_Score of 0.0 indicating a neutral evaluation

### Requirement 4: Search-Based Decision Exploration

**User Story:** As a trader, I want the system to explore multiple future scenarios using a minimax-style search algorithm, so that the recommended action accounts for various market outcomes.

#### Acceptance Criteria

1. THE Search_Module SHALL implement Minimax search with Alpha-Beta pruning over a tree of Action and Market_State response pairs, where Action is one of buy, sell, or hold for a given stock symbol
2. THE Search_Module SHALL use Evaluation_Model to score leaf nodes of the search tree and return the Action associated with the highest-scoring root-level branch as the recommended action
3. THE Search_Module SHALL support a configurable Search_Depth with a default of 3 levels and a maximum of 5 levels
4. THE Search_Module SHALL complete a full search for one stock symbol within 5 seconds on the target hardware (i5-10400F + RTX 2060)
5. IF the Search_Module cannot complete the search within 5 seconds, THEN THE Search_Module SHALL return the best action found from the deepest fully evaluated level at the time of termination
6. THE Scenario_Generator SHALL produce future Market_State candidates based on statistical distributions derived from a minimum of 30 trading days of historical price data for the given stock symbol
7. THE Scenario_Generator SHALL generate between 3 and 7 plausible scenarios per search node to balance breadth and computation time
8. IF fewer than 30 trading days of historical price data are available for a stock symbol, THEN THE Scenario_Generator SHALL reject the search request and return an error indication specifying insufficient historical data
9. WHEN Search_Depth is increased beyond the default of 3, THE Search_Module SHALL reduce the branching factor per node (within the 3 to 7 scenario range) as needed to maintain the 5-second time constraint

### Requirement 5: Decision Output

**User Story:** As a trader, I want clear actionable output from the decision engine, so that I can make informed trading decisions.

#### Acceptance Criteria

1. THE Decision_Engine SHALL output a Decision_Report containing the stock symbol analyzed and the recommended Action (BUY, HOLD, or SELL)
2. THE Decision_Report SHALL include a confidence score between 0.0 and 1.0 representing the agreement level among the top scenarios explored by Search_Module
3. THE Decision_Report SHALL include the Position_Score from the Evaluation_Model for the current Market_State
4. THE Decision_Report SHALL include the top 3 scenarios explored by Search_Module, each with its Action sequence and Position_Score at the leaf node
5. THE Decision_Report SHALL include the top 5 Technical_Indicator values ranked by absolute contribution to the Position_Score, with each indicator's name and current value
6. IF the confidence score is below 0.3, THEN THE Decision_Engine SHALL recommend HOLD regardless of other signals
7. IF the Search_Module explores fewer than 3 scenarios, THEN THE Decision_Report SHALL include all explored scenarios and indicate that fewer than 3 were available
8. IF the Decision_Engine cannot generate a Decision_Report due to insufficient data or model unavailability, THEN THE Decision_Engine SHALL return an error indicating the reason the report could not be produced

### Requirement 6: Training Pipeline

**User Story:** As a developer, I want a training pipeline that can train the evaluation model from historical data, so that the model learns to evaluate Vietnamese stock market conditions.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL train Evaluation_Model using historical OHLCV data from local CSV files, requiring a minimum of 250 trading sessions of data per stock symbol
2. THE Training_Pipeline SHALL generate training labels by computing the percentage price change over a configurable horizon (default: 5 trading sessions) and mapping the result to a Position_Score target between -1.0 and +1.0 using a normalized return value
3. THE Training_Pipeline SHALL split data into training (70%), validation (15%), and test (15%) sets chronologically, ensuring no future data leaks into earlier sets
4. THE Training_Pipeline SHALL produce a usable initial model within 2 hours of training on the target hardware (i5-10400F + RTX 2060) using the 30 VN30 component stocks plus VNINDEX as the default training dataset
5. THE Training_Pipeline SHALL complete a full training cycle (default VN30 + VNINDEX dataset) within 4 hours on the target hardware
6. THE Training_Pipeline SHALL save trained model weights to a local file in PyTorch format
7. THE Training_Pipeline SHALL log training metrics (loss, mean absolute error, and validation loss) to a local file after each epoch
8. THE Training_Pipeline SHALL save a checkpoint file at the end of every epoch during training, and WHEN training is resumed after interruption, THE Training_Pipeline SHALL continue from the last saved checkpoint (epoch, model weights, and optimizer state)
9. THE Training_Pipeline SHALL support incremental training mode where the model is fine-tuned with new daily data for a maximum of 10 epochs without full retraining
10. THE Training_Pipeline SHALL complete an incremental training session (one day of new data across all trained symbols) within 10 minutes on the target hardware
11. WHEN a user adds additional stock symbols through the UI, THE Training_Pipeline SHALL queue the new symbols for inclusion in the next incremental training session without restarting the application
12. THE Training_Pipeline SHALL always include VNINDEX data in training so that the Evaluation_Model learns overall market context before evaluating individual stocks
13. IF a stock symbol has fewer than 250 trading sessions of data, THEN THE Training_Pipeline SHALL skip that symbol and log a warning indicating insufficient data

### Requirement 7: Concurrent Training and Inference

**User Story:** As a trader, I want to use the decision engine for analysis while training continues in the background, so that I do not have to wait for training to finish before getting recommendations.

#### Acceptance Criteria

1. THE Decision_Engine SHALL support concurrent operation where inference runs on GPU while incremental training is queued, and inference requests SHALL complete within 5 seconds regardless of background training activity
2. WHILE a new training session is in progress, THE Decision_Engine SHALL use the most recently completed model checkpoint for inference
3. WHEN a new training session completes, THE Decision_Engine SHALL hot-swap to the updated model within 3 seconds, without requiring application restart, and without dropping or corrupting any in-flight inference requests
4. THE Decision_Engine SHALL display the model version and last training timestamp in the UI
5. WHILE training is active in the background, THE Decision_Engine SHALL display training progress (epoch, loss, estimated time remaining) in the UI, updated at least every 10 seconds
6. IF GPU memory utilization exceeds 90% during concurrent training and inference, THEN THE Decision_Engine SHALL prioritize inference, pause training until GPU memory utilization drops below 80%, and display a notification in the UI indicating that training has been paused due to resource constraints

### Requirement 8: Feature Vector Construction

**User Story:** As a developer, I want a reliable process to convert raw market data into model-ready feature vectors, so that the evaluation model receives consistent input.

#### Acceptance Criteria

1. THE Decision_Engine SHALL construct Feature_Vector from the Technical_Indicator values already computed by `analysis.py`, maintaining a fixed ordering of all indicator columns as defined in the `add_indicators` function output
2. THE Feature_Vector SHALL handle missing indicator values (NaN) by applying forward-fill along the time axis for each indicator column independently, followed by zero-fill for any remaining NaN values
3. THE Feature_Vector SHALL apply min-max normalization to scale each feature to the range [0, 1], using per-feature minimum and maximum values computed from the training dataset
4. THE Decision_Engine SHALL serialize normalization parameters (per-feature min and max values) to a local JSON file and load them during inference
5. IF the normalization parameter file is missing or unreadable at inference time, THEN THE Decision_Engine SHALL raise an error indicating the missing parameter file and abort feature vector construction without producing partial output
6. FOR ALL valid Market_State inputs, converting to Feature_Vector (normalize) then reconstructing the original indicator values (denormalize) SHALL produce values within 0.01% relative tolerance of the originals (round-trip property)

### Requirement 9: Backtesting Framework

**User Story:** As a trader, I want to backtest the decision engine against historical data, so that I can verify its effectiveness before using it with real capital.

#### Acceptance Criteria

1. THE Decision_Engine SHALL support backtesting on historical CSV data with configurable start date, end date, and initial capital (default 100,000,000 VND)
2. WHEN a backtest is executed, THE Decision_Engine SHALL simulate trade execution using the closing price of the signal day, enforce a minimum lot size of 100 shares, and reject any trade where the executed price would exceed the daily price limit of ±7% from the reference price
3. WHILE a position is held within the T+2.5 settlement period (2.5 trading days after purchase), THE Decision_Engine SHALL prevent any sell action on that position until settlement is complete
4. THE Decision_Engine SHALL track position size as a percentage of simulated portfolio value, with a maximum single position size of 20% of total portfolio value at the time of entry
5. THE Decision_Engine SHALL report total return percentage, win rate, maximum drawdown, and Sharpe ratio (calculated using a risk-free rate of 0%) for the backtest period
6. WHEN a backtest is completed with at least one executed trade, THE Decision_Engine SHALL display an equity curve chart and a list of all executed trades with entry price, exit price, position size, and profit/loss per trade
7. IF the specified date range contains no historical data or the start date is after the end date, THEN THE Decision_Engine SHALL display an error message indicating the invalid configuration and not execute the backtest

### Requirement 10: Philosophy-Based Strategy Comparison

**User Story:** As a trader, I want to compare the AI decision engine against multiple established trading philosophies, so that I can verify the AI outperforms traditional approaches.

#### Acceptance Criteria

1. THE Decision_Engine SHALL implement a Wyckoff-based strategy that generates BUY/HOLD/SELL signals using Wyckoff accumulation/distribution phase detection from existing scanner logic, where a composite Wyckoff score of +3 or above produces BUY, -3 or below produces SELL, and scores in between produce HOLD
2. THE Decision_Engine SHALL implement a Technical_Indicator-based strategy that generates BUY/HOLD/SELL signals using RSI, MACD, EMA crossover, and Bollinger Bands rules from existing `ui_analyze_signals.py` logic, where a composite score of +5 or above produces BUY, -4 or below produces SELL, and scores in between produce HOLD
3. THE Decision_Engine SHALL implement a Momentum-based strategy that generates BUY when ADX is above 25 and price rate-of-change over 14 periods is positive and volume exceeds the 20-period average by at least 1.5x, SELL when ADX is above 25 and rate-of-change is negative, and HOLD otherwise
4. THE Decision_Engine SHALL implement a Mean_Reversion strategy that generates BUY when price is below the lower Bollinger Band or RSI is below 30 or price is more than 2 standard deviations below the 20-period moving average, SELL when price is above the upper Bollinger Band or RSI is above 70 or price is more than 2 standard deviations above the 20-period moving average, and HOLD otherwise
5. THE Decision_Engine SHALL run identical backtest periods across all strategies (AI engine + 4 philosophies) using the same date range of at least 252 trading days, the same initial capital, the same universe of instruments, and the same position sizing rules for fair comparison
6. THE Decision_Engine SHALL display a comparison table showing total return (percentage), annualized return, win rate (percentage of profitable trades), maximum drawdown (percentage), and Sharpe ratio (assuming a risk-free rate of 0%) for each strategy
7. THE Decision_Engine SHALL display overlaid equity curves for all strategies on a single chart with distinct visual identifiers for each strategy line
8. THE Training_Pipeline SHALL use performance against these philosophy-based strategies as part of the training objective, optimizing the Evaluation_Model to achieve a higher Sharpe ratio than all four baseline philosophies on the validation backtest period
9. IF any philosophy-based strategy fails to generate at least one BUY and one SELL signal during the backtest period, THEN THE Decision_Engine SHALL exclude that strategy from the comparison and display a notification indicating insufficient signal generation for that strategy

### Requirement 11: Integration with Existing Application

**User Story:** As a trader, I want the decision engine to integrate with the existing Streamlit application, so that I can access it through the familiar interface.

#### Acceptance Criteria

1. THE Decision_Engine SHALL be accessible as a new tab in the existing Streamlit application (`app.py`), added to the existing `st.tabs()` list alongside the current tabs
2. THE Decision_Engine SHALL use the same CSV data files that are managed by the existing `update_data.py` module, reading from the application base directory using the same column format (time, open, high, low, close, volume, plus indicator columns)
3. THE Decision_Engine SHALL reuse the `add_indicators()` function from `analysis.py` for Technical_Indicator computation
4. WHEN a user selects a stock symbol in the Decision Engine tab, THE Decision_Engine SHALL display the Decision_Report within 10 seconds, containing at minimum: the overall signal (buy/sell/hold), confidence score, and the supporting indicator evidence
5. IF the selected stock symbol has no corresponding CSV data file available, THEN THE Decision_Engine SHALL display an error message indicating that data is unavailable for the selected symbol and prompt the user to update data first
6. THE Decision_Engine SHALL display a visual representation of the top 5 search scenarios explored, rendered as a ranked list or chart showing each scenario's name and score
7. THE Decision_Engine SHALL include a Backtest sub-tab showing strategy comparison results, displaying at minimum: total return percentage, win rate, and maximum drawdown for each strategy over a user-selectable time period (default 1 year)

### Requirement 12: Hardware Resource Management

**User Story:** As a user with limited hardware, I want the system to manage GPU and CPU resources efficiently, so that it runs smoothly on my i5-10400F and RTX 2060 setup.

#### Acceptance Criteria

1. THE Decision_Engine SHALL limit GPU memory usage to a maximum of 4GB of the available 6GB VRAM during inference
2. THE Training_Pipeline SHALL limit GPU memory usage to a maximum of 5.5GB VRAM during training
3. WHEN the Decision_Engine completes inference for a batch of requests, THE Decision_Engine SHALL release all GPU memory allocated for that batch within 1 second
4. WHILE the Decision_Engine is running inference, THE Decision_Engine SHALL process UI input events within 200 milliseconds (non-blocking execution)
5. IF GPU memory allocation fails, THEN THE Decision_Engine SHALL retry up to 3 times, halving the batch size on each retry, before falling back to CPU execution
6. IF CPU fallback execution also fails, THEN THE Decision_Engine SHALL cancel the current request and display an error indication to the user stating insufficient resources

### Requirement 13: Model Serialization

**User Story:** As a developer, I want reliable model save/load functionality, so that trained models can be stored and reused without retraining.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL save the complete model state (architecture, weights, optimizer state, epoch number, and normalization parameters) to a single local file in PyTorch format
2. THE Decision_Engine SHALL load a saved model and produce Position_Score outputs within ±1e-6 tolerance of the original model's outputs for the same Feature_Vector input when run on the same device type (CPU or GPU)
3. THE Decision_Engine SHALL validate model file integrity using a checksum before loading
4. IF the model file fails checksum validation, THEN THE Decision_Engine SHALL return an error indicating file corruption and refuse to load the model
5. IF the model file was saved with a different Feature_Vector dimension or incompatible architecture configuration than the current system expects, THEN THE Decision_Engine SHALL return an error indicating version incompatibility and refuse to load the model
6. THE Decision_Engine SHALL complete model loading and validation within 10 seconds on the target hardware (i5-10400F + RTX 2060)
