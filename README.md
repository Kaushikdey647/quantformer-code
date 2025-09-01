# Quantformer: Transformer-based Quantitative Trading

A PyTorch implementation of the Quantformer model as described in the paper "Quantformer: from attention to profit with a quantitative transformer trading strategy".

## Overview

This implementation adapts the transformer architecture for quantitative trading by:

1. **Linear Embedding**: Replaces word embeddings with linear layers for numerical time series data
2. **No Positional Encoding**: Time series data has inherent temporal order
3. **Simplified Decoder**: Outputs probability distributions for stock ranking
4. **Market Sentiment Integration**: Uses returns and turnover rates as input features

## Features

- ✅ Complete Quantformer model implementation
- ✅ Data preprocessing utilities for stock data
- ✅ Trading strategy with portfolio management
- ✅ Comprehensive backtesting framework
- ✅ Interactive Jupyter notebook interface
- ✅ Performance metrics and visualization

## Project Structure

```
quantformer/
├── quantformer/
│   ├── __init__.py          # Package initialization
│   ├── model.py             # Quantformer model architecture
│   ├── data_utils.py        # Data preprocessing utilities
│   └── trading_strategy.py  # Trading strategy implementation
├── quantformer_demo.ipynb   # Interactive demonstration notebook
├── pyproject.toml          # Project dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd quantformer
```

2. Install dependencies:
```bash
pip install -e .
```

Or install manually:
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn jupyter plotly tqdm
```

## Quick Start

### 1. Using the Jupyter Notebook (Recommended)

Open and run the interactive notebook:
```bash
jupyter notebook quantformer_demo.ipynb
```

The notebook provides a complete walkthrough including:
- Data generation and preprocessing
- Model training
- Trading strategy implementation
- Backtesting and results analysis

### 2. Using the Python API

```python
from quantformer import (
    create_quantformer_model, 
    prepare_training_data, 
    create_sample_data,
    QuantformerTradingStrategy, 
    TradingConfig
)

# Create sample data
feature_data, return_data = create_sample_data(n_stocks=100, n_timesteps=500)

# Prepare training data
train_loader, val_loader, processor = prepare_training_data(
    feature_data, return_data, seq_len=20, n_classes=3
)

# Create and train model
model = create_quantformer_model({
    'input_dim': 2,
    'd_model': 16,
    'n_heads': 8,
    'n_layers': 6,
    'n_classes': 3
})

# Initialize trading strategy
config = TradingConfig(initial_capital=1000000)
strategy = QuantformerTradingStrategy(model, config)

# Run backtest
results = strategy.backtest(features_data, returns_data)
```

## Model Architecture

The Quantformer consists of:

### Core Components
- **Linear Embedding Layer**: Converts numerical features to model dimension
- **Multi-Head Attention**: Captures temporal dependencies (8 heads, 6 layers)
- **Feed-Forward Networks**: Non-linear transformations
- **Classification Head**: Outputs quantile probability distributions

### Key Parameters
- Input dimension: 2 (returns + turnover rates)
- Sequence length: 20 time steps
- Model dimension: 16
- Number of classes: 3 or 5 quantiles
- Dropout: 0.1

## Trading Strategy

The implementation follows Algorithm 1 from the paper:

1. **Stock Ranking**: Use model predictions to rank stocks into quantiles
2. **Portfolio Construction**: Select top quantiles based on strategy parameters
3. **Weight Calculation**: Equal-weight selected stocks with transaction costs
4. **Rebalancing**: Update portfolio based on new predictions

### Strategy Parameters
- Transaction fee: 0.3% (as in paper)
- Quantile selection: Top 20% of stocks (φ = 0.2)
- Rebalancing: Configurable frequency

## Performance Metrics

The backtesting framework calculates comprehensive metrics:

- **Returns**: Total, annual, and excess returns
- **Risk**: Volatility, maximum drawdown, VaR
- **Risk-Adjusted**: Sharpe ratio, alpha, beta
- **Trading**: Win rate, turnover rate

## Data Format

### Input Features
The model expects time series data with:
- **Returns**: Daily profit rates for each stock
- **Turnover Rates**: Daily trading volume indicators
- **Format**: `(n_timesteps, n_stocks, 2)`

### Labels
Quantile-based labels generated from future returns:
- 3-class: Top 20%, Middle 20%, Bottom 20%
- 5-class: Five equal 20% quantiles
- One-hot encoded format

## Customization

### Model Configuration
```python
model_config = {
    'input_dim': 2,        # Number of input features
    'd_model': 16,         # Hidden dimension
    'n_heads': 8,          # Attention heads
    'n_layers': 6,         # Encoder layers
    'd_ff': 64,            # Feed-forward dimension
    'n_classes': 3,        # Output classes
    'seq_len': 20,         # Input sequence length
    'dropout': 0.1         # Dropout rate
}
```

### Trading Configuration
```python
trading_config = TradingConfig(
    n_classes=3,           # Number of quantile classes
    decision_factor=1,     # Number of quantiles to select
    phi=0.2,              # Quantile size (20%)
    transaction_fee=0.003, # Transaction cost (0.3%)
    initial_capital=1e6,   # Starting capital
    rebalance_frequency='monthly'
)
```

## Real Data Integration

To use with real stock data:

1. **Data Format**: Ensure data has columns for returns and turnover rates
2. **Preprocessing**: Apply Z-score normalization
3. **Sequence Creation**: Generate rolling 20-day windows
4. **Label Generation**: Create quantile-based labels from future returns

Example:
```python
# Load your data
df = pd.read_csv('stock_data.csv')

# Preprocess
processor = StockDataProcessor(seq_len=20, n_classes=3)
sequences, labels = processor.create_sequences(feature_data, return_data)

# Train model
train_loader, val_loader, _ = prepare_training_data(
    feature_data, return_data, train_ratio=0.8
)
```

## Paper Implementation Details

This implementation closely follows the original paper:

- **Section 3.2**: Linear embedding replaces word embeddings
- **Section 3.3**: Multi-head attention without positional encoding
- **Section 3.4**: Simplified output for classification
- **Algorithm 1**: Complete trading strategy implementation
- **Section 4**: MSE loss function and training procedure

## Performance Comparison

The paper reports the following results on Chinese stock market data (2010-2023):
- Annual Return: 17.35%
- Sharpe Ratio: 0.915
- Win Rate: 57.8%
- Max Drawdown: 18.35%

Our implementation provides the framework to achieve similar results with real data.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is provided for educational and research purposes. Please cite the original paper when using this implementation.

## Citation

```bibtex
@article{quantformer2024,
  title={Quantformer: from attention to profit with a quantitative transformer trading strategy},
  author={Zhang, Zhaofeng and Chen, Banghao and Zhu, Shengxin and Langrené, Nicolas},
  journal={arXiv preprint},
  year={2024}
}
```

## Acknowledgments

This implementation is based on the research paper by Zhang et al. The original paper provides the theoretical foundation and experimental validation for the Quantformer architecture.
