"""
Data preprocessing utilities for Quantformer model.

This module provides functions for:
1. Stock data normalization (Z-score normalization)
2. Label generation based on quantile ranking
3. Data loading and batching utilities
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Union
from sklearn.preprocessing import StandardScaler
import warnings


class StockDataProcessor:
    """
    Processor for stock data preprocessing as described in the Quantformer paper.
    
    This class handles:
    - Z-score normalization of features
    - Quantile-based label generation
    - Rolling window creation for time series
    """
    
    def __init__(self, seq_len: int = 20, n_classes: int = 3, phi: float = 0.2):
        """
        Initialize the data processor.
        
        Args:
            seq_len: Length of input sequences (default: 20 as in paper)
            n_classes: Number of output classes (3 or 5)
            phi: Percentage of stocks for each quantile (default: 0.2 for 20%)
        """
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.phi = phi
        self.scaler = StandardScaler()
        
        # Calculate boundary term xi for non-overlapping intervals
        self.xi = (1 - phi * n_classes) / (n_classes - 1) if n_classes > 1 else 0
        
    def normalize_features(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Z-score normalization to features.
        
        Args:
            data: Input data of shape (n_samples, n_features)
            
        Returns:
            Normalized data with zero mean and unit variance
        """
        return self.scaler.fit_transform(data)
    
    def create_quantile_labels(self, returns: np.ndarray) -> np.ndarray:
        """
        Create quantile-based labels as described in the paper.
        
        Args:
            returns: Array of stock returns for next period
            
        Returns:
            One-hot encoded labels based on quantile ranking
        """
        n_stocks = len(returns)
        
        # Calculate empirical quantile CDF
        sorted_indices = np.argsort(returns)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(n_stocks)
        quantiles = ranks / n_stocks
        
        # Create labels based on quantile ranges
        labels = np.zeros((n_stocks, self.n_classes))
        
        if self.n_classes == 3 and self.phi == 0.2:
            # Special case for 3 classes with 20% each (as in paper)
            # Bottom 20%, Middle 20%, Top 20%
            for i, (lower, upper) in enumerate([(0.0, 0.2), (0.4, 0.6), (0.8, 1.0)]):
                mask = (quantiles >= lower) & (quantiles < upper)
                labels[mask, i] = 1
        elif self.n_classes == 5 and self.phi == 0.2:
            # Five equal quantiles of 20% each
            for i in range(self.n_classes):
                lower = i * 0.2
                upper = (i + 1) * 0.2
                mask = (quantiles >= lower) & (quantiles < upper)
                labels[mask, i] = 1
        else:
            # General case - equal quantiles
            quantile_size = 1.0 / self.n_classes
            for i in range(self.n_classes):
                lower = i * quantile_size
                upper = (i + 1) * quantile_size
                if i == self.n_classes - 1:  # Last quantile includes 1.0
                    mask = (quantiles >= lower) & (quantiles <= upper)
                else:
                    mask = (quantiles >= lower) & (quantiles < upper)
                labels[mask, i] = 1
            
        return labels
    
    def create_sequences(self, data: np.ndarray, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create rolling window sequences for training.
        
        Args:
            data: Feature data of shape (n_timesteps, n_stocks, n_features)
            returns: Return data for label generation
            
        Returns:
            Tuple of (sequences, labels) for training
        """
        n_timesteps, n_stocks, n_features = data.shape
        
        sequences = []
        labels = []
        
        for t in range(self.seq_len, n_timesteps - 1):
            # Create input sequence
            seq = data[t - self.seq_len:t]  # Shape: (seq_len, n_stocks, n_features)
            
            # Create labels for next timestep
            next_returns = returns[t + 1]
            seq_labels = self.create_quantile_labels(next_returns)
            
            # Store sequences for each stock
            for stock_idx in range(n_stocks):
                if not np.isnan(seq[:, stock_idx]).any():  # Skip if any NaN values
                    sequences.append(seq[:, stock_idx])  # Shape: (seq_len, n_features)
                    labels.append(seq_labels[stock_idx])
        
        return np.array(sequences), np.array(labels)


class StockDataset(Dataset):
    """
    PyTorch Dataset for stock data.
    """
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            sequences: Input sequences of shape (n_samples, seq_len, n_features)
            labels: Labels of shape (n_samples, n_classes)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def load_stock_data(file_path: str, 
                   return_col: str = 'return', 
                   volume_col: str = 'turnover_rate',
                   date_col: str = 'date',
                   stock_col: str = 'stock_id') -> pd.DataFrame:
    """
    Load stock data from CSV file.
    
    Args:
        file_path: Path to CSV file
        return_col: Column name for returns
        volume_col: Column name for turnover rate
        date_col: Column name for dates
        stock_col: Column name for stock identifiers
        
    Returns:
        Loaded and formatted DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        
        # Ensure required columns exist
        required_cols = [return_col, volume_col, date_col, stock_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date and stock
        df = df.sort_values([date_col, stock_col])
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {str(e)}")


def create_sample_data(n_stocks: int = 100, 
                      n_timesteps: int = 500, 
                      seq_len: int = 20,
                      random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample stock data for testing and demonstration.
    
    Args:
        n_stocks: Number of stocks
        n_timesteps: Number of time steps
        seq_len: Sequence length for model input
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (feature_data, return_data)
    """
    np.random.seed(random_seed)
    
    # Create more realistic stock data with different regimes
    returns = np.zeros((n_timesteps, n_stocks))
    turnover_rates = np.zeros((n_timesteps, n_stocks))
    
    # Generate different stock "types" with varying characteristics
    for stock in range(n_stocks):
        # Each stock has different volatility and drift
        stock_vol = np.random.uniform(0.15, 0.4)  # Annual volatility 15-40%
        stock_drift = np.random.normal(0.0, 0.001)  # Small random drift
        
        # Generate returns with some autocorrelation to make prediction harder
        stock_returns = np.random.normal(stock_drift, stock_vol / np.sqrt(252), n_timesteps)
        
        # Add some momentum/mean reversion effects
        for t in range(1, n_timesteps):
            momentum = 0.05 * stock_returns[t-1]  # Small momentum effect
            noise = np.random.normal(0, stock_vol / np.sqrt(252))
            stock_returns[t] = stock_drift + momentum + noise
        
        returns[:, stock] = stock_returns
        
        # Turnover rates correlated with absolute returns (higher volatility = higher turnover)
        base_turnover = np.random.uniform(0.01, 0.1)  # Base turnover rate
        vol_effect = np.abs(stock_returns) * 5  # Higher turnover when volatile
        turnover_rates[:, stock] = base_turnover + vol_effect + np.random.exponential(0.02, n_timesteps)
    
    # Combine features
    feature_data = np.stack([returns, turnover_rates], axis=-1)
    
    # Future returns with some noise to make prediction challenging
    next_period_returns = np.roll(returns, -1, axis=0)
    next_period_returns[-1] = np.random.normal(0, 0.02, n_stocks)  # Random last period
    
    # Add some market-wide effects to make it more realistic
    market_effect = np.random.normal(0, 0.01, n_timesteps)
    for t in range(n_timesteps):
        next_period_returns[t] += market_effect[t] * np.random.uniform(0.5, 1.5, n_stocks)
    
    return feature_data, next_period_returns


def prepare_training_data(feature_data: np.ndarray, 
                         return_data: np.ndarray,
                         seq_len: int = 20,
                         n_classes: int = 3,
                         phi: float = 0.2,
                         train_ratio: float = 0.8,
                         batch_size: int = 64,
                         random_seed: int = 42) -> Tuple[DataLoader, DataLoader, StockDataProcessor]:
    """
    Prepare training and validation data loaders.
    
    Args:
        feature_data: Feature data of shape (n_timesteps, n_stocks, n_features)
        return_data: Return data for label generation
        seq_len: Sequence length
        n_classes: Number of output classes
        phi: Percentage of stocks for each quantile
        train_ratio: Ratio of data for training
        batch_size: Batch size for data loaders
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, processor)
    """
    # Initialize processor
    processor = StockDataProcessor(seq_len=seq_len, n_classes=n_classes, phi=phi)
    
    # Create sequences and labels
    sequences, labels = processor.create_sequences(feature_data, return_data)
    
    # Filter out samples with null labels (if n_classes=3 and phi=0.2)
    if n_classes == 3 and phi == 0.2:
        # Keep only samples with non-zero labels
        valid_samples = labels.sum(axis=1) > 0
        print(f"Before filtering: {len(sequences)} samples")
        print(f"Valid samples: {valid_samples.sum()} / {len(valid_samples)}")
        sequences = sequences[valid_samples]
        labels = labels[valid_samples]
        print(f"After filtering: {len(sequences)} samples")
        
        # Print label distribution
        label_counts = labels.sum(axis=0)
        print(f"Label distribution: {label_counts}")
        label_percentages = label_counts / len(labels) * 100
        print(f"Label percentages: [{label_percentages[0]:.1f}%, {label_percentages[1]:.1f}%, {label_percentages[2]:.1f}%]")
    
    # Split into train and validation
    n_samples = len(sequences)
    n_train = int(n_samples * train_ratio)
    
    # Shuffle data
    np.random.seed(random_seed)
    indices = np.random.permutation(n_samples)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Create datasets
    train_dataset = StockDataset(sequences[train_indices], labels[train_indices])
    val_dataset = StockDataset(sequences[val_indices], labels[val_indices])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, processor


def calculate_portfolio_metrics(returns: np.ndarray, 
                               weights: np.ndarray,
                               risk_free_rate: float = 0.02) -> dict:
    """
    Calculate portfolio performance metrics.
    
    Args:
        returns: Portfolio returns
        weights: Portfolio weights over time
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        
    Returns:
        Dictionary of performance metrics
    """
    # Calculate basic metrics
    total_return = np.prod(1 + returns) - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = np.std(returns) * np.sqrt(252)
    
    # Sharpe ratio
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    # Maximum drawdown
    cumulative_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Win rate
    win_rate = np.mean(returns > 0)
    
    # Turnover rate
    if len(weights) > 1:
        turnover_rate = np.mean(np.sum(np.abs(np.diff(weights, axis=0)), axis=1)) / 2
    else:
        turnover_rate = 0.0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'turnover_rate': turnover_rate
    }


if __name__ == "__main__":
    # Example usage
    print("Creating sample data...")
    feature_data, return_data = create_sample_data(n_stocks=100, n_timesteps=500)
    
    print("Preparing training data...")
    train_loader, val_loader, processor = prepare_training_data(
        feature_data, return_data, batch_size=32
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Test a batch
    for batch_x, batch_y in train_loader:
        print(f"Batch input shape: {batch_x.shape}")
        print(f"Batch label shape: {batch_y.shape}")
        break
