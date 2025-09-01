"""
Trading strategy implementation for Quantformer model.

This module implements the trading strategy described in Algorithm 1 of the paper,
including portfolio management, weight calculation, and backtesting utilities.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class TradingConfig:
    """Configuration for trading strategy."""
    n_classes: int = 3  # Number of quantile classes (ρ in paper)
    decision_factor: int = 1  # Number of quantile groups to select (b in paper)
    phi: float = 0.2  # Percentage of stocks for each quantile
    transaction_fee: float = 0.003  # Transaction fee (0.3% as in paper)
    initial_capital: float = 1000000.0  # Initial portfolio value
    rebalance_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly'


class QuantformerTradingStrategy:
    """
    Trading strategy implementation based on Quantformer predictions.
    
    This class implements Algorithm 1 from the paper, handling:
    1. Stock ranking based on model predictions
    2. Portfolio weight calculation
    3. Trading execution with transaction costs
    4. Performance tracking
    """
    
    def __init__(self, model: torch.nn.Module, config: TradingConfig = None, wandb_logger=None):
        """
        Initialize trading strategy.
        
        Args:
            model: Trained Quantformer model
            config: Trading configuration
            wandb_logger: Optional WandB logger for experiment tracking
        """
        self.model = model
        self.config = config or TradingConfig()
        self.wandb_logger = wandb_logger
        
        # Initialize portfolio state
        self.portfolio_value = self.config.initial_capital
        self.current_weights = None
        self.previous_weights = None
        self.holdings = {}
        
        # Performance tracking
        self.portfolio_history = []
        self.returns_history = []
        self.weights_history = []
        self.turnover_history = []
        
        # Calculate boundary term xi for non-overlapping intervals
        self.xi = (1 - self.config.phi * self.config.n_classes) / (self.config.n_classes - 1) \
                  if self.config.n_classes > 1 else 0
    
    def predict_stock_rankings(self, features: torch.Tensor) -> np.ndarray:
        """
        Generate stock predictions and rankings using the model.
        
        Args:
            features: Input features of shape (n_stocks, seq_len, n_features)
            
        Returns:
            Predicted probabilities for each stock
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(features)
        
        return predictions.cpu().numpy()
    
    def calculate_empirical_quantile_cdf(self, values: np.ndarray) -> np.ndarray:
        """
        Calculate empirical quantile CDF as described in the paper.
        
        Args:
            values: Array of values to rank
            
        Returns:
            Empirical quantile CDF values
        """
        n = len(values)
        sorted_indices = np.argsort(values)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(n)
        
        return ranks / n
    
    def create_trading_strategy_vector(self) -> np.ndarray:
        """
        Create trading strategy vector Φ based on configuration.
        
        Returns:
            Strategy vector indicating which quantiles to select
        """
        strategy = np.zeros(self.config.n_classes)
        
        # Select top quantiles (assuming we want to buy high-performing stocks)
        # This can be modified based on specific strategy requirements
        for i in range(self.config.decision_factor):
            strategy[-(i+1)] = 1  # Select top quantiles
            
        return strategy
    
    def sort_stocks_by_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Sort stocks into quantile groups based on predictions.
        
        Args:
            predictions: Model predictions of shape (n_stocks, n_classes)
            
        Returns:
            Sorted prediction labels for each stock
        """
        n_stocks = len(predictions)
        
        # Use first class probability for ranking (as in paper)
        first_class_probs = predictions[:, 0]
        quantiles = self.calculate_empirical_quantile_cdf(first_class_probs)
        
        # Create sorted labels
        sorted_labels = np.zeros((n_stocks, self.config.n_classes))
        
        for i in range(self.config.n_classes):
            # Define quantile range for each class
            lower_bound = (i * self.config.n_classes + i * self.xi) / self.config.n_classes
            upper_bound = ((i + 1) * self.config.phi + i * self.xi)
            
            # Assign labels
            mask = (quantiles >= lower_bound) & (quantiles < upper_bound)
            sorted_labels[mask, i] = 1
            
        return sorted_labels
    
    def calculate_portfolio_weights(self, 
                                  current_predictions: np.ndarray,
                                  previous_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate portfolio weights based on predictions and strategy.
        
        Args:
            current_predictions: Current model predictions
            previous_labels: Previous period sorted labels (for continuity)
            
        Returns:
            Portfolio weights for each stock
        """
        n_stocks = len(current_predictions)
        
        # Sort current predictions
        current_labels = self.sort_stocks_by_predictions(current_predictions)
        
        # Create strategy vector
        strategy = self.create_trading_strategy_vector()
        
        # If no previous labels, use current labels
        if previous_labels is None:
            previous_labels = current_labels
        
        # Calculate weights according to equation in paper
        weights = np.zeros(n_stocks)
        
        for i in range(n_stocks):
            # Calculate numerator: previous_label * strategy^T * current_label
            numerator = np.dot(previous_labels[i], strategy) * np.dot(strategy, current_labels[i])
            weights[i] = numerator
        
        # Normalize weights
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight
        else:
            # Equal weights if no stocks selected
            weights = np.ones(n_stocks) / n_stocks
        
        return weights
    
    def calculate_turnover_rate(self, new_weights: np.ndarray) -> float:
        """
        Calculate portfolio turnover rate.
        
        Args:
            new_weights: New portfolio weights
            
        Returns:
            Turnover rate
        """
        if self.previous_weights is None:
            return 0.0
        
        return 0.5 * np.sum(np.abs(new_weights - self.previous_weights))
    
    def execute_trades(self, 
                      new_weights: np.ndarray,
                      stock_prices: np.ndarray,
                      stock_returns: np.ndarray) -> Dict[str, float]:
        """
        Execute trades based on new portfolio weights.
        
        Args:
            new_weights: Target portfolio weights
            stock_prices: Current stock prices
            stock_returns: Stock returns for the period
            
        Returns:
            Trading results dictionary
        """
        # Calculate turnover and transaction costs
        turnover_rate = self.calculate_turnover_rate(new_weights)
        transaction_costs = turnover_rate * self.config.transaction_fee * self.portfolio_value
        
        # Update portfolio value based on returns
        if self.current_weights is not None:
            # Calculate portfolio return
            portfolio_return = np.sum(self.current_weights * stock_returns)
            self.portfolio_value *= (1 + portfolio_return)
        
        # Subtract transaction costs
        self.portfolio_value -= transaction_costs
        
        # Update weights
        self.previous_weights = self.current_weights.copy() if self.current_weights is not None else None
        self.current_weights = new_weights.copy()
        
        # Record history
        self.portfolio_history.append(self.portfolio_value)
        if self.previous_weights is not None:
            portfolio_return = np.sum(self.previous_weights * stock_returns)
            self.returns_history.append(portfolio_return)
        self.weights_history.append(new_weights.copy())
        self.turnover_history.append(turnover_rate)
        
        return {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': portfolio_return if self.previous_weights is not None else 0.0,
            'turnover_rate': turnover_rate,
            'transaction_costs': transaction_costs
        }
    
    def backtest(self, 
                features_data: np.ndarray,
                returns_data: np.ndarray,
                prices_data: Optional[np.ndarray] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Run backtest on historical data.
        
        Args:
            features_data: Feature data of shape (n_timesteps, n_stocks, seq_len, n_features)
            returns_data: Return data of shape (n_timesteps, n_stocks)
            prices_data: Price data (optional, used for transaction cost calculation)
            
        Returns:
            Backtest results dictionary
        """
        n_timesteps, n_stocks = returns_data.shape[:2]
        
        if prices_data is None:
            # Use unit prices if not provided
            prices_data = np.ones((n_timesteps, n_stocks))
        
        previous_labels = None
        
        print(f"Starting backtest with {n_timesteps} timesteps and {n_stocks} stocks...")
        
        for t in range(n_timesteps - 1):  # -1 because we need next period returns
            try:
                # Get features for current timestep
                current_features = torch.FloatTensor(features_data[t])
                
                # Generate predictions
                predictions = self.predict_stock_rankings(current_features)
                
                # Calculate portfolio weights
                new_weights = self.calculate_portfolio_weights(predictions, previous_labels)
                
                # Execute trades
                next_returns = returns_data[t + 1]
                current_prices = prices_data[t]
                
                trade_results = self.execute_trades(new_weights, current_prices, next_returns)
                
                # Update previous labels for next iteration
                previous_labels = self.sort_stocks_by_predictions(predictions)
                
                if (t + 1) % 50 == 0:
                    print(f"Processed timestep {t + 1}/{n_timesteps - 1}, "
                          f"Portfolio value: ${self.portfolio_value:,.2f}")
                    
            except Exception as e:
                warnings.warn(f"Error at timestep {t}: {str(e)}")
                continue
        
        # Calculate performance metrics
        results = self.calculate_performance_metrics()
        
        # Log to WandB if available
        if self.wandb_logger and self.wandb_logger.enabled:
            portfolio_df = self.get_portfolio_history()
            self.wandb_logger.log_trading_results(results, portfolio_df)
        
        print(f"Backtest completed. Final portfolio value: ${self.portfolio_value:,.2f}")
        
        return results
    
    def calculate_performance_metrics(self, 
                                    benchmark_returns: Optional[np.ndarray] = None,
                                    risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            benchmark_returns: Benchmark returns for comparison
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            Dictionary of performance metrics
        """
        if not self.returns_history:
            return {}
        
        returns = np.array(self.returns_history)
        
        # Basic metrics
        total_return = (self.portfolio_value / self.config.initial_capital) - 1
        n_periods = len(returns)
        
        # Annualized metrics (assuming daily data)
        periods_per_year = 252
        annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        volatility = np.std(returns) * np.sqrt(periods_per_year)
        
        # Sharpe ratio
        excess_returns = returns - risk_free_rate / periods_per_year
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(periods_per_year) \
                      if np.std(returns) > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_rate = np.mean(returns > 0)
        
        # Average turnover rate
        avg_turnover = np.mean(self.turnover_history) if self.turnover_history else 0
        
        # Alpha and Beta (if benchmark provided)
        alpha, beta = 0.0, 1.0
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            # Simple linear regression for beta
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            
            # Alpha calculation
            benchmark_mean = np.mean(benchmark_returns)
            alpha = np.mean(returns) - beta * benchmark_mean
        
        # Value at Risk (99%)
        var_99 = np.percentile(returns, 1)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_excess_return': annual_return - risk_free_rate,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'average_turnover': avg_turnover,
            'alpha': alpha,
            'beta': beta,
            'var_99': var_99,
            'final_portfolio_value': self.portfolio_value
        }
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """
        Get portfolio history as DataFrame.
        
        Returns:
            DataFrame with portfolio performance over time
        """
        if not self.portfolio_history:
            return pd.DataFrame()
        
        data = {
            'portfolio_value': self.portfolio_history,
            'returns': [0] + self.returns_history,
            'turnover': [0] + self.turnover_history
        }
        
        # Add cumulative returns
        returns_array = np.array(data['returns'])
        data['cumulative_return'] = np.cumprod(1 + returns_array) - 1
        
        return pd.DataFrame(data)


def run_strategy_comparison(models: Dict[str, torch.nn.Module],
                          features_data: np.ndarray,
                          returns_data: np.ndarray,
                          config: TradingConfig = None) -> pd.DataFrame:
    """
    Compare multiple trading strategies.
    
    Args:
        models: Dictionary of model name to model instance
        features_data: Feature data for backtesting
        returns_data: Return data for backtesting
        config: Trading configuration
        
    Returns:
        Comparison results as DataFrame
    """
    results = []
    
    for model_name, model in models.items():
        print(f"\nRunning backtest for {model_name}...")
        
        strategy = QuantformerTradingStrategy(model, config)
        backtest_results = strategy.backtest(features_data, returns_data)
        
        backtest_results['model'] = model_name
        results.append(backtest_results)
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage with dummy data
    from quantformer.model import create_quantformer_model
    from quantformer.data_utils import create_sample_data
    
    print("Creating sample model and data...")
    
    # Create model
    model = create_quantformer_model()
    
    # Create sample data
    n_stocks, n_timesteps = 50, 100
    feature_data, return_data = create_sample_data(n_stocks, n_timesteps)
    
    # Reshape for strategy (add sequence dimension)
    seq_len = 20
    features_for_strategy = np.zeros((n_timesteps - seq_len, n_stocks, seq_len, 2))
    
    for t in range(n_timesteps - seq_len):
        features_for_strategy[t] = feature_data[t:t + seq_len].transpose(1, 0, 2)
    
    returns_for_strategy = return_data[seq_len:]
    
    # Run backtest
    config = TradingConfig(initial_capital=100000)
    strategy = QuantformerTradingStrategy(model, config)
    
    results = strategy.backtest(features_for_strategy, returns_for_strategy)
    
    print("\nBacktest Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
