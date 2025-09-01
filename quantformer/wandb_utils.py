"""
Weights & Biases (WandB) integration utilities for Quantformer experiments.

This module provides utilities for:
1. Experiment configuration and initialization
2. Logging training metrics and hyperparameters
3. Tracking trading strategy performance
4. Model artifact management
"""

import wandb
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
import matplotlib.pyplot as plt
from pathlib import Path
import json


class WandBConfig:
    """Configuration class for WandB experiments."""
    
    def __init__(self, 
                 project_name: str = "quantformer-experiments",
                 entity: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 tags: Optional[list] = None,
                 notes: Optional[str] = None):
        """
        Initialize WandB configuration.
        
        Args:
            project_name: WandB project name
            entity: WandB entity (username or team)
            experiment_name: Name for this specific experiment
            tags: List of tags for the experiment
            notes: Description or notes for the experiment
        """
        self.project_name = project_name
        self.entity = entity
        self.experiment_name = experiment_name
        self.tags = tags or []
        self.notes = notes


class QuantformerWandBLogger:
    """
    WandB logger for Quantformer experiments.
    
    Handles logging of training metrics, hyperparameters, model artifacts,
    and trading strategy performance.
    """
    
    def __init__(self, config: WandBConfig, enabled: bool = True):
        """
        Initialize WandB logger.
        
        Args:
            config: WandB configuration
            enabled: Whether to enable WandB logging
        """
        self.config = config
        self.enabled = enabled
        self.run = None
        
        if self.enabled:
            self._initialize_wandb()
    
    def _initialize_wandb(self):
        """Initialize WandB run."""
        try:
            self.run = wandb.init(
                project=self.config.project_name,
                entity=self.config.entity,
                name=self.config.experiment_name,
                tags=self.config.tags,
                notes=self.config.notes,
                reinit=True
            )
            print(f"✅ WandB initialized: {self.run.url}")
        except Exception as e:
            print(f"⚠️ WandB initialization failed: {e}")
            self.enabled = False
    
    def log_hyperparameters(self, 
                           model_config: Dict[str, Any],
                           training_config: Dict[str, Any],
                           data_config: Dict[str, Any]):
        """
        Log experiment hyperparameters.
        
        Args:
            model_config: Model architecture configuration
            training_config: Training configuration
            data_config: Data configuration
        """
        if not self.enabled:
            return
        
        config_dict = {
            "model": model_config,
            "training": training_config,
            "data": data_config
        }
        
        wandb.config.update(config_dict)
        print("📊 Hyperparameters logged to WandB")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """
        Log training or validation metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step/epoch number
        """
        if not self.enabled:
            return
        
        wandb.log(metrics, step=step)
    
    def log_model_artifact(self, 
                          model: torch.nn.Module, 
                          name: str = "quantformer_model",
                          metadata: Optional[Dict] = None):
        """
        Log model as WandB artifact.
        
        Args:
            model: PyTorch model to save
            name: Artifact name
            metadata: Additional metadata
        """
        if not self.enabled:
            return
        
        try:
            # Save model locally first
            model_path = f"{name}.pth"
            torch.save(model.state_dict(), model_path)
            
            # Create artifact
            artifact = wandb.Artifact(
                name=name,
                type="model",
                metadata=metadata or {}
            )
            artifact.add_file(model_path)
            
            # Log artifact
            wandb.log_artifact(artifact)
            
            # Clean up local file
            Path(model_path).unlink()
            
            print(f"💾 Model artifact '{name}' logged to WandB")
            
        except Exception as e:
            print(f"⚠️ Failed to log model artifact: {e}")
    
    def log_trading_results(self, 
                           results: Dict[str, Any],
                           portfolio_history: Optional[pd.DataFrame] = None):
        """
        Log trading strategy backtest results.
        
        Args:
            results: Trading strategy results dictionary
            portfolio_history: Portfolio performance over time
        """
        if not self.enabled:
            return
        
        # Log summary metrics
        trading_metrics = {}
        for key, value in results.items():
            if isinstance(value, (int, float)):
                trading_metrics[f"trading/{key}"] = value
        
        wandb.log(trading_metrics)
        
        # Log portfolio performance chart
        if portfolio_history is not None and not portfolio_history.empty:
            self._log_portfolio_chart(portfolio_history)
        
        print("📈 Trading results logged to WandB")
    
    def _log_portfolio_chart(self, portfolio_df: pd.DataFrame):
        """Log portfolio performance chart to WandB."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Portfolio value
            axes[0, 0].plot(portfolio_df['portfolio_value'])
            axes[0, 0].set_title('Portfolio Value Over Time')
            axes[0, 0].set_ylabel('Value ($)')
            
            # Cumulative returns
            axes[0, 1].plot(portfolio_df['cumulative_return'] * 100)
            axes[0, 1].set_title('Cumulative Returns')
            axes[0, 1].set_ylabel('Return (%)')
            
            # Daily returns distribution
            daily_returns = portfolio_df['returns'][1:] * 100
            axes[1, 0].hist(daily_returns, bins=30, alpha=0.7)
            axes[1, 0].set_title('Daily Returns Distribution')
            axes[1, 0].set_xlabel('Return (%)')
            
            # Turnover
            axes[1, 1].plot(portfolio_df['turnover'] * 100)
            axes[1, 1].set_title('Portfolio Turnover')
            axes[1, 1].set_ylabel('Turnover (%)')
            
            plt.tight_layout()
            
            # Log to WandB
            wandb.log({"trading/portfolio_performance": wandb.Image(fig)})
            plt.close(fig)
            
        except Exception as e:
            print(f"⚠️ Failed to log portfolio chart: {e}")
    
    def log_training_curves(self, 
                           train_losses: list, 
                           val_losses: list, 
                           val_accuracies: list):
        """
        Log training curves as WandB charts.
        
        Args:
            train_losses: Training losses per epoch
            val_losses: Validation losses per epoch
            val_accuracies: Validation accuracies per epoch
        """
        if not self.enabled:
            return
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Loss curves
            epochs = range(1, len(train_losses) + 1)
            axes[0].plot(epochs, train_losses, label='Training Loss')
            axes[0].plot(epochs, val_losses, label='Validation Loss')
            axes[0].set_title('Training and Validation Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Accuracy curve
            axes[1].plot(epochs, val_accuracies, label='Validation Accuracy', color='green')
            axes[1].axhline(y=0.33, color='gray', linestyle='--', alpha=0.7, label='Random (33%)')
            axes[1].set_title('Validation Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Log to WandB
            wandb.log({"training/learning_curves": wandb.Image(fig)})
            plt.close(fig)
            
            print("📊 Training curves logged to WandB")
            
        except Exception as e:
            print(f"⚠️ Failed to log training curves: {e}")
    
    def log_data_statistics(self, 
                           feature_data: np.ndarray, 
                           return_data: np.ndarray,
                           label_distribution: Optional[np.ndarray] = None):
        """
        Log data statistics and distributions.
        
        Args:
            feature_data: Input feature data
            return_data: Return data
            label_distribution: Distribution of labels across classes
        """
        if not self.enabled:
            return
        
        # Log basic statistics
        data_stats = {
            "data/n_timesteps": feature_data.shape[0],
            "data/n_stocks": feature_data.shape[1],
            "data/n_features": feature_data.shape[2],
            "data/returns_mean": float(return_data.mean()),
            "data/returns_std": float(return_data.std()),
            "data/returns_min": float(return_data.min()),
            "data/returns_max": float(return_data.max()),
            "data/turnover_mean": float(feature_data[:, :, 1].mean()),
            "data/turnover_std": float(feature_data[:, :, 1].std())
        }
        
        if label_distribution is not None:
            for i, count in enumerate(label_distribution):
                data_stats[f"data/label_class_{i}_count"] = int(count)
                data_stats[f"data/label_class_{i}_percent"] = float(count / label_distribution.sum() * 100)
        
        wandb.log(data_stats)
        print("📊 Data statistics logged to WandB")
    
    def finish(self):
        """Finish WandB run."""
        if self.enabled and self.run:
            wandb.finish()
            print("✅ WandB run finished")


def create_experiment_config(experiment_type: str = "training",
                           model_params: Optional[Dict] = None,
                           **kwargs) -> WandBConfig:
    """
    Create a WandB configuration for different experiment types.
    
    Args:
        experiment_type: Type of experiment ('training', 'backtesting', 'hyperparameter_search')
        model_params: Model parameters to include in experiment name
        **kwargs: Additional configuration parameters
        
    Returns:
        WandB configuration object
    """
    # Generate experiment name based on type and parameters
    if model_params:
        name_parts = [
            experiment_type,
            f"d{model_params.get('d_model', 16)}",
            f"h{model_params.get('n_heads', 8)}",
            f"l{model_params.get('n_layers', 6)}"
        ]
        experiment_name = "_".join(name_parts)
    else:
        experiment_name = experiment_type
    
    # Default tags based on experiment type
    default_tags = {
        "training": ["training", "model-development"],
        "backtesting": ["backtesting", "strategy-evaluation"],
        "hyperparameter_search": ["hyperparameter-tuning", "optimization"]
    }
    
    tags = kwargs.get('tags', []) + default_tags.get(experiment_type, [])
    
    return WandBConfig(
        project_name=kwargs.get('project_name', "quantformer-experiments"),
        entity=kwargs.get('entity'),
        experiment_name=experiment_name,
        tags=tags,
        notes=kwargs.get('notes')
    )


# Example usage functions
def setup_training_experiment(model_config: Dict, 
                            training_config: Dict,
                            data_config: Dict,
                            **wandb_kwargs) -> QuantformerWandBLogger:
    """
    Set up a WandB experiment for model training.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration  
        data_config: Data configuration
        **wandb_kwargs: Additional WandB configuration
        
    Returns:
        Configured WandB logger
    """
    config = create_experiment_config(
        experiment_type="training",
        model_params=model_config,
        **wandb_kwargs
    )
    
    logger = QuantformerWandBLogger(config)
    
    if logger.enabled:
        logger.log_hyperparameters(model_config, training_config, data_config)
    
    return logger


def setup_backtesting_experiment(strategy_config: Dict,
                               **wandb_kwargs) -> QuantformerWandBLogger:
    """
    Set up a WandB experiment for strategy backtesting.
    
    Args:
        strategy_config: Trading strategy configuration
        **wandb_kwargs: Additional WandB configuration
        
    Returns:
        Configured WandB logger
    """
    config = create_experiment_config(
        experiment_type="backtesting",
        **wandb_kwargs
    )
    
    logger = QuantformerWandBLogger(config)
    
    if logger.enabled:
        wandb.config.update({"strategy": strategy_config})
    
    return logger


if __name__ == "__main__":
    # Example usage
    print("WandB utilities for Quantformer experiments")
    
    # Test configuration
    config = create_experiment_config(
        experiment_type="training",
        model_params={"d_model": 32, "n_heads": 8, "n_layers": 4}
    )
    
    print(f"Example config: {config.experiment_name}")
    print(f"Tags: {config.tags}")
