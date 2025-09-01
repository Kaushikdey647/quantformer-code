#!/usr/bin/env python3
"""
Example script demonstrating WandB integration with Quantformer.

This script shows how to:
1. Set up WandB experiments
2. Track training metrics
3. Log model artifacts
4. Compare different configurations
5. Track trading strategy performance
"""

import torch
import numpy as np
from quantformer import (
    create_quantformer_model,
    create_sample_data,
    prepare_training_data,
    QuantformerTrainer,
    QuantformerTradingStrategy,
    TradingConfig,
    setup_training_experiment,
    setup_backtesting_experiment
)


def run_training_experiment(config_name: str, model_config: dict, training_config: dict):
    """
    Run a single training experiment with WandB tracking.
    
    Args:
        config_name: Name for this configuration
        model_config: Model architecture configuration
        training_config: Training hyperparameters
    """
    print(f"\n{'='*50}")
    print(f"RUNNING EXPERIMENT: {config_name}")
    print(f"{'='*50}")
    
    # Generate data
    feature_data, return_data = create_sample_data(
        n_stocks=100, n_timesteps=500, random_seed=42
    )
    
    # Prepare training data
    train_loader, val_loader, _ = prepare_training_data(
        feature_data, return_data, 
        seq_len=model_config['seq_len'],
        n_classes=model_config['n_classes'],
        batch_size=training_config['batch_size']
    )
    
    # Data configuration
    data_config = {
        "n_stocks": 100,
        "n_timesteps": 500,
        "seq_len": model_config['seq_len'],
        "n_classes": model_config['n_classes']
    }
    
    # Set up WandB experiment
    wandb_logger = setup_training_experiment(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        notes=f"Experiment: {config_name}",
        tags=[config_name, "hyperparameter_search"]
    )
    
    if not wandb_logger.enabled:
        print("⚠️ WandB not available, running without logging")
    
    # Create and train model
    model = create_quantformer_model(model_config)
    trainer = QuantformerTrainer(
        model=model,
        device='cpu',
        learning_rate=training_config['learning_rate'],
        wandb_logger=wandb_logger
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(training_config['epochs']):
        # Training
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            loss = trainer.train_step(batch_x, batch_y)
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        val_loss, val_acc = trainer.evaluate(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{training_config['epochs']}: "
                  f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Log final results
    if wandb_logger.enabled:
        wandb_logger.log_training_curves(train_losses, val_losses, val_accuracies)
        wandb_logger.log_model_artifact(
            model, 
            name=f"model_{config_name}",
            metadata={"config_name": config_name, "final_accuracy": val_accuracies[-1]}
        )
        wandb_logger.finish()
    
    print(f"✅ {config_name} completed - Final accuracy: {val_accuracies[-1]:.4f}")
    return model, val_accuracies[-1]


def run_backtesting_experiment(model, config_name: str):
    """
    Run backtesting experiment with WandB tracking.
    
    Args:
        model: Trained model
        config_name: Configuration name for tracking
    """
    print(f"\n{'='*30}")
    print(f"BACKTESTING: {config_name}")
    print(f"{'='*30}")
    
    # Generate backtest data
    feature_data, return_data = create_sample_data(
        n_stocks=50, n_timesteps=200, random_seed=123
    )
    
    # Prepare backtest data
    n_periods = 50
    features_for_backtest = np.zeros((n_periods, 50, 20, 2))
    returns_for_backtest = np.zeros((n_periods, 50))
    
    for t in range(n_periods):
        for stock in range(50):
            features_for_backtest[t, stock] = feature_data[t:t+20, stock]
        returns_for_backtest[t] = return_data[t+20]
    
    # Trading configuration
    trading_config = TradingConfig(
        n_classes=3,
        decision_factor=1,
        phi=0.2,
        transaction_fee=0.003,
        initial_capital=100000
    )
    
    # Set up WandB for backtesting
    strategy_config_dict = {
        "n_classes": trading_config.n_classes,
        "decision_factor": trading_config.decision_factor,
        "phi": trading_config.phi,
        "transaction_fee": trading_config.transaction_fee,
        "initial_capital": trading_config.initial_capital
    }
    
    backtest_logger = setup_backtesting_experiment(
        strategy_config=strategy_config_dict,
        notes=f"Backtesting for {config_name}",
        tags=[config_name, "backtesting"]
    )
    
    # Run backtest
    strategy = QuantformerTradingStrategy(
        model, trading_config, wandb_logger=backtest_logger
    )
    
    results = strategy.backtest(features_for_backtest, returns_for_backtest)
    
    if backtest_logger.enabled:
        backtest_logger.finish()
    
    print(f"✅ Backtesting completed - Final return: {results.get('total_return', 0):.2%}")
    return results


def main():
    """Run comprehensive WandB experiment suite."""
    print("🚀 QUANTFORMER WANDB EXPERIMENT SUITE")
    print("=====================================")
    
    # Define different configurations to test
    experiments = {
        "small_model": {
            "model_config": {
                "input_dim": 2,
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 2,
                "d_ff": 32,
                "n_classes": 3,
                "seq_len": 20,
                "dropout": 0.1
            },
            "training_config": {
                "epochs": 15,
                "learning_rate": 0.001,
                "batch_size": 32
            }
        },
        "medium_model": {
            "model_config": {
                "input_dim": 2,
                "d_model": 32,
                "n_heads": 8,
                "n_layers": 4,
                "d_ff": 64,
                "n_classes": 3,
                "seq_len": 20,
                "dropout": 0.2
            },
            "training_config": {
                "epochs": 15,
                "learning_rate": 0.0005,
                "batch_size": 64
            }
        },
        "large_model": {
            "model_config": {
                "input_dim": 2,
                "d_model": 64,
                "n_heads": 8,
                "n_layers": 6,
                "d_ff": 128,
                "n_classes": 3,
                "seq_len": 20,
                "dropout": 0.3
            },
            "training_config": {
                "epochs": 15,
                "learning_rate": 0.0003,
                "batch_size": 32
            }
        }
    }
    
    # Run training experiments
    trained_models = {}
    
    for config_name, config in experiments.items():
        model, accuracy = run_training_experiment(
            config_name, 
            config["model_config"], 
            config["training_config"]
        )
        trained_models[config_name] = (model, accuracy)
    
    # Find best model
    best_config = max(trained_models.items(), key=lambda x: x[1][1])
    print(f"\n🏆 Best model: {best_config[0]} (Accuracy: {best_config[1][1]:.4f})")
    
    # Run backtesting on best model
    run_backtesting_experiment(best_config[1][0], best_config[0])
    
    print(f"\n🎉 All experiments completed!")
    print("📊 Check your WandB dashboard for detailed comparisons:")
    print("   - Training curves across different configurations")
    print("   - Model performance metrics")
    print("   - Trading strategy results")
    print("   - Hyperparameter importance")


if __name__ == "__main__":
    main()
