#!/usr/bin/env python3
"""
Simple test script to verify Quantformer implementation works correctly.
"""

import torch
import numpy as np
from quantformer import (
    create_quantformer_model,
    create_sample_data,
    prepare_training_data,
    QuantformerTradingStrategy,
    TradingConfig
)

def test_model_creation():
    """Test model creation and forward pass."""
    print("Testing model creation...")
    
    model = create_quantformer_model({
        'input_dim': 2,
        'd_model': 16,
        'n_heads': 8,
        'n_layers': 6,
        'n_classes': 3,
        'seq_len': 20
    })
    
    # Test forward pass
    batch_size = 32
    seq_len = 20
    input_dim = 2
    
    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)
    
    assert output.shape == (batch_size, 3), f"Expected shape (32, 3), got {output.shape}"
    assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6), "Output should sum to 1"
    
    print(f"✓ Model created successfully with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"✓ Forward pass works: input {x.shape} -> output {output.shape}")
    return model

def test_data_processing():
    """Test data generation and preprocessing."""
    print("\nTesting data processing...")
    
    # Generate sample data
    feature_data, return_data = create_sample_data(n_stocks=50, n_timesteps=200)
    
    assert feature_data.shape == (200, 50, 2), f"Expected feature shape (200, 50, 2), got {feature_data.shape}"
    assert return_data.shape == (200, 50), f"Expected return shape (200, 50), got {return_data.shape}"
    
    # Prepare training data
    train_loader, val_loader, processor = prepare_training_data(
        feature_data, return_data, seq_len=20, n_classes=3, batch_size=16
    )
    
    # Test a batch
    for batch_x, batch_y in train_loader:
        assert batch_x.shape[1:] == (20, 2), f"Expected batch input shape (*, 20, 2), got {batch_x.shape}"
        assert batch_y.shape[1] == 3, f"Expected batch label shape (*, 3), got {batch_y.shape}"
        break
    
    print(f"✓ Data generation works: {feature_data.shape} features, {return_data.shape} returns")
    print(f"✓ Data preprocessing works: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples")
    return feature_data, return_data, train_loader, val_loader

def test_trading_strategy(model, feature_data, return_data):
    """Test trading strategy implementation."""
    print("\nTesting trading strategy...")
    
    # Create trading config
    config = TradingConfig(
        n_classes=3,
        decision_factor=1,
        phi=0.2,
        transaction_fee=0.003,
        initial_capital=100000
    )
    
    # Initialize strategy
    strategy = QuantformerTradingStrategy(model, config)
    
    # Prepare small backtest data
    n_periods = 10
    n_stocks = 20
    seq_len = 20
    
    # Create backtest data
    backtest_features = np.random.randn(n_periods, n_stocks, seq_len, 2)
    backtest_returns = np.random.randn(n_periods, n_stocks) * 0.02
    
    # Run mini backtest
    results = strategy.backtest(backtest_features, backtest_returns)
    
    # Verify results
    assert 'total_return' in results, "Results should contain total_return"
    assert 'sharpe_ratio' in results, "Results should contain sharpe_ratio"
    assert isinstance(results['final_portfolio_value'], float), "Portfolio value should be float"
    
    print(f"✓ Trading strategy works")
    print(f"✓ Backtest completed: {len(results)} metrics calculated")
    print(f"✓ Final portfolio value: ${results['final_portfolio_value']:,.2f}")
    return results

def test_training_step(model, train_loader):
    """Test a single training step."""
    print("\nTesting training step...")
    
    from quantformer import QuantformerTrainer
    
    trainer = QuantformerTrainer(model, device='cpu', learning_rate=0.001)
    
    # Get a batch and train
    for batch_x, batch_y in train_loader:
        initial_loss = trainer.train_step(batch_x, batch_y)
        break
    
    assert isinstance(initial_loss, float), "Loss should be a float"
    assert initial_loss > 0, "Loss should be positive"
    
    print(f"✓ Training step works: loss = {initial_loss:.4f}")
    return initial_loss

def main():
    """Run all tests."""
    print("=" * 60)
    print("QUANTFORMER IMPLEMENTATION TEST")
    print("=" * 60)
    
    try:
        # Test model
        model = test_model_creation()
        
        # Test data processing
        feature_data, return_data, train_loader, val_loader = test_data_processing()
        
        # Test training
        loss = test_training_step(model, train_loader)
        
        # Test trading strategy
        results = test_trading_strategy(model, feature_data, return_data)
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe Quantformer implementation is working correctly.")
        print("You can now use the Jupyter notebook for a full demonstration.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        raise

if __name__ == "__main__":
    main()
