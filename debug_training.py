#!/usr/bin/env python3
"""
Debug script to test the fixed Quantformer training.
"""

import torch
import numpy as np
from quantformer import (
    create_quantformer_model,
    create_sample_data,
    prepare_training_data,
    QuantformerTrainer
)

def debug_data_generation():
    """Debug the data generation and labeling process."""
    print("=== DEBUGGING DATA GENERATION ===")
    
    # Create sample data
    feature_data, return_data = create_sample_data(n_stocks=50, n_timesteps=200)
    
    print(f"Feature data shape: {feature_data.shape}")
    print(f"Return data shape: {return_data.shape}")
    
    # Check data statistics
    print(f"Returns - Mean: {return_data.mean():.6f}, Std: {return_data.std():.6f}")
    print(f"Returns - Min: {return_data.min():.6f}, Max: {return_data.max():.6f}")
    
    # Prepare training data
    train_loader, val_loader, processor = prepare_training_data(
        feature_data, return_data, seq_len=20, n_classes=3, batch_size=32
    )
    
    # Examine a few batches
    print(f"\n=== EXAMINING BATCHES ===")
    for i, (batch_x, batch_y) in enumerate(train_loader):
        if i >= 3:  # Only check first 3 batches
            break
            
        print(f"\nBatch {i+1}:")
        print(f"  Input shape: {batch_x.shape}")
        print(f"  Label shape: {batch_y.shape}")
        print(f"  Label distribution: {batch_y.sum(dim=0).numpy()}")
        
        # Check if all samples have exactly one label
        label_sums = batch_y.sum(dim=1)
        print(f"  Samples with exactly 1 label: {(label_sums == 1).sum().item()}/{len(label_sums)}")
        print(f"  Label sums: {label_sums.unique()}")
    
    return train_loader, val_loader

def debug_model_training(train_loader, val_loader):
    """Debug the model training process."""
    print(f"\n=== DEBUGGING MODEL TRAINING ===")
    
    # Create model
    model = create_quantformer_model({
        'input_dim': 2,
        'd_model': 16,
        'n_heads': 8,
        'n_layers': 3,  # Smaller model for debugging
        'n_classes': 3,
        'seq_len': 20,
        'dropout': 0.1
    })
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    for batch_x, batch_y in train_loader:
        print(f"\nTesting forward pass:")
        print(f"  Input range: [{batch_x.min():.4f}, {batch_x.max():.4f}]")
        
        with torch.no_grad():
            output = model(batch_x)
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
            print(f"  Output sums: {output.sum(dim=1)[:5]}")  # Should be ~1.0
            
            # Check if model is just predicting one class
            predicted_classes = torch.argmax(output, dim=1)
            unique_preds = predicted_classes.unique()
            print(f"  Unique predictions: {unique_preds.numpy()}")
            
        break
    
    # Initialize trainer
    trainer = QuantformerTrainer(model, device='cpu', learning_rate=0.01)  # Higher LR for debugging
    
    # Train for a few epochs
    print(f"\n=== TRAINING FOR 5 EPOCHS ===")
    for epoch in range(5):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_x, batch_y in train_loader:
            loss = trainer.train_step(batch_x, batch_y)
            epoch_loss += loss
            batch_count += 1
            
            if batch_count >= 5:  # Only train on first 5 batches for debugging
                break
        
        avg_loss = epoch_loss / batch_count
        val_loss, val_acc = trainer.evaluate(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}, Val Acc = {val_acc:.4f}")
        
        # Check if loss is decreasing
        if epoch == 0:
            initial_loss = avg_loss
        elif epoch == 4:
            if avg_loss >= initial_loss * 0.9:  # Loss should decrease by at least 10%
                print(f"  WARNING: Loss not decreasing significantly!")

def main():
    """Run debugging tests."""
    print("QUANTFORMER TRAINING DEBUG")
    print("=" * 50)
    
    # Set seeds for reproducibility
    np.random.seed(123)  # Different seed to test robustness
    torch.manual_seed(123)
    
    try:
        # Debug data
        train_loader, val_loader = debug_data_generation()
        
        # Debug training
        debug_model_training(train_loader, val_loader)
        
        print(f"\n" + "=" * 50)
        print("✅ DEBUGGING COMPLETED")
        print("If you still see perfect accuracy, the issue might be:")
        print("1. Model architecture too simple for the data")
        print("2. Learning rate too high causing instability")
        print("3. Data still too predictable")
        print("Try running the notebook again with these fixes!")
        
    except Exception as e:
        print(f"\n❌ DEBUG FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
