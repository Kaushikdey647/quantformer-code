"""
Quantformer: A Transformer-based Model for Quantitative Trading

This module implements the Quantformer architecture as described in the paper:
"Quantformer: from attention to profit with a quantitative transformer trading strategy"

The model adapts the transformer architecture for numerical time series data
by replacing word embeddings with linear embeddings and removing positional encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class LinearEmbedding(nn.Module):
    """
    Linear embedding layer to replace word embeddings for numerical data.
    
    Args:
        input_dim (int): Input feature dimension (2 for return and turnover rate)
        d_model (int): Model dimension
    """
    
    def __init__(self, input_dim: int = 2, d_model: int = 16):
        super(LinearEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of linear embedding.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Embedded tensor of shape (batch_size, seq_len, d_model)
        """
        return self.linear(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for Quantformer.
    
    Args:
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model: int = 16, n_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            mask: Optional attention mask
            
        Returns:
            Attention output and attention weights
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask
            
        Returns:
            Multi-head attention output
        """
        batch_size = query.size(0)
        
        # Linear transformations and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(attention_output)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Args:
        d_model (int): Model dimension
        d_ff (int): Feed-forward dimension
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model: int = 16, d_ff: int = 64, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of feed-forward network."""
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Single encoder layer with multi-head attention and feed-forward network.
    
    Args:
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        d_ff (int): Feed-forward dimension
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model: int = 16, n_heads: int = 8, d_ff: int = 64, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of encoder layer."""
        # Multi-head attention with residual connection and layer norm
        attn_output = self.multi_head_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class Quantformer(nn.Module):
    """
    Quantformer model for stock prediction.
    
    This model adapts the transformer architecture for quantitative trading by:
    1. Using linear embedding instead of word embedding
    2. Removing positional encoding (time series have inherent order)
    3. Simplifying the decoder for classification tasks
    
    Args:
        input_dim (int): Input feature dimension (default: 2 for return and turnover)
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        n_layers (int): Number of encoder layers
        d_ff (int): Feed-forward dimension
        n_classes (int): Number of output classes (3 or 5 based on paper)
        seq_len (int): Input sequence length (default: 20 as in paper)
        dropout (float): Dropout probability
    """
    
    def __init__(self, 
                 input_dim: int = 2,
                 d_model: int = 16,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 64,
                 n_classes: int = 3,
                 seq_len: int = 20,
                 dropout: float = 0.1):
        super(Quantformer, self).__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.n_classes = n_classes
        
        # Linear embedding layer (replaces word embedding)
        self.embedding = LinearEmbedding(input_dim, d_model)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of Quantformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Class probabilities of shape (batch_size, n_classes)
        """
        # Linear embedding
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        # Global average pooling over sequence dimension
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(x)  # (batch_size, n_classes)
        
        return F.softmax(logits, dim=-1)
    
    def get_attention_weights(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Extract attention weights from a specific layer.
        
        Args:
            x: Input tensor
            layer_idx: Layer index to extract attention from (-1 for last layer)
            
        Returns:
            Attention weights
        """
        x = self.embedding(x)
        
        for i, encoder_layer in enumerate(self.encoder_layers):
            if i == len(self.encoder_layers) + layer_idx:
                # Modify this to return attention weights if needed for analysis
                pass
            x = encoder_layer(x)
            
        return x


class QuantformerTrainer:
    """
    Training utilities for Quantformer model with WandB integration.
    
    Args:
        model: Quantformer model instance
        device: Training device ('cuda' or 'cpu')
        learning_rate: Learning rate for optimizer
        wandb_logger: Optional WandB logger for experiment tracking
    """
    
    def __init__(self, 
                 model: Quantformer, 
                 device: str = 'cuda', 
                 learning_rate: float = 0.001,
                 wandb_logger=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()  # As specified in the paper
        self.wandb_logger = wandb_logger
        self.step_count = 0
        
    def train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
        """
        Single training step.
        
        Args:
            batch_x: Input batch
            batch_y: Target batch
            
        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)
        
        outputs = self.model(batch_x)
        loss = self.criterion(outputs, batch_y)
        
        loss.backward()
        self.optimizer.step()
        
        # Log to WandB if available
        if self.wandb_logger and self.wandb_logger.enabled:
            self.wandb_logger.log_metrics({
                "train/loss": loss.item(),
                "train/step": self.step_count
            }, step=self.step_count)
        
        self.step_count += 1
        return loss.item()
    
    def evaluate(self, dataloader) -> Tuple[float, float]:
        """
        Evaluate model on validation/test data.
        
        Args:
            dataloader: DataLoader for evaluation
            
        Returns:
            Average loss and accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
                # Calculate accuracy (for classification)
                predicted = torch.argmax(outputs, dim=1)
                actual = torch.argmax(batch_y, dim=1)
                correct_predictions += (predicted == actual).sum().item()
                total_samples += batch_y.size(0)
                
                # Store for debugging
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(actual.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        # Log to WandB if available
        if self.wandb_logger and self.wandb_logger.enabled:
            self.wandb_logger.log_metrics({
                "val/loss": avg_loss,
                "val/accuracy": accuracy
            })
        
        # Debug information (only print occasionally)
        if hasattr(self, '_eval_count'):
            self._eval_count += 1
        else:
            self._eval_count = 1
            
        if self._eval_count % 5 == 0:  # Print every 5th evaluation
            pred_dist = np.bincount(all_predictions, minlength=3)
            target_dist = np.bincount(all_targets, minlength=3)
            print(f"  Prediction distribution: {pred_dist}")
            print(f"  Target distribution: {target_dist}")
        
        return avg_loss, accuracy


def create_quantformer_model(config: dict = None) -> Quantformer:
    """
    Factory function to create Quantformer model with default or custom configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured Quantformer model
    """
    default_config = {
        'input_dim': 2,
        'd_model': 16,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 64,
        'n_classes': 3,
        'seq_len': 20,
        'dropout': 0.1
    }
    
    if config:
        default_config.update(config)
    
    return Quantformer(**default_config)


if __name__ == "__main__":
    # Example usage
    model = create_quantformer_model()
    
    # Create dummy data
    batch_size = 32
    seq_len = 20
    input_dim = 2
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
