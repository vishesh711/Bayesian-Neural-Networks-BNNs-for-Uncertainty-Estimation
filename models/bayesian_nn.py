import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import numpy as np
from typing import Tuple, Optional


class BayesianLinear(nn.Module):
    """Bayesian Linear Layer with weight uncertainty"""
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters (mean and log std)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_log_sigma = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 2)
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_log_sigma = nn.Parameter(torch.randn(out_features) * 0.1 - 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample weights from posterior
        weight_sigma = torch.exp(self.weight_log_sigma)
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        
        # Sample bias from posterior
        bias_sigma = torch.exp(self.bias_log_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior"""
        # KL for weights
        weight_sigma = torch.exp(self.weight_log_sigma)
        weight_kl = 0.5 * torch.sum(
            (self.weight_mu**2 + weight_sigma**2) / (self.prior_std**2) 
            - 2 * self.weight_log_sigma + 2 * np.log(self.prior_std) - 1
        )
        
        # KL for bias
        bias_sigma = torch.exp(self.bias_log_sigma)
        bias_kl = 0.5 * torch.sum(
            (self.bias_mu**2 + bias_sigma**2) / (self.prior_std**2)
            - 2 * self.bias_log_sigma + 2 * np.log(self.prior_std) - 1
        )
        
        return weight_kl + bias_kl


class BayesianNN(pl.LightningModule):
    """Bayesian Neural Network with uncertainty estimation"""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list, 
        output_dim: int,
        task_type: str = 'classification',
        learning_rate: float = 1e-3,
        kl_weight: float = 1e-3,
        num_samples: int = 10
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.task_type = task_type
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.num_samples = num_samples
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
            
        layers.append(BayesianLinear(prev_dim, output_dim))
        self.layers = nn.ModuleList(layers)
        
        # Track metrics
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation on output layer
                x = F.relu(x)
        return x
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        num_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with uncertainty estimates"""
        if num_samples is None:
            num_samples = self.num_samples
            
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x)
                if self.task_type == 'classification':
                    pred = F.softmax(pred, dim=-1)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [num_samples, batch_size, output_dim]
        
        # Compute mean and uncertainty
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty
    
    def compute_kl_loss(self) -> torch.Tensor:
        """Compute total KL divergence loss"""
        kl_loss = 0
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                kl_loss += layer.kl_divergence()
        return kl_loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Flatten input if needed (for MNIST)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # Forward pass
        logits = self.forward(x)
        
        # Compute likelihood loss
        if self.task_type == 'classification':
            likelihood_loss = F.cross_entropy(logits, y)
        else:  # regression
            likelihood_loss = F.mse_loss(logits.squeeze(), y)
        
        # Compute KL loss
        kl_loss = self.compute_kl_loss()
        
        # Total loss (ELBO)
        total_loss = likelihood_loss + self.kl_weight * kl_loss
        
        # Logging
        self.log('train_likelihood', likelihood_loss, prog_bar=True)
        self.log('train_kl', kl_loss, prog_bar=True)
        self.log('train_loss', total_loss, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Flatten input if needed (for MNIST)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # Get predictions with uncertainty
        mean_pred, uncertainty = self.predict_with_uncertainty(x)
        
        # Compute loss
        if self.task_type == 'classification':
            val_loss = F.cross_entropy(mean_pred, y)
            # Compute accuracy
            pred_labels = mean_pred.argmax(dim=-1)
            accuracy = (pred_labels == y).float().mean()
            self.log('val_accuracy', accuracy, prog_bar=True)
        else:  # regression
            val_loss = F.mse_loss(mean_pred.squeeze(), y)
        
        # Log average uncertainty
        avg_uncertainty = uncertainty.mean()
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_uncertainty', avg_uncertainty, prog_bar=True)
        
        return val_loss
    
    def test_step(self, batch, batch_idx):
        # Same as validation step
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)