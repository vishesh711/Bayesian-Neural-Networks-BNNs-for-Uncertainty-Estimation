import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from models.bayesian_nn import BayesianNN
import seaborn as sns


def generate_noisy_data(n_samples=1000, noise_std=0.3):
    """Generate synthetic regression data with varying noise"""
    np.random.seed(42)
    
    # Generate x values
    x = np.linspace(-3, 3, n_samples)
    
    # True function: sinusoidal with varying amplitude
    y_true = np.sin(2 * x) * np.exp(-0.1 * x**2) + 0.5 * np.cos(3 * x)
    
    # Add heteroscedastic noise (more noise at edges)
    noise_scale = noise_std * (1 + 0.5 * np.abs(x))
    noise = np.random.normal(0, noise_scale)
    y_noisy = y_true + noise
    
    return x.astype(np.float32), y_noisy.astype(np.float32), y_true.astype(np.float32)


class RegressionDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64):
        super().__init__()
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        # Generate data
        x, y_noisy, y_true = generate_noisy_data(n_samples=800)
        
        # Split into train/val
        split_idx = int(0.8 * len(x))
        
        x_train, y_train = x[:split_idx], y_noisy[:split_idx]
        x_val, y_val = x[split_idx:], y_noisy[split_idx:]
        
        # Convert to tensors and reshape
        x_train = torch.from_numpy(x_train).unsqueeze(1)
        y_train = torch.from_numpy(y_train)
        x_val = torch.from_numpy(x_val).unsqueeze(1)
        y_val = torch.from_numpy(y_val)
        
        self.train_dataset = TensorDataset(x_train, y_train)
        self.val_dataset = TensorDataset(x_val, y_val)
        
        # Store for visualization
        self.x_all, self.y_all, self.y_true = generate_noisy_data(n_samples=1000)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


def visualize_regression_uncertainty(model, data_module):
    """Visualize regression predictions with uncertainty bands"""
    model.eval()
    
    # Generate test points
    x_test = np.linspace(-4, 4, 200)
    x_test_tensor = torch.from_numpy(x_test.astype(np.float32)).unsqueeze(1)
    
    # Get predictions with uncertainty
    mean_pred, uncertainty = model.predict_with_uncertainty(x_test_tensor, num_samples=100)
    
    mean_pred = mean_pred.squeeze().numpy()
    uncertainty = uncertainty.squeeze().numpy()
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot training data
    plt.scatter(data_module.x_all, data_module.y_all, alpha=0.6, s=20, 
               color='lightcoral', label='Noisy Training Data')
    
    # Plot true function
    plt.plot(data_module.x_all, data_module.y_true, 'g-', linewidth=2, 
            label='True Function')
    
    # Plot predictions with uncertainty
    plt.plot(x_test, mean_pred, 'b-', linewidth=2, label='BNN Prediction')
    
    # Plot uncertainty bands (1 and 2 standard deviations)
    plt.fill_between(x_test, mean_pred - uncertainty, mean_pred + uncertainty,
                    alpha=0.3, color='blue', label='±1σ Uncertainty')
    plt.fill_between(x_test, mean_pred - 2*uncertainty, mean_pred + 2*uncertainty,
                    alpha=0.2, color='blue', label='±2σ Uncertainty')
    
    plt.xlabel('Input (x)')
    plt.ylabel('Output (y)')
    plt.title('Bayesian Neural Network: Regression with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('regression_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot uncertainty vs distance from training data
    plt.figure(figsize=(10, 6))
    plt.plot(x_test, uncertainty, 'r-', linewidth=2)
    plt.xlabel('Input (x)')
    plt.ylabel('Prediction Uncertainty (σ)')
    plt.title('Model Uncertainty: Higher Outside Training Region')
    plt.grid(True, alpha=0.3)
    
    # Mark training data region
    train_min, train_max = data_module.x_all.min(), data_module.x_all.max()
    plt.axvspan(train_min, train_max, alpha=0.2, color='green', 
               label='Training Data Region')
    plt.legend()
    plt.savefig('uncertainty_vs_input.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_with_standard_nn(data_module):
    """Compare Bayesian NN with standard NN"""
    
    # Standard NN (deterministic)
    class StandardNN(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(1, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 1)
            )
        
        def forward(self, x):
            return self.layers(x)
        
        def training_step(self, batch, batch_idx):
            x, y = batch
            pred = self.forward(x).squeeze()
            loss = F.mse_loss(pred, y)
            return loss
        
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    # Train standard NN
    standard_model = StandardNN()
    trainer = pl.Trainer(max_epochs=100, enable_progress_bar=False, 
                        enable_model_summary=False, logger=False)
    trainer.fit(standard_model, data_module)
    
    # Generate predictions
    x_test = np.linspace(-4, 4, 200)
    x_test_tensor = torch.from_numpy(x_test.astype(np.float32)).unsqueeze(1)
    
    standard_model.eval()
    with torch.no_grad():
        standard_pred = standard_model(x_test_tensor).squeeze().numpy()
    
    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    # Standard NN
    plt.subplot(1, 2, 1)
    plt.scatter(data_module.x_all, data_module.y_all, alpha=0.6, s=20, 
               color='lightcoral', label='Training Data')
    plt.plot(data_module.x_all, data_module.y_true, 'g-', linewidth=2, 
            label='True Function')
    plt.plot(x_test, standard_pred, 'r-', linewidth=2, label='Standard NN')
    plt.xlabel('Input (x)')
    plt.ylabel('Output (y)')
    plt.title('Standard Neural Network\n(No Uncertainty)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bayesian NN (placeholder - would need trained model)
    plt.subplot(1, 2, 2)
    plt.scatter(data_module.x_all, data_module.y_all, alpha=0.6, s=20, 
               color='lightcoral', label='Training Data')
    plt.plot(data_module.x_all, data_module.y_true, 'g-', linewidth=2, 
            label='True Function')
    plt.xlabel('Input (x)')
    plt.ylabel('Output (y)')
    plt.title('Bayesian Neural Network\n(With Uncertainty Bands)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nn_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # Set random seeds
    pl.seed_everything(42)
    
    # Initialize data module
    data_module = RegressionDataModule(batch_size=64)
    data_module.setup()
    
    # Initialize Bayesian model
    model = BayesianNN(
        input_dim=1,
        hidden_dims=[100, 100],
        output_dim=1,
        task_type='regression',
        learning_rate=1e-3,
        kl_weight=1e-3,
        num_samples=20
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        filename='best-regression-bnn'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        mode='min'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator='auto',
        log_every_n_steps=20
    )
    
    # Train model
    print("🚀 Training Bayesian Neural Network for Regression...")
    trainer.fit(model, data_module)
    
    # Load best model
    best_model = BayesianNN.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        input_dim=1,
        hidden_dims=[100, 100],
        output_dim=1,
        task_type='regression'
    )
    
    # Visualize results
    print("📊 Generating uncertainty visualizations...")
    visualize_regression_uncertainty(best_model, data_module)
    compare_with_standard_nn(data_module)
    
    print("✅ Regression Bayesian NN training complete!")
    print("📈 Check 'regression_uncertainty.png' and 'uncertainty_vs_input.png'")


if __name__ == "__main__":
    main()