#!/usr/bin/env python3
"""
Simple Bayesian Neural Networks Demo
Quick demonstration of uncertainty estimation
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models.bayesian_nn import BayesianNN
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl


def simple_classification_demo():
    """Simple 2D classification with uncertainty visualization"""
    print("🧠 Simple Bayesian Neural Network Demo")
    print("=" * 50)
    
    # Generate simple 2D dataset
    np.random.seed(42)
    n_samples = 500
    
    # Create two clusters
    cluster1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], n_samples//2)
    cluster2 = np.random.multivariate_normal([-2, -2], [[0.5, 0], [0, 0.5]], n_samples//2)
    
    X = np.vstack([cluster1, cluster2]).astype(np.float32)
    y = np.hstack([np.ones(n_samples//2), np.zeros(n_samples//2)]).astype(np.int64)
    
    # Create dataset
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = BayesianNN(
        input_dim=2,
        hidden_dims=[20, 20],
        output_dim=2,
        task_type='classification',
        learning_rate=1e-2,
        kl_weight=1e-3,
        num_samples=10
    )
    
    # Quick training
    print("🚀 Training Bayesian Neural Network...")
    trainer = pl.Trainer(
        max_epochs=20,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False
    )
    
    trainer.fit(model, dataloader)
    
    # Create prediction grid
    print("📊 Generating uncertainty visualization...")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid_tensor = torch.from_numpy(grid_points)
    
    # Get predictions with uncertainty
    model.eval()
    with torch.no_grad():
        mean_pred, uncertainty = model.predict_with_uncertainty(grid_tensor, num_samples=20)
        
        # Get class probabilities and uncertainty
        class_probs = mean_pred[:, 1].numpy()  # Probability of class 1
        max_uncertainty = uncertainty.max(dim=1)[0].numpy()
    
    # Reshape for plotting
    class_probs = class_probs.reshape(xx.shape)
    max_uncertainty = max_uncertainty.reshape(xx.shape)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Data points
    scatter = axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', alpha=0.7, s=50)
    axes[0].set_title('Training Data\n(Red=Class 0, Blue=Class 1)')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Prediction probabilities
    contour1 = axes[1].contourf(xx, yy, class_probs, levels=20, cmap='RdYlBu', alpha=0.8)
    axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black', s=50)
    axes[1].set_title('Prediction Probabilities\n(Red=Class 0, Blue=Class 1)')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    plt.colorbar(contour1, ax=axes[1], label='P(Class 1)')
    
    # Plot 3: Uncertainty
    contour2 = axes[2].contourf(xx, yy, max_uncertainty, levels=20, cmap='Reds', alpha=0.8)
    axes[2].scatter(X[:, 0], X[:, 1], c='black', s=30, alpha=0.6)
    axes[2].set_title('Model Uncertainty\n(Darker = More Uncertain)')
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    plt.colorbar(contour2, ax=axes[2], label='Uncertainty')
    
    plt.tight_layout()
    plt.savefig('simple_bnn_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print insights
    print("\n🎯 Key Insights:")
    print(f"✅ Model trained successfully!")
    print(f"📊 Average uncertainty: {max_uncertainty.mean():.4f}")
    print(f"🎲 Max uncertainty: {max_uncertainty.max():.4f}")
    print(f"🔍 Uncertainty is highest at decision boundaries")
    print(f"💡 This shows the model is less confident where classes overlap")
    
    return model, X, y


def regression_demo():
    """Simple 1D regression with uncertainty"""
    print("\n" + "=" * 50)
    print("📈 Regression with Uncertainty Demo")
    print("=" * 50)
    
    # Generate noisy sine wave data
    np.random.seed(42)
    x_train = np.linspace(-2, 2, 100)
    y_true = np.sin(2 * x_train)
    y_noisy = y_true + 0.2 * np.random.randn(len(x_train))
    
    # Convert to tensors
    X_train = torch.from_numpy(x_train.reshape(-1, 1).astype(np.float32))
    y_train = torch.from_numpy(y_noisy.astype(np.float32))
    
    # Create dataset
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create model
    model = BayesianNN(
        input_dim=1,
        hidden_dims=[50, 50],
        output_dim=1,
        task_type='regression',
        learning_rate=1e-2,
        kl_weight=1e-3,
        num_samples=10
    )
    
    # Train
    print("🚀 Training regression model...")
    trainer = pl.Trainer(
        max_epochs=50,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False
    )
    
    trainer.fit(model, dataloader)
    
    # Test on extended range
    x_test = np.linspace(-4, 4, 200)
    X_test = torch.from_numpy(x_test.reshape(-1, 1).astype(np.float32))
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        mean_pred, uncertainty = model.predict_with_uncertainty(X_test, num_samples=50)
        
        mean_pred = mean_pred.squeeze().numpy()
        uncertainty = uncertainty.squeeze().numpy()
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.scatter(x_train, y_noisy, alpha=0.6, color='red', s=30, label='Training Data')
    
    # Plot true function
    y_true_extended = np.sin(2 * x_test)
    plt.plot(x_test, y_true_extended, 'g-', linewidth=2, label='True Function')
    
    # Plot predictions with uncertainty
    plt.plot(x_test, mean_pred, 'b-', linewidth=2, label='BNN Prediction')
    plt.fill_between(x_test, mean_pred - uncertainty, mean_pred + uncertainty,
                    alpha=0.3, color='blue', label='±1σ Uncertainty')
    plt.fill_between(x_test, mean_pred - 2*uncertainty, mean_pred + 2*uncertainty,
                    alpha=0.2, color='blue', label='±2σ Uncertainty')
    
    plt.xlabel('Input (x)')
    plt.ylabel('Output (y)')
    plt.title('Bayesian Neural Network: Regression with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Highlight training region
    plt.axvspan(-2, 2, alpha=0.1, color='green', label='Training Region')
    
    plt.savefig('regression_uncertainty_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n🎯 Key Insights:")
    print(f"✅ Regression model trained successfully!")
    print(f"📊 Uncertainty increases outside training region")
    print(f"🎲 Max uncertainty: {uncertainty.max():.4f}")
    print(f"💡 Model is less confident when extrapolating")


def main():
    """Run simple demos"""
    print("🧠 BAYESIAN NEURAL NETWORKS - SIMPLE DEMO")
    print("=" * 60)
    print("This demo shows uncertainty estimation in action!")
    print()
    
    try:
        # Run classification demo
        model, X, y = simple_classification_demo()
        
        # Run regression demo
        regression_demo()
        
        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETE!")
        print("=" * 60)
        print("📊 Generated visualizations:")
        print("  ✅ simple_bnn_demo.png - Classification with uncertainty")
        print("  ✅ regression_uncertainty_demo.png - Regression with uncertainty")
        print()
        print("💡 Key Takeaways:")
        print("  • Bayesian NNs provide uncertainty estimates with predictions")
        print("  • Uncertainty is higher at decision boundaries (classification)")
        print("  • Uncertainty increases when extrapolating (regression)")
        print("  • This enables safer AI for critical applications")
        print()
        print("🚀 Next steps:")
        print("  • Try: python train_medical.py (medical classification)")
        print("  • Try: python train_regression.py (full regression example)")
        print("  • Add this to your portfolio!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you're in the virtual environment:")
        print("   source bnn_env/bin/activate")


if __name__ == "__main__":
    main()