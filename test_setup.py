#!/usr/bin/env python3
"""
Test script to verify Bayesian Neural Networks setup
Quick validation that all components work correctly
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from models.bayesian_nn import BayesianNN, BayesianLinear
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset


def test_bayesian_linear():
    """Test BayesianLinear layer"""
    print("🧪 Testing BayesianLinear layer...")
    
    layer = BayesianLinear(10, 5)
    x = torch.randn(32, 10)
    
    # Test forward pass
    output = layer(x)
    assert output.shape == (32, 5), f"Expected (32, 5), got {output.shape}"
    
    # Test KL divergence
    kl = layer.kl_divergence()
    assert kl.item() > 0, "KL divergence should be positive"
    
    # Test multiple forward passes give different outputs (stochastic)
    output1 = layer(x)
    output2 = layer(x)
    assert not torch.allclose(output1, output2), "Outputs should be different (stochastic)"
    
    print("✅ BayesianLinear layer test passed!")


def test_bayesian_nn():
    """Test BayesianNN model"""
    print("🧪 Testing BayesianNN model...")
    
    model = BayesianNN(
        input_dim=10,
        hidden_dims=[20, 15],
        output_dim=3,
        task_type='classification'
    )
    
    x = torch.randn(16, 10)
    
    # Test forward pass
    output = model(x)
    assert output.shape == (16, 3), f"Expected (16, 3), got {output.shape}"
    
    # Test uncertainty prediction
    mean_pred, uncertainty = model.predict_with_uncertainty(x, num_samples=5)
    assert mean_pred.shape == (16, 3), f"Expected (16, 3), got {mean_pred.shape}"
    assert uncertainty.shape == (16, 3), f"Expected (16, 3), got {uncertainty.shape}"
    
    # Test KL loss
    kl_loss = model.compute_kl_loss()
    assert kl_loss.item() > 0, "KL loss should be positive"
    
    print("✅ BayesianNN model test passed!")


def test_training_step():
    """Test training functionality"""
    print("🧪 Testing training step...")
    
    # Create simple dataset
    X = torch.randn(100, 5)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16)
    
    model = BayesianNN(
        input_dim=5,
        hidden_dims=[10],
        output_dim=2,
        task_type='classification'
    )
    
    # Test training step
    batch = next(iter(dataloader))
    loss = model.training_step(batch, 0)
    
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.item() > 0, "Loss should be positive"
    
    print("✅ Training step test passed!")


def test_uncertainty_behavior():
    """Test that uncertainty behaves as expected"""
    print("🧪 Testing uncertainty behavior...")
    
    # Create model
    model = BayesianNN(
        input_dim=2,
        hidden_dims=[10],
        output_dim=1,
        task_type='regression'
    )
    
    # Test on same input multiple times
    x = torch.tensor([[1.0, 2.0]])
    
    predictions = []
    for _ in range(20):
        pred = model(x)
        predictions.append(pred.item())
    
    # Check that predictions vary (uncertainty)
    predictions = np.array(predictions)
    std_dev = np.std(predictions)
    
    assert std_dev > 0, f"Standard deviation should be > 0, got {std_dev}"
    print(f"✅ Prediction std dev: {std_dev:.4f} (good - shows uncertainty!)")


def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        import torchvision
        import pytorch_lightning as pl
        import pyro
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd
        from sklearn.datasets import make_classification
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_visualization():
    """Test basic visualization functionality"""
    print("🧪 Testing visualization...")
    
    try:
        # Test matplotlib
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title("Test Plot")
        plt.close(fig)
        
        print("✅ Visualization test passed!")
        return True
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False


def run_quick_demo():
    """Run a very quick demo to show the system works"""
    print("🚀 Running quick demo...")
    
    # Generate tiny dataset
    X = torch.randn(50, 3)
    y = (X.sum(dim=1) > 0).long()
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=10)
    
    # Create model
    model = BayesianNN(
        input_dim=3,
        hidden_dims=[5],
        output_dim=2,
        task_type='classification'
    )
    
    # Quick training
    trainer = pl.Trainer(
        max_epochs=3,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False
    )
    
    try:
        trainer.fit(model, train_loader)
        
        # Test predictions
        test_x = torch.randn(5, 3)
        mean_pred, uncertainty = model.predict_with_uncertainty(test_x, num_samples=5)
        
        print(f"✅ Quick demo successful!")
        print(f"   Mean prediction shape: {mean_pred.shape}")
        print(f"   Uncertainty shape: {uncertainty.shape}")
        print(f"   Sample uncertainty: {uncertainty[0].max().item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Quick demo failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧠 BAYESIAN NEURAL NETWORKS - SETUP TEST")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("BayesianLinear Layer", test_bayesian_linear),
        ("BayesianNN Model", test_bayesian_nn),
        ("Training Step", test_training_step),
        ("Uncertainty Behavior", test_uncertainty_behavior),
        ("Visualization", test_visualization),
        ("Quick Demo", run_quick_demo)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your setup is ready.")
        print("\n🚀 Next steps:")
        print("   • Run: python demo.py")
        print("   • Or run individual examples:")
        print("     - python train_mnist.py")
        print("     - python train_regression.py") 
        print("     - python train_medical.py")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the errors above.")
        print("💡 Try: pip install -r requirements.txt")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)