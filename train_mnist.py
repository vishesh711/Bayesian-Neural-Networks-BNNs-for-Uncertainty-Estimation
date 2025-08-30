import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from models.bayesian_nn import BayesianNN
import seaborn as sns


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 128):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def prepare_data(self):
        torchvision.datasets.MNIST(root='./data', train=True, download=True)
        torchvision.datasets.MNIST(root='./data', train=False, download=True)
    
    def setup(self, stage=None):
        self.train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, transform=self.transform
        )
        self.val_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, transform=self.transform
        )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


def visualize_uncertainty(model, dataloader, num_samples=5):
    """Visualize predictions with uncertainty for MNIST"""
    model.eval()
    
    # Get a batch of test data
    batch = next(iter(dataloader))
    images, labels = batch
    images_flat = images.view(images.size(0), -1)
    
    # Select first few samples
    images_subset = images_flat[:num_samples]
    labels_subset = labels[:num_samples]
    
    # Get predictions with uncertainty
    mean_pred, uncertainty = model.predict_with_uncertainty(images_subset, num_samples=50)
    
    # Create visualization
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # Original image
        img = images[i].squeeze().numpy()
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'True: {labels_subset[i].item()}')
        axes[0, i].axis('off')
        
        # Prediction uncertainty
        pred_probs = mean_pred[i].numpy()
        pred_uncertainty = uncertainty[i].numpy()
        
        x_pos = np.arange(10)
        axes[1, i].bar(x_pos, pred_probs, yerr=pred_uncertainty, 
                      capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
        axes[1, i].set_xlabel('Digit Class')
        axes[1, i].set_ylabel('Probability')
        axes[1, i].set_title(f'Pred: {pred_probs.argmax()}, Max Unc: {pred_uncertainty.max():.3f}')
        axes[1, i].set_xticks(x_pos)
        axes[1, i].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('mnist_uncertainty_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_uncertainty_vs_accuracy(model, dataloader):
    """Analyze relationship between uncertainty and prediction accuracy"""
    model.eval()
    
    all_uncertainties = []
    all_correct = []
    
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images_flat = images.view(images.size(0), -1)
            
            mean_pred, uncertainty = model.predict_with_uncertainty(images_flat, num_samples=20)
            
            # Get predictions and correctness
            pred_labels = mean_pred.argmax(dim=-1)
            correct = (pred_labels == labels).float()
            
            # Store max uncertainty per sample
            max_uncertainty = uncertainty.max(dim=-1)[0]
            
            all_uncertainties.extend(max_uncertainty.numpy())
            all_correct.extend(correct.numpy())
    
    # Create uncertainty vs accuracy plot
    plt.figure(figsize=(10, 6))
    
    # Bin uncertainties and compute accuracy per bin
    uncertainty_bins = np.linspace(0, max(all_uncertainties), 20)
    bin_accuracies = []
    bin_centers = []
    
    for i in range(len(uncertainty_bins) - 1):
        mask = (np.array(all_uncertainties) >= uncertainty_bins[i]) & \
               (np.array(all_uncertainties) < uncertainty_bins[i + 1])
        if mask.sum() > 0:
            bin_accuracy = np.array(all_correct)[mask].mean()
            bin_accuracies.append(bin_accuracy)
            bin_centers.append((uncertainty_bins[i] + uncertainty_bins[i + 1]) / 2)
    
    plt.plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Model Uncertainty (Max Std Dev)')
    plt.ylabel('Prediction Accuracy')
    plt.title('Uncertainty vs Accuracy: Higher Uncertainty → Lower Accuracy')
    plt.grid(True, alpha=0.3)
    plt.savefig('uncertainty_vs_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Correlation between uncertainty and error: {np.corrcoef(all_uncertainties, 1 - np.array(all_correct))[0,1]:.3f}")


def main():
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    
    # Initialize data module
    data_module = MNISTDataModule(batch_size=128)
    
    # Initialize model
    model = BayesianNN(
        input_dim=784,  # 28x28 flattened
        hidden_dims=[400, 400],
        output_dim=10,
        task_type='classification',
        learning_rate=1e-3,
        kl_weight=1e-4,
        num_samples=10
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        save_top_k=1,
        filename='best-mnist-bnn'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=5,  # Reduced for demo
        callbacks=[checkpoint_callback, early_stopping],
        accelerator='auto',
        log_every_n_steps=50
    )
    
    # Train model
    print("🚀 Training Bayesian Neural Network on MNIST...")
    trainer.fit(model, data_module)
    
    # Load best model
    best_model = BayesianNN.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        input_dim=784,
        hidden_dims=[400, 400],
        output_dim=10,
        task_type='classification'
    )
    
    # Test the model
    trainer.test(best_model, data_module)
    
    # Visualize uncertainty
    print("📊 Generating uncertainty visualizations...")
    test_loader = data_module.val_dataloader()
    visualize_uncertainty(best_model, test_loader)
    analyze_uncertainty_vs_accuracy(best_model, test_loader)
    
    print("✅ MNIST Bayesian NN training complete!")
    print("📈 Check 'mnist_uncertainty_visualization.png' and 'uncertainty_vs_accuracy.png'")


if __name__ == "__main__":
    main()