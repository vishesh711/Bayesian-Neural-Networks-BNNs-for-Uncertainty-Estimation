import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Tuple, List, Optional


def setup_plotting_style():
    """Setup consistent plotting style for all visualizations"""
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Set default figure parameters
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def plot_uncertainty_histogram(uncertainties: np.ndarray, 
                             correct_mask: np.ndarray,
                             title: str = "Uncertainty Distribution",
                             save_path: Optional[str] = None):
    """Plot histogram of uncertainties split by correctness"""
    setup_plotting_style()
    
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(uncertainties[correct_mask], bins=30, alpha=0.7, 
            label='Correct Predictions', color='green', density=True)
    plt.hist(uncertainties[~correct_mask], bins=30, alpha=0.7, 
            label='Incorrect Predictions', color='red', density=True)
    
    plt.xlabel('Model Uncertainty')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_confidence_vs_uncertainty(predictions: np.ndarray,
                                 uncertainties: np.ndarray,
                                 labels: np.ndarray,
                                 title: str = "Confidence vs Uncertainty",
                                 save_path: Optional[str] = None):
    """Plot confidence vs uncertainty scatter plot"""
    setup_plotting_style()
    
    # Get predicted classes and confidence
    pred_classes = predictions.argmax(axis=1)
    confidence = predictions.max(axis=1)
    max_uncertainty = uncertainties.max(axis=1)
    
    # Color by correctness
    colors = ['red' if pred != true else 'green' 
             for pred, true in zip(pred_classes, labels)]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(confidence, max_uncertainty, c=colors, alpha=0.6, s=30)
    
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Model Uncertainty')
    plt.title(title + '\nRed: Incorrect, Green: Correct')
    
    # Add decision boundaries
    plt.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, 
               label='High Uncertainty Threshold')
    plt.axvline(x=0.8, color='blue', linestyle='--', alpha=0.7, 
               label='High Confidence Threshold')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_calibration_curve(predictions: np.ndarray,
                          labels: np.ndarray,
                          n_bins: int = 10,
                          title: str = "Calibration Plot",
                          save_path: Optional[str] = None):
    """Plot calibration curve for model predictions"""
    setup_plotting_style()
    
    pred_classes = predictions.argmax(axis=1)
    confidence = predictions.max(axis=1)
    
    # Bin predictions by confidence
    conf_bins = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(len(conf_bins) - 1):
        mask = (confidence >= conf_bins[i]) & (confidence < conf_bins[i + 1])
        if mask.sum() > 0:
            bin_accuracy = (pred_classes[mask] == labels[mask]).mean()
            bin_confidence = confidence[mask].mean()
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(mask.sum())
    
    plt.figure(figsize=(10, 6))
    
    # Plot calibration curve
    plt.plot(bin_confidences, bin_accuracies, 'o-', linewidth=2, 
            markersize=8, label='Model')
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2, 
            label='Perfect Calibration')
    
    # Add bin counts as text
    for i, (conf, acc, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
        plt.annotate(f'n={count}', (conf, acc), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9, alpha=0.7)
    
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_uncertainty_vs_accuracy_bins(uncertainties: np.ndarray,
                                    predictions: np.ndarray,
                                    labels: np.ndarray,
                                    n_bins: int = 10,
                                    title: str = "Uncertainty vs Accuracy",
                                    save_path: Optional[str] = None):
    """Plot accuracy vs uncertainty in bins"""
    setup_plotting_style()
    
    pred_classes = predictions.argmax(axis=1)
    max_uncertainty = uncertainties.max(axis=1)
    
    # Bin by uncertainty
    uncertainty_bins = np.percentile(max_uncertainty, np.linspace(0, 100, n_bins + 1))
    bin_accuracies = []
    bin_uncertainties = []
    bin_counts = []
    
    for i in range(len(uncertainty_bins) - 1):
        mask = (max_uncertainty >= uncertainty_bins[i]) & \
               (max_uncertainty < uncertainty_bins[i + 1])
        if mask.sum() > 0:
            bin_accuracy = (pred_classes[mask] == labels[mask]).mean()
            bin_uncertainty = max_uncertainty[mask].mean()
            bin_accuracies.append(bin_accuracy)
            bin_uncertainties.append(bin_uncertainty)
            bin_counts.append(mask.sum())
    
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    bars = plt.bar(range(len(bin_accuracies)), bin_accuracies, 
                  color=plt.cm.RdYlGn(bin_accuracies), alpha=0.8)
    
    # Add uncertainty values as text
    for i, (acc, unc, count) in enumerate(zip(bin_accuracies, bin_uncertainties, bin_counts)):
        plt.text(i, acc + 0.02, f'σ={unc:.3f}\nn={count}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Uncertainty Bin (Low → High)')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.ylim(0, 1.1)
    plt.xticks(range(len(bin_accuracies)), 
              [f'Bin {i+1}' for i in range(len(bin_accuracies))])
    plt.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def create_uncertainty_summary_plot(model, dataloader, 
                                  title: str = "Uncertainty Analysis Summary",
                                  save_path: Optional[str] = None):
    """Create comprehensive uncertainty analysis plot"""
    setup_plotting_style()
    
    # Collect predictions
    all_predictions = []
    all_uncertainties = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                features, labels = batch
            else:
                features, labels = batch[0], batch[1]
            
            mean_pred, uncertainty = model.predict_with_uncertainty(features, num_samples=20)
            
            all_predictions.extend(mean_pred.numpy())
            all_uncertainties.extend(uncertainty.numpy())
            all_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_uncertainties = np.array(all_uncertainties)
    all_labels = np.array(all_labels)
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # 1. Confidence vs Uncertainty
    pred_classes = all_predictions.argmax(axis=1)
    confidence = all_predictions.max(axis=1)
    max_uncertainty = all_uncertainties.max(axis=1)
    
    colors = ['red' if pred != true else 'green' 
             for pred, true in zip(pred_classes, all_labels)]
    
    axes[0, 0].scatter(confidence, max_uncertainty, c=colors, alpha=0.6, s=20)
    axes[0, 0].set_xlabel('Prediction Confidence')
    axes[0, 0].set_ylabel('Model Uncertainty')
    axes[0, 0].set_title('Confidence vs Uncertainty')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Uncertainty Distribution
    correct_mask = pred_classes == all_labels
    axes[0, 1].hist(max_uncertainty[correct_mask], bins=20, alpha=0.7, 
                   label='Correct', color='green', density=True)
    axes[0, 1].hist(max_uncertainty[~correct_mask], bins=20, alpha=0.7, 
                   label='Incorrect', color='red', density=True)
    axes[0, 1].set_xlabel('Model Uncertainty')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Uncertainty Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Calibration Plot
    conf_bins = np.linspace(0, 1, 11)
    bin_accuracies = []
    bin_confidences = []
    
    for i in range(len(conf_bins) - 1):
        mask = (confidence >= conf_bins[i]) & (confidence < conf_bins[i + 1])
        if mask.sum() > 0:
            bin_accuracy = (pred_classes[mask] == all_labels[mask]).mean()
            bin_confidence = confidence[mask].mean()
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
    
    axes[1, 0].plot(bin_confidences, bin_accuracies, 'o-', label='Model')
    axes[1, 0].plot([0, 1], [0, 1], '--', color='gray', label='Perfect')
    axes[1, 0].set_xlabel('Mean Predicted Confidence')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Calibration Plot')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Accuracy by Uncertainty Quartiles
    uncertainty_quartiles = np.percentile(max_uncertainty, [25, 50, 75])
    quartile_labels = ['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)']
    quartile_accuracies = []
    
    for i in range(4):
        if i == 0:
            mask = max_uncertainty <= uncertainty_quartiles[0]
        elif i == 3:
            mask = max_uncertainty > uncertainty_quartiles[2]
        else:
            mask = (max_uncertainty > uncertainty_quartiles[i-1]) & \
                   (max_uncertainty <= uncertainty_quartiles[i])
        
        if mask.sum() > 0:
            accuracy = (pred_classes[mask] == all_labels[mask]).mean()
            quartile_accuracies.append(accuracy)
        else:
            quartile_accuracies.append(0)
    
    bars = axes[1, 1].bar(quartile_labels, quartile_accuracies, 
                         color=['green', 'yellow', 'orange', 'red'], alpha=0.7)
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_xlabel('Uncertainty Quartile')
    axes[1, 1].set_title('Accuracy vs Uncertainty')
    axes[1, 1].set_ylim(0, 1)
    
    # Add accuracy values on bars
    for i, acc in enumerate(quartile_accuracies):
        axes[1, 1].text(i, acc + 0.02, f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
    return {
        'overall_accuracy': (pred_classes == all_labels).mean(),
        'mean_uncertainty': max_uncertainty.mean(),
        'uncertainty_accuracy_correlation': np.corrcoef(max_uncertainty, 1 - correct_mask.astype(float))[0, 1]
    }