import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models.bayesian_nn import BayesianNN
import seaborn as sns


def generate_medical_data(n_samples=2000, n_features=20):
    """Generate synthetic medical classification data"""
    np.random.seed(42)
    
    # Generate base classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=2,
        class_sep=0.8,
        random_state=42
    )
    
    # Add some medical-like feature names
    feature_names = [
        'age', 'bmi', 'blood_pressure_sys', 'blood_pressure_dia', 'cholesterol',
        'glucose', 'heart_rate', 'temperature', 'white_blood_cells', 'red_blood_cells',
        'hemoglobin', 'platelets', 'creatinine', 'protein_levels', 'calcium',
        'sodium', 'potassium', 'liver_enzymes', 'tumor_markers', 'inflammation_markers'
    ]
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled.astype(np.float32), y.astype(np.int64), feature_names


class MedicalDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64):
        super().__init__()
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        # Generate medical data
        X, y, self.feature_names = generate_medical_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Convert to tensors
        self.train_dataset = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        self.val_dataset = TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val)
        )
        self.test_dataset = TensorDataset(
            torch.from_numpy(X_test), torch.from_numpy(y_test)
        )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


def analyze_medical_predictions(model, data_module):
    """Analyze medical predictions with uncertainty for clinical decision making"""
    model.eval()
    
    test_loader = data_module.test_dataloader()
    
    all_predictions = []
    all_uncertainties = []
    all_labels = []
    all_features = []
    
    with torch.no_grad():
        for batch in test_loader:
            features, labels = batch
            
            # Get predictions with uncertainty
            mean_pred, uncertainty = model.predict_with_uncertainty(features, num_samples=50)
            
            all_predictions.extend(mean_pred.numpy())
            all_uncertainties.extend(uncertainty.numpy())
            all_labels.extend(labels.numpy())
            all_features.extend(features.numpy())
    
    all_predictions = np.array(all_predictions)
    all_uncertainties = np.array(all_uncertainties)
    all_labels = np.array(all_labels)
    all_features = np.array(all_features)
    
    # Get predicted classes and confidence
    pred_classes = all_predictions.argmax(axis=1)
    pred_confidence = all_predictions.max(axis=1)
    max_uncertainty = all_uncertainties.max(axis=1)
    
    # Create clinical decision framework
    plt.figure(figsize=(15, 10))
    
    # 1. Confidence vs Uncertainty scatter plot
    plt.subplot(2, 3, 1)
    colors = ['red' if pred != true else 'green' 
             for pred, true in zip(pred_classes, all_labels)]
    plt.scatter(pred_confidence, max_uncertainty, c=colors, alpha=0.6)
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Model Uncertainty')
    plt.title('Clinical Decision Matrix\nRed: Wrong, Green: Correct')
    
    # Add decision boundaries
    plt.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='High Uncertainty')
    plt.axvline(x=0.8, color='blue', linestyle='--', alpha=0.7, label='High Confidence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Uncertainty distribution by correctness
    plt.subplot(2, 3, 2)
    correct_mask = pred_classes == all_labels
    
    plt.hist(max_uncertainty[correct_mask], bins=20, alpha=0.7, 
            label='Correct Predictions', color='green', density=True)
    plt.hist(max_uncertainty[~correct_mask], bins=20, alpha=0.7, 
            label='Incorrect Predictions', color='red', density=True)
    plt.xlabel('Model Uncertainty')
    plt.ylabel('Density')
    plt.title('Uncertainty Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Clinical decision categories
    plt.subplot(2, 3, 3)
    
    # Define decision categories
    high_conf_low_unc = (pred_confidence > 0.8) & (max_uncertainty < 0.1)
    low_conf_high_unc = (pred_confidence < 0.6) & (max_uncertainty > 0.15)
    moderate = ~(high_conf_low_unc | low_conf_high_unc)
    
    categories = ['High Confidence\nLow Uncertainty\n(Auto-approve)', 
                 'Moderate\n(Review)', 
                 'Low Confidence\nHigh Uncertainty\n(Manual review)']
    counts = [high_conf_low_unc.sum(), moderate.sum(), low_conf_high_unc.sum()]
    colors_cat = ['green', 'orange', 'red']
    
    plt.pie(counts, labels=categories, colors=colors_cat, autopct='%1.1f%%')
    plt.title('Clinical Decision Categories')
    
    # 4. Accuracy by uncertainty quartiles
    plt.subplot(2, 3, 4)
    
    # Divide into uncertainty quartiles
    uncertainty_quartiles = np.percentile(max_uncertainty, [25, 50, 75])
    quartile_labels = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
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
    
    plt.bar(quartile_labels, quartile_accuracies, color=['green', 'yellow', 'orange', 'red'])
    plt.ylabel('Accuracy')
    plt.xlabel('Uncertainty Quartile')
    plt.title('Accuracy vs Uncertainty Quartiles')
    plt.ylim(0, 1)
    
    # Add accuracy values on bars
    for i, acc in enumerate(quartile_accuracies):
        plt.text(i, acc + 0.02, f'{acc:.3f}', ha='center')
    
    # 5. Feature importance for high uncertainty cases
    plt.subplot(2, 3, 5)
    
    # Find high uncertainty cases
    high_unc_mask = max_uncertainty > np.percentile(max_uncertainty, 80)
    high_unc_features = all_features[high_unc_mask]
    
    # Compute feature statistics for high uncertainty cases
    feature_means = np.abs(high_unc_features).mean(axis=0)
    top_features_idx = np.argsort(feature_means)[-10:]  # Top 10 features
    
    plt.barh(range(10), feature_means[top_features_idx])
    plt.yticks(range(10), [data_module.feature_names[i] for i in top_features_idx])
    plt.xlabel('Mean Absolute Feature Value')
    plt.title('Top Features in High Uncertainty Cases')
    
    # 6. Calibration plot
    plt.subplot(2, 3, 6)
    
    # Bin predictions by confidence
    conf_bins = np.linspace(0, 1, 11)
    bin_accuracies = []
    bin_confidences = []
    
    for i in range(len(conf_bins) - 1):
        mask = (pred_confidence >= conf_bins[i]) & (pred_confidence < conf_bins[i + 1])
        if mask.sum() > 0:
            bin_accuracy = (pred_classes[mask] == all_labels[mask]).mean()
            bin_confidence = pred_confidence[mask].mean()
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
    
    plt.plot(bin_confidences, bin_accuracies, 'o-', label='Model')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('medical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print clinical insights
    print("\n🏥 CLINICAL DECISION SUPPORT ANALYSIS")
    print("=" * 50)
    
    total_samples = len(all_labels)
    high_conf_low_unc_count = high_conf_low_unc.sum()
    low_conf_high_unc_count = low_conf_high_unc.sum()
    
    print(f"📊 Total test samples: {total_samples}")
    print(f"✅ Auto-approve candidates: {high_conf_low_unc_count} ({100*high_conf_low_unc_count/total_samples:.1f}%)")
    print(f"⚠️  Manual review required: {low_conf_high_unc_count} ({100*low_conf_high_unc_count/total_samples:.1f}%)")
    
    # Accuracy in each category
    if high_conf_low_unc_count > 0:
        auto_approve_acc = (pred_classes[high_conf_low_unc] == all_labels[high_conf_low_unc]).mean()
        print(f"🎯 Auto-approve accuracy: {auto_approve_acc:.3f}")
    
    if low_conf_high_unc_count > 0:
        manual_review_acc = (pred_classes[low_conf_high_unc] == all_labels[low_conf_high_unc]).mean()
        print(f"🔍 Manual review accuracy: {manual_review_acc:.3f}")
    
    # Overall metrics
    overall_acc = (pred_classes == all_labels).mean()
    print(f"📈 Overall accuracy: {overall_acc:.3f}")
    
    # Uncertainty correlation with errors
    error_mask = pred_classes != all_labels
    if error_mask.sum() > 0 and (~error_mask).sum() > 0:
        error_uncertainty = max_uncertainty[error_mask].mean()
        correct_uncertainty = max_uncertainty[~error_mask].mean()
        print(f"🎲 Avg uncertainty (errors): {error_uncertainty:.3f}")
        print(f"🎲 Avg uncertainty (correct): {correct_uncertainty:.3f}")


def main():
    # Set random seeds
    pl.seed_everything(42)
    
    # Initialize data module
    data_module = MedicalDataModule(batch_size=64)
    data_module.setup()
    
    # Initialize Bayesian model
    model = BayesianNN(
        input_dim=20,  # Number of medical features
        hidden_dims=[128, 64, 32],
        output_dim=2,  # Binary classification (healthy/disease)
        task_type='classification',
        learning_rate=1e-3,
        kl_weight=1e-4,
        num_samples=20
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        save_top_k=1,
        filename='best-medical-bnn'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        mode='min'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=10,  # Reduced for demo
        callbacks=[checkpoint_callback, early_stopping],
        accelerator='auto',
        log_every_n_steps=20
    )
    
    # Train model
    print("🚀 Training Bayesian Neural Network for Medical Classification...")
    trainer.fit(model, data_module)
    
    # Load best model
    best_model = BayesianNN.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        input_dim=20,
        hidden_dims=[128, 64, 32],
        output_dim=2,
        task_type='classification'
    )
    
    # Test the model
    trainer.test(best_model, data_module)
    
    # Analyze medical predictions
    print("🏥 Generating clinical decision support analysis...")
    analyze_medical_predictions(best_model, data_module)
    
    print("✅ Medical Bayesian NN training complete!")
    print("📈 Check 'medical_analysis.png' for clinical insights")
    print("\n💡 Key Insight: Use uncertainty to identify cases requiring manual review!")


if __name__ == "__main__":
    main()