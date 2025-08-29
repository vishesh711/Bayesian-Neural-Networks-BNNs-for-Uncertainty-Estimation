"""
Configuration file for Bayesian Neural Networks project
"""

# Model hyperparameters
MODEL_CONFIG = {
    'mnist': {
        'input_dim': 784,
        'hidden_dims': [400, 400],
        'output_dim': 10,
        'learning_rate': 1e-3,
        'kl_weight': 1e-4,
        'num_samples': 10,
        'batch_size': 128,
        'max_epochs': 50
    },
    'regression': {
        'input_dim': 1,
        'hidden_dims': [100, 100],
        'output_dim': 1,
        'learning_rate': 1e-3,
        'kl_weight': 1e-3,
        'num_samples': 20,
        'batch_size': 64,
        'max_epochs': 200
    },
    'medical': {
        'input_dim': 20,
        'hidden_dims': [128, 64, 32],
        'output_dim': 2,
        'learning_rate': 1e-3,
        'kl_weight': 1e-4,
        'num_samples': 20,
        'batch_size': 64,
        'max_epochs': 100
    }
}

# Training configuration
TRAINING_CONFIG = {
    'patience': 15,
    'monitor_metric': 'val_loss',
    'monitor_mode': 'min',
    'log_every_n_steps': 20,
    'accelerator': 'auto'
}

# Visualization configuration
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': 'husl',
    'uncertainty_samples': 50,
    'calibration_bins': 10
}

# Data configuration
DATA_CONFIG = {
    'mnist': {
        'data_dir': './data',
        'normalize_mean': (0.1307,),
        'normalize_std': (0.3081,)
    },
    'regression': {
        'n_samples': 1000,
        'noise_std': 0.3,
        'test_range': (-4, 4),
        'test_points': 200
    },
    'medical': {
        'n_samples': 2000,
        'n_features': 20,
        'n_informative': 15,
        'class_sep': 0.8,
        'test_size': 0.2
    }
}

# Clinical decision thresholds
CLINICAL_CONFIG = {
    'high_confidence_threshold': 0.8,
    'low_confidence_threshold': 0.6,
    'high_uncertainty_threshold': 0.15,
    'low_uncertainty_threshold': 0.1
}

# File paths
PATHS = {
    'models': './models',
    'results': './results',
    'data': './data',
    'checkpoints': './checkpoints'
}

# Resume-worthy metrics to track
RESUME_METRICS = [
    'uncertainty_accuracy_correlation',
    'calibration_error',
    'clinical_decision_accuracy',
    'auto_approve_percentage',
    'manual_review_percentage'
]