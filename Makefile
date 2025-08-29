# Bayesian Neural Networks - Project Makefile

.PHONY: install test demo clean mnist regression medical help

# Default target
help:
	@echo "🧠 Bayesian Neural Networks - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install    - Install all dependencies"
	@echo "  make test      - Run setup validation tests"
	@echo ""
	@echo "Training:"
	@echo "  make demo      - Run complete interactive demo"
	@echo "  make mnist     - Train MNIST classification with uncertainty"
	@echo "  make regression - Train regression with uncertainty bands"
	@echo "  make medical   - Train medical classification for clinical decisions"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean     - Clean up generated files"
	@echo "  make help      - Show this help message"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Installation complete!"

# Run tests
test:
	@echo "🧪 Running setup validation tests..."
	python test_setup.py

# Run complete demo
demo:
	@echo "🚀 Running complete Bayesian Neural Networks demo..."
	python demo.py

# Individual training scripts
mnist:
	@echo "🔢 Training MNIST classification with uncertainty..."
	python train_mnist.py

regression:
	@echo "📈 Training regression with uncertainty bands..."
	python train_regression.py

medical:
	@echo "🏥 Training medical classification for clinical decisions..."
	python train_medical.py

# Clean up generated files
clean:
	@echo "🧹 Cleaning up generated files..."
	rm -f *.png
	rm -rf lightning_logs/
	rm -rf checkpoints/
	rm -rf data/
	rm -rf __pycache__/
	rm -rf models/__pycache__/
	rm -rf utils/__pycache__/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "✅ Cleanup complete!"

# Quick start - install, test, and run demo
quickstart: install test demo
	@echo "🎉 Quickstart complete! Check the generated visualizations."