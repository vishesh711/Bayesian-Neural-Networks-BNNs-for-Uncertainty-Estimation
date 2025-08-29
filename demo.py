#!/usr/bin/env python3
"""
Bayesian Neural Networks Demo Script
Run all examples and generate comprehensive visualizations
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors gracefully"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'torchvision', 'pytorch_lightning', 
        'pyro-ppl', 'matplotlib', 'seaborn', 'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'pyro-ppl':
                import pyro
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies satisfied!")
    return True


def create_results_directory():
    """Create directory for results"""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    print(f"📁 Results will be saved to: {results_dir.absolute()}")
    return results_dir


def main():
    """Run complete Bayesian Neural Networks demo"""
    
    print("🧠 BAYESIAN NEURAL NETWORKS DEMO")
    print("=" * 60)
    print("This demo showcases uncertainty estimation in deep learning")
    print("for safety-critical applications like healthcare.")
    print()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first!")
        sys.exit(1)
    
    # Create results directory
    results_dir = create_results_directory()
    
    # Demo menu
    print("\n📋 Available Demos:")
    print("1. MNIST Classification with Uncertainty")
    print("2. Regression with Uncertainty Bands") 
    print("3. Medical Classification for Clinical Decisions")
    print("4. Run All Demos")
    print("5. Quick Test (Fast version)")
    
    choice = input("\nSelect demo (1-5): ").strip()
    
    success_count = 0
    total_demos = 0
    
    if choice in ['1', '4', '5']:
        total_demos += 1
        print("\n" + "="*60)
        print("🔢 DEMO 1: MNIST Classification with Uncertainty")
        print("="*60)
        print("Shows how BNNs provide confidence estimates for digit classification.")
        print("Key insight: Model is less confident on ambiguous/corrupted digits.")
        
        if run_command("python train_mnist.py", "Training MNIST Bayesian Classifier"):
            success_count += 1
            print("📊 Generated: mnist_uncertainty_visualization.png")
            print("📊 Generated: uncertainty_vs_accuracy.png")
    
    if choice in ['2', '4', '5']:
        total_demos += 1
        print("\n" + "="*60)
        print("📈 DEMO 2: Regression with Uncertainty Bands")
        print("="*60)
        print("Shows how BNNs provide uncertainty estimates for continuous predictions.")
        print("Key insight: Uncertainty increases outside training data region.")
        
        if run_command("python train_regression.py", "Training Regression with Uncertainty"):
            success_count += 1
            print("📊 Generated: regression_uncertainty.png")
            print("📊 Generated: uncertainty_vs_input.png")
    
    if choice in ['3', '4', '5']:
        total_demos += 1
        print("\n" + "="*60)
        print("🏥 DEMO 3: Medical Classification for Clinical Decisions")
        print("="*60)
        print("Shows how BNNs enable safe AI in healthcare by flagging uncertain cases.")
        print("Key insight: High uncertainty cases should be reviewed by doctors.")
        
        if run_command("python train_medical.py", "Training Medical Bayesian Classifier"):
            success_count += 1
            print("📊 Generated: medical_analysis.png")
    
    # Summary
    print("\n" + "="*60)
    print("📋 DEMO SUMMARY")
    print("="*60)
    print(f"✅ Successful demos: {success_count}/{total_demos}")
    
    if success_count == total_demos:
        print("\n🎉 All demos completed successfully!")
        print("\n💼 RESUME IMPACT:")
        print("• Implemented Bayesian Neural Networks for uncertainty quantification")
        print("• Applied probabilistic ML to healthcare for safer AI deployment")
        print("• Used PyTorch Lightning and Pyro for scalable Bayesian inference")
        print("• Demonstrated model uncertainty visualization and clinical decision support")
        
        print("\n📊 Generated Visualizations:")
        for img_file in ["mnist_uncertainty_visualization.png", 
                        "uncertainty_vs_accuracy.png",
                        "regression_uncertainty.png", 
                        "uncertainty_vs_input.png",
                        "medical_analysis.png"]:
            if os.path.exists(img_file):
                print(f"  ✅ {img_file}")
        
        print("\n🔗 Next Steps:")
        print("• Add these visualizations to your portfolio")
        print("• Extend to real medical datasets (with proper permissions)")
        print("• Implement other uncertainty methods (MC Dropout, Deep Ensembles)")
        print("• Deploy as a web service with uncertainty-aware predictions")
        
    else:
        print(f"\n⚠️  {total_demos - success_count} demos failed. Check error messages above.")
    
    print(f"\n📁 All results saved in: {os.path.abspath('.')}")


if __name__ == "__main__":
    main()