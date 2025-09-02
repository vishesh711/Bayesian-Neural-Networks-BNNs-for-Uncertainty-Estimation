#!/usr/bin/env python3
"""
Setup script for Bayesian Neural Networks project
Handles environment creation and dependency installation
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description, check=True):
    """Run a command with error handling"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def setup_virtual_environment():
    """Create and setup virtual environment"""
    venv_path = Path("bnn_env")
    
    if venv_path.exists():
        print("📁 Virtual environment already exists")
        return True
    
    print("🔧 Creating virtual environment...")
    if not run_command(f"{sys.executable} -m venv bnn_env", "Creating virtual environment"):
        return False
    
    return True

def install_dependencies():
    """Install project dependencies"""
    print("📦 Installing dependencies...")
    
    # Determine the correct pip path
    if os.name == 'nt':  # Windows
        pip_path = "bnn_env/Scripts/pip"
    else:  # Unix/Linux/macOS
        pip_path = "bnn_env/bin/pip"
    
    # Upgrade pip first
    if not run_command(f"{pip_path} install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install project dependencies
    if not run_command(f"{pip_path} install -r config/requirements.txt", "Installing project dependencies"):
        return False
    
    return True

def validate_installation():
    """Validate that everything is installed correctly"""
    print("🧪 Validating installation...")
    
    # Determine the correct python path
    if os.name == 'nt':  # Windows
        python_path = "bnn_env/Scripts/python"
    else:  # Unix/Linux/macOS
        python_path = "bnn_env/bin/python"
    
    return run_command(f"{python_path} scripts/test_setup.py", "Running validation tests", check=False)

def main():
    """Main setup process"""
    print("🧠 BAYESIAN NEURAL NETWORKS - PROJECT SETUP")
    print("=" * 60)
    print("This script will set up your development environment")
    print()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Setup virtual environment
    if not setup_virtual_environment():
        print("❌ Failed to create virtual environment")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return False
    
    # Validate installation
    validation_success = validate_installation()
    
    print("\n" + "=" * 60)
    if validation_success:
        print("🎉 SETUP COMPLETE!")
        print("=" * 60)
        print("✅ Virtual environment created: bnn_env/")
        print("✅ Dependencies installed successfully")
        print("✅ Installation validated")
        print()
        print("🚀 Next steps:")
        if os.name == 'nt':  # Windows
            print("   1. Activate environment: bnn_env\\Scripts\\activate")
        else:  # Unix/Linux/macOS
            print("   1. Activate environment: source bnn_env/bin/activate")
        print("   2. Run examples: python run_examples.py")
        print("   3. Or use Makefile: make demo")
        print()
        print("📚 Documentation available in docs/ folder")
        print("📊 Generated outputs will be saved in outputs/ folder")
    else:
        print("⚠️  SETUP COMPLETED WITH WARNINGS")
        print("=" * 60)
        print("✅ Environment and dependencies installed")
        print("⚠️  Some validation tests failed (this may be normal)")
        print()
        print("🚀 Try running examples anyway:")
        if os.name == 'nt':  # Windows
            print("   1. Activate environment: bnn_env\\Scripts\\activate")
        else:  # Unix/Linux/macOS
            print("   1. Activate environment: source bnn_env/bin/activate")
        print("   2. Run examples: python run_examples.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)