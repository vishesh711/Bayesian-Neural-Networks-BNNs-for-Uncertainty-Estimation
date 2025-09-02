#!/usr/bin/env python3
"""
Simple launcher script for Bayesian Neural Networks examples
Run from the project root directory
"""

import subprocess
import sys
import os

def run_example(script_name, description):
    """Run an example script with error handling"""
    print(f"\n🚀 {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run([sys.executable, f"examples/{script_name}"], 
                              check=True, cwd=os.getcwd())
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Could not find examples/{script_name}")
        print("💡 Make sure you're running from the project root directory")
        return False

def main():
    print("🧠 BAYESIAN NEURAL NETWORKS - EXAMPLE LAUNCHER")
    print("=" * 60)
    print("Choose an example to run:")
    print()
    print("1. Simple Demo (2D classification + regression) - 2 minutes")
    print("2. Medical Classification (clinical decisions) - 5 minutes") 
    print("3. MNIST Classification (digit recognition) - 7 minutes")
    print("4. Regression Analysis (confidence bands) - 3 minutes")
    print("5. Run All Examples")
    print("6. Interactive Demo Menu")
    print()
    
    choice = input("Enter your choice (1-6): ").strip()
    
    examples = {
        '1': ('simple_demo.py', 'Simple Demo'),
        '2': ('train_medical.py', 'Medical Classification'),
        '3': ('train_mnist.py', 'MNIST Classification'),
        '4': ('train_regression.py', 'Regression Analysis')
    }
    
    if choice in examples:
        script, description = examples[choice]
        run_example(script, description)
    elif choice == '5':
        print("\n🚀 Running all examples...")
        success_count = 0
        for script, description in examples.values():
            if run_example(script, description):
                success_count += 1
        print(f"\n📊 Results: {success_count}/{len(examples)} examples completed successfully")
    elif choice == '6':
        print("\n🎮 Launching interactive demo menu...")
        try:
            subprocess.run([sys.executable, "scripts/demo.py"], check=True)
        except subprocess.CalledProcessError:
            print("❌ Interactive demo failed")
        except FileNotFoundError:
            print("❌ Could not find scripts/demo.py")
    else:
        print("❌ Invalid choice. Please run the script again and choose 1-6.")
        return
    
    print(f"\n📁 Generated visualizations are saved in: outputs/")
    print("📚 Check docs/ folder for complete documentation")

if __name__ == "__main__":
    main()