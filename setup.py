"""
Setup script for SkyPin
Installs dependencies and sets up the environment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🧭 SkyPin Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("✗ Failed to install dependencies")
        sys.exit(1)
    
    # Create necessary directories
    directories = ['models', 'data', 'output', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Download models (if needed)
    print("\n📥 Downloading models...")
    
    # Note: In a real implementation, you would download pre-trained models here
    # For now, we'll create placeholder files
    models_dir = Path("models")
    
    # Create placeholder model files
    placeholder_models = [
        "sun_detector.pt",
        "shadow_detector.pt",
        "tamper_detector.pt"
    ]
    
    for model in placeholder_models:
        model_path = models_dir / model
        if not model_path.exists():
            # Create a small placeholder file
            with open(model_path, 'w') as f:
                f.write("# Placeholder model file\n")
            print(f"✓ Created placeholder: {model}")
    
    # Set up logging
    print("\n📝 Setting up logging...")
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    print("✓ Logging directory created")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write("# SkyPin Environment Variables\n")
            f.write("SKYPIN_ENV=development\n")
            f.write("LOG_LEVEL=INFO\n")
            f.write("MAX_UPLOAD_SIZE=200MB\n")
        print("✓ Created .env file")
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo run SkyPin:")
    print("  streamlit run app.py")
    print("\nTo run with custom settings:")
    print("  SKYPIN_ENV=production streamlit run app.py")

if __name__ == "__main__":
    main()