#!/usr/bin/env python3
"""
Run script for SkyPin
Simple script to run the SkyPin application
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if requirements are installed."""
    try:
        import streamlit
        import cv2
        import numpy
        import skyfield
        import ultralytics
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Please run: python setup.py")
        return False

def main():
    """Main run function."""
    print("🧭 Starting SkyPin...")
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("✗ app.py not found. Please run from the SkyPin directory.")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Set environment variables
    os.environ.setdefault('SKYPIN_ENV', 'development')
    
    # Run Streamlit
    print("🚀 Launching Streamlit app...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to start Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 SkyPin stopped by user")

if __name__ == "__main__":
    main()