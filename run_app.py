import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import flask
        import cv2
        import numpy
        import mediapipe
        import sklearn
        print("✅ All core dependencies are installed.")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def install_dependencies():
    """Install required dependencies from requirements.txt"""
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    print("📦 Installing dependencies from requirements.txt...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    return True

def check_model_files():
    """Check if model files exist, if not, train them"""
    model_file = Path("gesture_model.pkl")
    if not model_file.exists():
        print("🔍 Gesture model not found. Checking for datasets...")
        
        # Check if datasets exist
        dataset_dir = Path("datasets")
        if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
            print("🔍 Datasets not found. Downloading datasets...")
            try:
                import download_datasets
                download_datasets.main()
            except Exception as e:
                print(f"❌ Error downloading datasets: {e}")
                return False
        
        # Train model
        print("🧠 Training gesture recognition model...")
        try:
            import train_gesture_model
            train_gesture_model.main()
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return False
    
    return True

def run_app():
    """Run the Flask application"""
    print("🚀 Starting Guardian Eye application...")
    
    # Start Flask app in a separate process
    flask_process = subprocess.Popen(
        [sys.executable, "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for the server to start
    time.sleep(2)
    
    # Check if the process is still running
    if flask_process.poll() is not None:
        stdout, stderr = flask_process.communicate()
        print("❌ Failed to start the application!")
        print(f"Error: {stderr}")
        return False
    
    # Open browser
    print("🌐 Opening application in web browser...")
    webbrowser.open("http://localhost:5000")
    
    print("\n✨ Guardian Eye is now running!")
    print("Press Ctrl+C to stop the application")
    
    try:
        # Keep the script running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping Guardian Eye...")
        flask_process.terminate()
        flask_process.wait()
        print("✅ Application stopped successfully.")
    
    return True

def main():
    print("\n🛡️  Guardian Eye - Women Safety Analytics")
    print("=======================================\n")
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️  Some dependencies are missing. Installing required packages...")
        if not install_dependencies():
            print("❌ Failed to install dependencies. Please install them manually.")
            return False
    
    # Check and prepare model files
    if not check_model_files():
        print("⚠️  Warning: Model preparation incomplete. The application may not function correctly.")
        proceed = input("Do you want to proceed anyway? (y/n): ")
        if proceed.lower() != 'y':
            return False
    
    # Run the application
    return run_app()

if __name__ == "__main__":
    main()