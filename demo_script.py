#!/usr/bin/env python
"""
Guardian Eye Demo Script

This script provides a guided demonstration of the Guardian Eye application
for presentation purposes. It walks through key features and showcases
the application's capabilities in a structured manner.
"""

import os
import sys
import time
import webbrowser
import subprocess
from pathlib import Path

# ANSI color codes for terminal output
COLORS = {
    'HEADER': '\033[95m',
    'BLUE': '\033[94m',
    'GREEN': '\033[92m',
    'YELLOW': '\033[93m',
    'RED': '\033[91m',
    'ENDC': '\033[0m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m'
}

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(text):
    """Print a formatted header"""
    print(f"\n{COLORS['HEADER']}{COLORS['BOLD']}=== {text} ==={COLORS['ENDC']}\n")

def print_step(step_num, text):
    """Print a formatted step"""
    print(f"{COLORS['BLUE']}{COLORS['BOLD']}Step {step_num}:{COLORS['ENDC']} {text}")

def print_feature(text):
    """Print a formatted feature description"""
    print(f"  {COLORS['GREEN']}• {text}{COLORS['ENDC']}")

def wait_for_key():
    """Wait for a key press to continue"""
    input(f"\n{COLORS['YELLOW']}Press Enter to continue...{COLORS['ENDC']}")

def run_app():
    """Run the Flask application"""
    print(f"{COLORS['BLUE']}Starting Guardian Eye application...{COLORS['ENDC']}")
    
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
        print(f"{COLORS['RED']}Failed to start the application!{COLORS['ENDC']}")
        print(f"{COLORS['RED']}Error: {stderr}{COLORS['ENDC']}")
        return None
    
    return flask_process

def demo_introduction():
    """Introduction to the Guardian Eye application"""
    clear_screen()
    print_header("GUARDIAN EYE - WOMEN SAFETY ANALYTICS")
    print(f"{COLORS['BOLD']}Smart India Hackathon 2025 - Internal Edition{COLORS['ENDC']}\n")
    
    print("Guardian Eye is an innovative solution designed to enhance women's safety")
    print("through real-time video analysis and emergency gesture recognition.\n")
    
    print("This demonstration will walk you through the key features and capabilities")
    print("of our application, showcasing how it can help in emergency situations.\n")
    
    wait_for_key()

def demo_features():
    """Showcase the key features of the application"""
    clear_screen()
    print_header("KEY FEATURES")
    
    print_feature("Advanced Gesture Recognition using MediaPipe and Machine Learning")
    print("   - Detects emergency hand gestures with high accuracy")
    print("   - Works in various lighting conditions")
    print("   - Trained on comprehensive Kaggle datasets\n")
    
    print_feature("Environmental Safety Analysis")
    print("   - Evaluates lighting conditions for safety assessment")
    print("   - Detects crowd density and potential risks")
    print("   - Analyzes surroundings for hazardous elements\n")
    
    print_feature("Real-time Risk Assessment")
    print("   - Combines gesture and environmental data")
    print("   - Provides comprehensive safety score")
    print("   - Offers contextual safety recommendations\n")
    
    print_feature("Emergency Alert System")
    print("   - Triggers alerts when emergency gestures are detected")
    print("   - Can be integrated with notification systems")
    print("   - Provides location data for emergency services\n")
    
    wait_for_key()

def demo_technical_architecture():
    """Explain the technical architecture"""
    clear_screen()
    print_header("TECHNICAL ARCHITECTURE")
    
    print(f"{COLORS['BOLD']}Frontend:{COLORS['ENDC']} HTML, CSS, JavaScript")
    print("- Responsive web interface")
    print("- Real-time video processing")
    print("- Interactive safety dashboard\n")
    
    print(f"{COLORS['BOLD']}Backend:{COLORS['ENDC']} Flask (Python)")
    print("- RESTful API endpoints")
    print("- Efficient image processing pipeline")
    print("- Robust error handling\n")
    
    print(f"{COLORS['BOLD']}Computer Vision:{COLORS['ENDC']} OpenCV, MediaPipe")
    print("- Hand landmark detection")
    print("- Environmental analysis")
    print("- Image preprocessing\n")
    
    print(f"{COLORS['BOLD']}Machine Learning:{COLORS['ENDC']} scikit-learn, Random Forest")
    print("- Gesture classification model")
    print("- Feature extraction from hand landmarks")
    print("- Trained on diverse datasets\n")
    
    wait_for_key()

def demo_live_application():
    """Launch and demonstrate the live application"""
    clear_screen()
    print_header("LIVE DEMONSTRATION")
    
    print_step(1, "Launching the Guardian Eye application")
    flask_process = run_app()
    if not flask_process:
        return
    
    print_step(2, "Opening the application in your web browser")
    webbrowser.open("http://localhost:5000")
    
    print("\nThe application is now running in your web browser.")
    print("Please allow camera access when prompted.\n")
    
    print_step(3, "Demonstration instructions:")
    print("  1. Show your hand to the camera with all five fingers extended (Help gesture)")
    print("  2. Make a fist and show it to the camera (Distress signal)")
    print("  3. Point in a direction (Directional signal)")
    print("  4. Try different lighting conditions to see environmental analysis\n")
    
    print(f"{COLORS['YELLOW']}The application will analyze these gestures and provide safety assessments.{COLORS['ENDC']}")
    print(f"{COLORS['YELLOW']}Watch the interface for emergency alerts and safety recommendations.{COLORS['ENDC']}\n")
    
    print(f"{COLORS['BOLD']}Press Ctrl+C when you're ready to end the demonstration.{COLORS['ENDC']}")
    
    try:
        # Keep the script running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n{COLORS['BLUE']}Stopping the demonstration...{COLORS['ENDC']}")
        flask_process.terminate()
        flask_process.wait()

def demo_conclusion():
    """Conclude the demonstration"""
    clear_screen()
    print_header("CONCLUSION")
    
    print("Thank you for experiencing Guardian Eye - Women Safety Analytics!\n")
    
    print("Our solution demonstrates how technology can be leveraged to enhance")
    print("women's safety through:")
    print(f"  {COLORS['GREEN']}• Real-time gesture recognition{COLORS['ENDC']}")
    print(f"  {COLORS['GREEN']}• Environmental safety analysis{COLORS['ENDC']}")
    print(f"  {COLORS['GREEN']}• Comprehensive risk assessment{COLORS['ENDC']}")
    print(f"  {COLORS['GREEN']}• Contextual safety recommendations{COLORS['ENDC']}\n")
    
    print("With further development, this system could be integrated with:")
    print(f"  {COLORS['BLUE']}• Mobile applications for wider accessibility{COLORS['ENDC']}")
    print(f"  {COLORS['BLUE']}• Emergency services for immediate response{COLORS['ENDC']}")
    print(f"  {COLORS['BLUE']}• Public safety infrastructure{COLORS['ENDC']}")
    print(f"  {COLORS['BLUE']}• Community alert networks{COLORS['ENDC']}\n")
    
    print(f"{COLORS['BOLD']}Guardian Eye: Empowering safety through technology.{COLORS['ENDC']}")
    
    wait_for_key()

def main():
    """Run the complete demonstration"""
    try:
        demo_introduction()
        demo_features()
        demo_technical_architecture()
        demo_live_application()
        demo_conclusion()
    except KeyboardInterrupt:
        clear_screen()
        print("Demonstration ended by user.")
    except Exception as e:
        print(f"\n{COLORS['RED']}Error during demonstration: {str(e)}{COLORS['ENDC']}")
    
    print("\nThank you for your attention!")

if __name__ == "__main__":
    main()