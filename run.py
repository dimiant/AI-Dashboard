import streamlit as st
import subprocess
import sys
import os

def main():
    """Simple runner for the Streamlit app"""
    print("🚀 Starting AI Visibility Monitor...")
    print("📁 Make sure your CSV files are in the 'data' folder")
    print("🌐 The app will open in your browser")
    print("⏹️  Press Ctrl+C to stop the app")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "visibility_dashboard.py",
        "--server.headless", "false",
    ])

if __name__ == "__main__":
    main()
