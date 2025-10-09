#!/usr/bin/env python3
"""
Quick Start Script for FUREcast GBR Demo

This script checks prerequisites and guides you through setup.
"""

import sys
import subprocess
import os
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def print_step(number, text):
    """Print a step number and description"""
    print(f"[{number}] {text}")

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_error(text):
    """Print error message"""
    print(f"❌ {text}")

def print_warning(text):
    """Print warning message"""
    print(f"⚠️  {text}")

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print_error(f"Python 3.8+ required, found {version.major}.{version.minor}.{version.micro}")
        return False

def check_pip():
    """Check if pip is available"""
    try:
        subprocess.run(['pip', '--version'], capture_output=True, check=True)
        print_success("pip is available")
        return True
    except:
        print_error("pip not found")
        return False

def check_env_file():
    """Check if .env file exists and is configured"""
    env_path = Path('..') / '.env'
    
    if not env_path.exists():
        print_warning(".env file not found in workspace root")
        return False
    
    with open(env_path, 'r') as f:
        content = f.read()
        
    if 'your_openai_api_key_here' in content or 'OPENAI_API_KEY=' not in content:
        print_warning(".env exists but OpenAI API key not configured")
        return False
    
    print_success(".env file found and configured")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print_step("Installing", "Python packages from requirements.txt...")
    try:
        subprocess.run(
            ['pip', 'install', '-r', 'requirements.txt'],
            check=True,
            capture_output=True
        )
        print_success("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error("Failed to install dependencies")
        print(e.stderr.decode())
        return False

def setup_env_file():
    """Guide user through .env setup"""
    print("\nLet's set up your OpenAI API key.")
    print("\nYou need an OpenAI API key to run this demo.")
    print("Get one at: https://platform.openai.com/api-keys\n")
    
    api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print_warning("Skipping API key setup. You'll need to configure it manually later.")
        return False
    
    env_path = Path('..') / '.env'
    with open(env_path, 'w') as f:
        f.write(f"OPENAI_API_KEY={api_key}\n")
    
    print_success(f".env file created at {env_path.absolute()}")
    return True

def main():
    """Main setup flow"""
    print_header("FUREcast GBR Demo - Quick Start")
    
    print("This script will help you set up the demo.\n")
    
    # Check prerequisites
    print_step(1, "Checking Python version...")
    if not check_python_version():
        print("\nPlease install Python 3.8 or higher and try again.")
        sys.exit(1)
    
    print_step(2, "Checking pip...")
    if not check_pip():
        print("\nPlease install pip and try again.")
        sys.exit(1)
    
    # Check if in correct directory
    if not Path('requirements.txt').exists():
        print_error("requirements.txt not found!")
        print("Please run this script from the gbr_ui_test directory.")
        sys.exit(1)
    
    # Install dependencies
    print_step(3, "Checking dependencies...")
    
    # Try to import required packages
    try:
        import streamlit
        import plotly
        import openai
        print_success("All dependencies already installed")
    except ImportError:
        if not install_dependencies():
            print("\nFailed to install dependencies. Try manually:")
            print("  pip install -r requirements.txt")
            sys.exit(1)
    
    # Check .env file
    print_step(4, "Checking OpenAI API key configuration...")
    if not check_env_file():
        print("\nThe app needs an OpenAI API key to function.")
        setup_choice = input("\nWould you like to set it up now? (y/n): ").lower()
        
        if setup_choice == 'y':
            if not setup_env_file():
                print("\nYou can configure it later by creating ../.env with:")
                print("  OPENAI_API_KEY=your_key_here")
        else:
            print("\nRemember to create ../.env before running the app:")
            print("  cp .env.example ../.env")
            print("  # Then edit ../.env with your API key")
    
    # Success!
    print_header("Setup Complete!")
    
    print("✨ You're all set! Here's how to run the demo:\n")
    print("  streamlit run app.py\n")
    
    print("📚 Helpful resources:")
    print("  • README.md          - Overview and setup instructions")
    print("  • DEMO_GUIDE.md      - How to present this demo")
    print("  • ARCHITECTURE.md    - Technical details")
    print("  • TROUBLESHOOTING.md - Common issues and solutions")
    
    print("\n💡 Tips:")
    print("  • Start with Lite mode to see basic predictions")
    print("  • Switch to Pro mode for natural language queries")
    print("  • Try example queries in the Pro mode interface")
    
    print("\n🎯 Example queries to try:")
    print('  • "Is now a good time to invest in SPLG?"')
    print('  • "Which sectors look stable this quarter?"')
    print('  • "What influenced today\'s prediction?"')
    
    # Ask if they want to run now
    print()
    run_now = input("Would you like to launch the app now? (y/n): ").lower()
    
    if run_now == 'y':
        print("\n🚀 Launching FUREcast...\n")
        try:
            subprocess.run(['streamlit', 'run', 'app.py'])
        except KeyboardInterrupt:
            print("\n\nApp stopped. Run again with: streamlit run app.py")
        except Exception as e:
            print_error(f"Failed to launch app: {e}")
            print("\nTry running manually: streamlit run app.py")
    else:
        print("\n👋 When you're ready, run: streamlit run app.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(0)
