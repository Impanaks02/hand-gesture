"""
Hand Gesture Applications Main Menu
"""
import os
import sys
import subprocess

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_menu():
    """Display the main menu"""
    print("=" * 50)
    print("         HAND GESTURE APPLICATIONS")
    print("=" * 50)
    print("1. Hand Tracker")
    print("2. Hand Poem Creator (requires Gemini API key)")
    print("3. Install Dependencies")
    print("4. Exit")
    print("=" * 50)

def run_hand_tracker():
    """Run the Hand Tracker application"""
    print("Starting Hand Tracker...")
    try:
        # Use shell=True to ensure the command is executed correctly in the shell
        subprocess.run([sys.executable, "hand_track.py"], check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Hand Tracker: {e}")
    except FileNotFoundError:
        print("Error: hand_track.py not found. Make sure it's in the same directory.")
    input("Press Enter to continue...")

def run_poem_creator():
    """Run the Hand Poem Creator application"""
    print("Starting Hand Poem Creator...")
    # Check if .env file exists with API key
    if not os.path.exists('.env'):
        print("Warning: .env file not found.")
        print("You need a Google Gemini API key for this application.")
        print("Create a .env file with: API_KEY=your_actual_api_key_here")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    try:
        # Try to run using uv first
        result = subprocess.run([sys.executable, "-m", "uv", "run", "hand_poem_creator.py"], 
                                check=False)
        if result.returncode != 0:
            # Fallback to direct Python execution
            subprocess.run([sys.executable, "hand_poem_creator.py"])
    except FileNotFoundError:
        # Fallback if uv is not available
        subprocess.run([sys.executable, "hand_poem_creator.py"])
    input("Press Enter to continue...")

def install_dependencies():
    """Install project dependencies"""
    print("Installing dependencies...")
    try:
        # Try using uv
        subprocess.run([sys.executable, "-m", "uv", "pip", "install", "-r", "requirements.txt"], 
                       check=True)
        print("Dependencies installed successfully with uv!")
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Fallback to pip
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                           check=True)
            print("Dependencies installed successfully with pip!")
        except subprocess.CalledProcessError:
            print("Failed to install dependencies. Please check your Python setup.")
    input("Press Enter to continue...")

def main():
    """Main application menu"""
    while True:
        clear_screen()
        display_menu()
        
        choice = input("Select an option (1-4): ").strip()
        
        if choice == '1':
            clear_screen()
            run_hand_tracker()
        elif choice == '2':
            clear_screen()
            run_poem_creator()
        elif choice == '3':
            clear_screen()
            install_dependencies()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
