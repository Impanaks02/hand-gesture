# Hand Gesture Projects

A collection of interactive hand gesture applications built with Python, OpenCV, and MediaPipe.

## Projects

### 1. Hand Tracker (`hand_tracker.py`)
A real-time hand tracking application with stabilization, FPS counter, and interactive UI controls.

**Features:**
- Real-time hand detection with bounding box
- Stabilization algorithm for smooth tracking
- FPS counter display
- Toggle landmarks and FPS display
- Enhanced visual feedback with corner markers and information panel

### 2. Hand Poem Creator (`hand_poem_creator.py`)
An AI-powered creative application that generates poems based on hand gestures using Google's Gemini AI.

**Features:**
- Hand gesture-based poem generation
- Integration with Google Gemini AI
- Multiple poetry topics
- Dynamic text rendering within hand bounding boxes
- Poem saving functionality (JSON format)
- Beautiful UI with animations and visual effects

## Installation

Using UV (recommended):

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <your-repo-url>
cd hand-gesture

# Sync dependencies with UV
uv sync

# Alternatively, install directly
uv pip install -r requirements.txt
```

Using traditional pip:

```bash
pip install -r requirements.txt
```

## Setup

1. For the Hand Poem Creator, you need a Google Gemini API key:
   - Get an API key from [Google AI Studio](https://aistudio.google.com/)
   - Create a `.env` file in the project root:
     ```
     API_KEY=your_actual_api_key_here
     ```

## Usage

Run the main menu:
```bash
uv run main.py
```

Or run individual applications directly:
```bash
# Run hand tracker
uv run hand_tracker.py

# Run poem creator (requires API key setup)
uv run hand_poem_creator.py
```

## Controls

### Hand Tracker
- `l` - Toggle hand landmarks
- `f` - Toggle FPS display
- `r` - Reset stabilization buffer
- `q` - Quit application

### Hand Poem Creator
- `p` - Generate poem based on hand position
- `t` - Change poem topic
- `l` - Toggle hand landmarks
- `s` - Save all generated poems to file
- `c` - Clear current poem
- `q` - Quit application

## Requirements

The project requires Python 3.8+ and the following dependencies:

- opencv-python
- mediapipe
- google-generativeai
- python-dotenv
- Pillow
- numpy

## Project Structure

```
hand-gesture/
├── main.py              # Main application menu
├── hand_tracker.py      # Hand tracking application
├── hand_poem_creator.py # AI poem generation application
├── requirements.txt     # Project dependencies
├── .env                # Environment variables (create this)
└── README.md           # This file
```

## License

This project is open source and available under the MIT License.
