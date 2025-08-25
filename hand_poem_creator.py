import os
import cv2
import mediapipe as mp
import google.generativeai as genai
import numpy as np
import time
import json
from datetime import datetime
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("Warning: API key not found. Hand Poem Creator will not work.")
    exit()

class HandPoemCreator:
    """
    A class to create a hand-tracking application that generates poems.
    It detects two hands to form a bounding box and then generates a poem
    that is dynamically fitted into the box.
    """
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Configure the Gemini model
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # UI Configuration
        self.colors = {
            'box': (255, 255, 255),  # Changed to White
            'glow': (255, 255, 255), # Changed to White
            'text': (0, 255, 0),     # Changed to Green
            'background': (20, 20, 20),
            'landmarks': (0, 255, 255),
            'connections': (255, 0, 255)
        }
        
        # State variables
        self.topic = "Nature"
        self.show_landmarks = False
        self.show_fps = True
        self.stabilization_buffer = []
        # Increased buffer size for greater stability
        self.buffer_size = 10 
        
        # FPS calculation
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Poem state
        self.poem_text = ""
        self.poem_generated_count = 0
        self.status_message = "Move hands to form a box."
        self.user_poem = None # New variable to store user's poem
        
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            end_time = time.time()
            self.current_fps = self.fps_counter / (end_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def stabilize_box(self, box_coords):
        """Stabilize bounding box using moving average"""
        self.stabilization_buffer.append(box_coords)
        if len(self.stabilization_buffer) > self.buffer_size:
            self.stabilization_buffer.pop(0)
        
        if not self.stabilization_buffer:
            return box_coords
        
        # Calculate average
        avg_coords = np.mean(self.stabilization_buffer, axis=0)
        return tuple(map(int, avg_coords))
    
    def draw_ui_panel(self, frame):
        """Draw a sleek information panel at the top"""
        h, w = frame.shape[:2]
        panel_height = 120
        
        # Create a semi-transparent panel overlay with a smooth gradient effect
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), self.colors['background'], -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Draw a subtle line at the bottom of the panel for separation
        cv2.line(frame, (0, panel_height), (w, panel_height), self.colors['box'], 2)
        
        # Title with a different font for emphasis
        cv2.putText(frame, "Hand Poem Creator", (20, 25), 
                    cv2.FONT_HERSHEY_COMPLEX, 1.0, self.colors['box'], 2)
        
        # Instructions and info
        lines = [
            f"Topic: {self.topic}",
            f"Status: {self.status_message}",
            f"Poems Generated: {self.poem_generated_count}",
            "Press 'p' to generate a poem, 'q' to quit."
        ]
        
        y_offset = 55
        for line in lines:
            cv2.putText(frame, line, (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
            y_offset += 25
        
        if self.show_fps:
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
        
        return frame
    
    def draw_enhanced_box(self, frame, box_coords, box_width, box_height):
        """Draw a visually enhanced bounding box with corner markers"""
        min_x, min_y, max_x, max_y = box_coords
        
        # Add a "glow" effect by drawing a slightly larger, semi-transparent box
        cv2.rectangle(frame, (min_x - 1, min_y - 1), (max_x + 1, max_y + 1), self.colors['glow'], 2)
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), self.colors['glow'], 1)

        # Draw sleek corner markers
        marker_size = 25
        thickness = 2
        # Top-left
        cv2.line(frame, (min_x, min_y), (min_x + marker_size, min_y), self.colors['box'], thickness)
        cv2.line(frame, (min_x, min_y), (min_x, min_y + marker_size), self.colors['box'], thickness)
        # Top-right
        cv2.line(frame, (max_x, min_y), (max_x - marker_size, min_y), self.colors['box'], thickness)
        cv2.line(frame, (max_x, min_y), (max_x, min_y + marker_size), self.colors['box'], thickness)
        # Bottom-left
        cv2.line(frame, (min_x, max_y), (min_x + marker_size, max_y), self.colors['box'], thickness)
        cv2.line(frame, (min_x, max_y), (min_x, max_y - marker_size), self.colors['box'], thickness)
        # Bottom-right
        cv2.line(frame, (max_x, max_y), (max_x - marker_size, max_y), self.colors['box'], thickness)
        cv2.line(frame, (max_x, max_y), (max_x, max_y - marker_size), self.colors['box'], thickness)
        
        # Center point
        center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2
        cv2.circle(frame, (center_x, center_y), 5, self.colors['box'], -1)
        
        # Draw the poem inside the box if it exists
        if self.poem_text:
            self.draw_wrapped_text(frame, self.poem_text, box_coords)
            
    def draw_wrapped_text(self, frame, text, box_coords):
        """
        Draws text wrapped and scaled to fit dynamically inside a bounding box.
        This version iteratively finds the best font scale for legibility.
        """
        min_x, min_y, max_x, max_y = box_coords
        box_width = max_x - min_x
        box_height = max_y - min_y
        
        if box_width <= 0 or box_height <= 0:
            return

        font = cv2.FONT_HERSHEY_COMPLEX # Changed font style for the poem
        font_scale = 1.0  # Start with a good font scale
        thickness = 1
        
        # Find the best font scale that fits both horizontally and vertically
        while font_scale > 0.1: # Prevent scale from getting too small
            lines = []
            current_line = ""
            words = text.split(' ')

            for word in words:
                test_line = current_line + word + " "
                text_size, _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                text_width, _ = text_size
                
                if text_width > box_width and len(current_line) > 0:
                    lines.append(current_line.strip())
                    current_line = word + " "
                else:
                    current_line = test_line
            lines.append(current_line.strip())

            # Calculate total height of the wrapped text
            line_height = cv2.getTextSize("A", font, font_scale, thickness)[0][1] + 5 # Add extra space
            total_text_height = len(lines) * line_height
            
            if total_text_height <= box_height:
                break # Found a scale that fits
            
            font_scale -= 0.05 # Reduce font size and try again
            
        # Draw the lines with the final calculated font scale
        y_offset = min_y + int(cv2.getTextSize("A", font, font_scale, thickness)[0][1]) + 10 # 10px padding from top
        for line in lines:
            text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
            text_width, _ = text_size
            
            x_pos = min_x + (box_width - text_width) // 2 # Center text horizontally
            cv2.putText(frame, line, (x_pos, y_offset), 
                        font, font_scale, self.colors['text'], thickness)
            y_offset += int(cv2.getTextSize("A", font, font_scale, thickness)[0][1]) + 5 # Use same spacing

    def process_hands(self, frame):
        """Process hand detection and return bounding box coordinates based on finger tips"""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) < 2:
            return None, frame
        
        all_relevant_tip_coords = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the coordinates for the thumb and index finger tips
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Convert to pixel coordinates and add to list
            all_relevant_tip_coords.append((int(thumb_tip.x * w), int(thumb_tip.y * h)))
            all_relevant_tip_coords.append((int(index_finger_tip.x * w), int(index_finger_tip.y * h)))
            
        if all_relevant_tip_coords:
            # Calculate bounding box from all selected finger tips
            min_x = min(x for x, y in all_relevant_tip_coords)
            min_y = min(y for x, y in all_relevant_tip_coords)
            max_x = max(x for x, y in all_relevant_tip_coords)
            max_y = max(y for x, y in all_relevant_tip_coords)
            
            # Add padding
            padding = 15
            min_x = max(0, min_x - padding)
            min_y = max(130, min_y - padding)  # Account for UI panel
            max_x = min(w, max_x + padding)
            max_y = min(h, max_y + padding)
            
            return (min_x, min_y, max_x, max_y), frame
        
        return None, frame

    def save_poem(self):
        """Saves the generated poem to a JSON file."""
        if not self.poem_text:
            return
            
        # Create the 'poems' directory if it doesn't exist
        poems_dir = "poems"
        if not os.path.exists(poems_dir):
            os.makedirs(poems_dir)

        # Get current timestamp
        now = datetime.now()
        timestamp = now.isoformat()

        # Count words and lines
        words = len(self.poem_text.split())
        lines = len(self.poem_text.strip().split('\n'))

        # Create the data dictionary
        poem_data = {
            "timestamp": timestamp,
            "topic": self.topic,
            "words": words,
            "lines": lines,
            "text": self.poem_text
        }
        
        # Generate filename from timestamp
        filename = os.path.join(poems_dir, f"poem_{now.strftime('%Y%m%d_%H%M%S')}.json")

        # Write the data to a JSON file
        with open(filename, 'w') as f:
            json.dump(poem_data, f, indent=2)
        
        self.status_message = f"Poem saved to {filename}"
        print(self.status_message)

    def get_user_choice(self):
        """Asks the user to choose between writing their own poem or generating one."""
        print("\n--- Hand Poem Creator ---")
        print("Choose an option:")
        print("1. Enter your own poem")
        print("2. Generate a new poem")
        choice = input("Enter your choice (1 or 2): ")
        
        if choice == '1':
            user_input = input("Enter your poem (use '\\n' for new lines): ")
            self.user_poem = user_input.replace('\\n', '\n')
            self.status_message = "Your poem is loaded. Find the box to display it."
        elif choice == '2':
            self.user_poem = None
            self.status_message = "Move hands to form a box and press 'p' to generate a poem."
        else:
            print("Invalid choice. Defaulting to generating a new poem.")
            self.user_poem = None
            self.status_message = "Invalid choice. Defaulting to generating a new poem. Move hands to form a box and press 'p' to generate a poem."

    def run(self):
        """Main application loop."""
        
        # Get user choice before starting the camera
        self.get_user_choice()

        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set the window to fullscreen
        cv2.namedWindow('Hand Poem Creator', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Hand Poem Creator', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("Hand Poem Creator Started!")
        
        is_box_present = False

        if self.user_poem:
            self.poem_text = self.user_poem

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Calculate FPS
            if self.show_fps:
                self.calculate_fps()
            
            # Process hands
            box_coords, frame = self.process_hands(frame)
            
            is_box_present = box_coords is not None
            
            if is_box_present:
                # Stabilize bounding box
                stabilized_coords = self.stabilize_box(box_coords)
                min_x, min_y, max_x, max_y = stabilized_coords
                
                box_width = max_x - min_x
                box_height = max_y - min_y
                
                # Draw enhanced bounding box
                if box_width > 10 and box_height > 10:  # Minimum size threshold
                    self.draw_enhanced_box(frame, stabilized_coords, box_width, box_height)
                    self.status_message = "Box is ready. Press 'p' to generate."
            else:
                self.status_message = "Move both hands into view to form a box."
            
            # Draw UI panel
            frame = self.draw_ui_panel(frame)
            
            # Display frame
            cv2.imshow('Hand Poem Creator', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Key 'p' to generate poem
            if key == ord('p') and is_box_present:
                if self.user_poem:
                    self.status_message = "Your poem is already loaded."
                else:
                    self.status_message = "Generating poem..."
                    try:
                        # Make the prompt unique with a timestamp
                        unique_prompt = f"Write a unique short poem about {self.topic} based on a hand gesture. Current time: {datetime.now().isoformat()}"
                        response = self.gemini_model.generate_content(unique_prompt)
                        self.poem_text = response.text
                        self.poem_generated_count += 1
                        self.status_message = "Poem ready!"
                        # Save the generated poem
                        self.save_poem()
                    except genai.APIError as e:
                        self.poem_text = ""
                        self.status_message = f"API Error: {e}"
                    except Exception as e:
                        self.poem_text = ""
                        self.status_message = f"Error: {e}"
            
            # Key 'q' to quit
            if key == ord('q'):
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("Hand Poem Creator closed successfully!")

if __name__ == "__main__":
    app = HandPoemCreator()
    app.run()
