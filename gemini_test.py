import cv2
import mediapipe as mp
import google.generativeai as genai
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap
from dotenv import load_dotenv
import time
import threading
import json
from datetime import datetime

# Load environment variables
load_dotenv()

class HandPoemCreator:
    def __init__(self):
        # API Configuration
        self.api_key = os.getenv("API_KEY")
        if not self.api_key:
            raise ValueError("API_KEY not found in environment variables")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
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
        
        # UI Configuration
        self.colors = {
            'box': (0, 255, 0),
            'text': (255, 255, 255),
            'poem_text': (0, 255, 0),
            'background': (0, 0, 0),
            'accent': (255, 165, 0),
            'error': (0, 0, 255)
        }
        
        # State variables
        self.poem_text = "Position your hand and press 'p' to generate a poem!"
        self.poem_generated = False
        self.poem_topics = ["nature", "love", "dreams", "adventure", "friendship", "ocean", "mountains", "stars"]
        self.current_topic_index = 0
        self.show_landmarks = False
        self.generating_poem = False
        self.last_generation_time = 0
        self.generation_cooldown = 3  # seconds
        
        # Font configuration
        self.font_paths = ["arial.ttf", "Arial.ttf", "/System/Library/Fonts/Arial.ttf", 
                          "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
        self.base_font_path = self.find_font()
        
        # Stabilization
        self.stabilization_buffer = []
        self.buffer_size = 3
        
        # Statistics
        self.poems_generated = 0
        self.saved_poems = []
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
    
    def find_font(self):
        """Find available font file"""
        for font_path in self.font_paths:
            if os.path.exists(font_path):
                return font_path
        return None  # Will use default font
    
    def calculate_fps(self):
        """Calculate current FPS"""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            end_time = time.time()
            self.current_fps = self.fps_counter / (end_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def stabilize_box(self, box_coords):
        """Stabilize bounding box coordinates"""
        self.stabilization_buffer.append(box_coords)
        if len(self.stabilization_buffer) > self.buffer_size:
            self.stabilization_buffer.pop(0)
        
        if len(self.stabilization_buffer) == 0:
            return box_coords
        
        avg_coords = np.mean(self.stabilization_buffer, axis=0)
        return tuple(map(int, avg_coords))
    
    def get_new_poem_async(self, words, lines, topic):
        """Generate poem in background thread"""
        def generate():
            try:
                self.generating_poem = True
                prompt = (f"Write a beautiful, creative poem about {topic}. "
                         f"Keep it to approximately {words} words and {lines} lines. "
                         f"Make it inspiring and emotionally resonant.")
                
                response = self.model.generate_content(prompt)
                self.poem_text = response.text.strip()
                self.poems_generated += 1
                self.last_generation_time = time.time()
                
                # Save poem
                poem_data = {
                    "timestamp": datetime.now().isoformat(),
                    "topic": topic,
                    "words": words,
                    "lines": lines,
                    "text": self.poem_text
                }
                self.saved_poems.append(poem_data)
                
                print(f"Poem #{self.poems_generated} generated successfully!")
                
            except Exception as e:
                print(f"Gemini API error: {e}")
                self.poem_text = f"Error generating poem about {topic}.\nCheck API key and internet connection.\nTry again in a moment."
            finally:
                self.generating_poem = False
                self.poem_generated = True
        
        thread = threading.Thread(target=generate)
        thread.daemon = True
        thread.start()
    
    def draw_ui_panel(self, frame):
        """Draw comprehensive UI panel"""
        h, w = frame.shape[:2]
        panel_height = 140
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), self.colors['background'], -1)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)
        
        # Title with accent
        cv2.putText(frame, "Hand Poem Creator", (10, 30), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, self.colors['accent'], 3)
        
        # Current topic
        current_topic = self.poem_topics[self.current_topic_index]
        cv2.putText(frame, f"Topic: {current_topic.title()}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
        
        # Statistics
        cv2.putText(frame, f"Poems Generated: {self.poems_generated}", (250, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
        
        # Status
        if self.generating_poem:
            status_text = "Generating poem..."
            status_color = self.colors['accent']
        elif self.poem_generated and not self.generating_poem:
            status_text = "Poem ready! Move hand to see it."
            status_color = self.colors['poem_text']
        else:
            status_text = "Position hand and press 'p' to generate"
            status_color = self.colors['text']
        
        cv2.putText(frame, f"Status: {status_text}", (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Controls
        controls = [
            "'p' - Generate | 't' - Change topic | 'l' - Landmarks | 's' - Save poems | 'c' - Clear | 'q' - Quit"
        ]
        
        cv2.putText(frame, controls[0], (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # FPS and cooldown
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(frame, fps_text, (w - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
        
        # Generation cooldown
        time_since_last = time.time() - self.last_generation_time
        if time_since_last < self.generation_cooldown:
            cooldown_remaining = self.generation_cooldown - time_since_last
            cv2.putText(frame, f"Cooldown: {cooldown_remaining:.1f}s", (w - 150, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['accent'], 1)
        
        return frame
    
    def render_poem_in_box(self, frame, box_coords):
        """Render poem text within bounding box with enhanced styling"""
        min_x, min_y, max_x, max_y = box_coords
        box_width = max_x - min_x
        box_height = max_y - min_y
        
        if box_width < 50 or box_height < 30:
            return frame
        
        # Dynamic font sizing with better algorithm
        dynamic_font_size = min(100, max(12, int(box_width / 8)))
        
        font = None
        wrapped_lines = []
        
        for attempt in range(20):  # More attempts for better fitting
            try:
                if self.base_font_path:
                    font = ImageFont.truetype(self.base_font_path, dynamic_font_size)
                else:
                    font = ImageFont.load_default()
                
                # Calculate text metrics
                test_text = "A"
                if hasattr(font, 'getlength'):
                    avg_char_width = font.getlength(test_text)
                else:
                    avg_char_width = font.getsize(test_text)[0]
                
                if avg_char_width == 0:
                    max_chars = 10
                else:
                    max_chars = max(5, int((box_width - 40) / avg_char_width))
                
                wrapped_lines = textwrap.wrap(self.poem_text, width=max_chars)
                
                # Calculate total height
                if hasattr(font, 'getbbox'):
                    line_height = font.getbbox(test_text)[3] + 6
                else:
                    line_height = font.getsize(test_text)[1] + 6
                
                total_text_height = len(wrapped_lines) * line_height
                
                if total_text_height <= box_height - 40:  # Leave more padding
                    break
                
                dynamic_font_size -= 2
                
            except Exception as e:
                print(f"Font error: {e}")
                font = ImageFont.load_default()
                wrapped_lines = textwrap.wrap(self.poem_text, width=20)
                line_height = 20
                break
        
        # Convert to PIL for better text rendering
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Draw semi-transparent background for poem
        poem_bg_coords = [
            (min_x + 5, min_y + 5),
            (max_x - 5, max_y - 5)
        ]
        
        # Create overlay for poem background
        overlay_img = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay_img)
        overlay_draw.rectangle(poem_bg_coords, fill=(0, 0, 0, 120))
        
        # Composite the overlay
        pil_img = Image.alpha_composite(pil_img.convert('RGBA'), overlay_img).convert('RGB')
        draw = ImageDraw.Draw(pil_img)
        
        # Draw poem text with shadow effect
        padding = 20
        text_y = min_y + padding
        
        for line in wrapped_lines:
            if text_y + line_height > max_y - padding:
                break
            
            # Shadow
            draw.text((min_x + padding + 2, text_y + 2), line, 
                     font=font, fill=(0, 0, 0))
            # Main text
            draw.text((min_x + padding, text_y), line, 
                     font=font, fill=(0, 255, 0))
            text_y += int(line_height)
        
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def draw_enhanced_box(self, frame, box_coords):
        """Draw enhanced bounding box with animations"""
        min_x, min_y, max_x, max_y = box_coords
        box_width = max_x - min_x
        box_height = max_y - min_y
        
        # Animated border effect
        timestamp = time.time()
        pulse = int(abs(np.sin(timestamp * 3) * 10))
        
        # Main border
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), self.colors['box'], 3)
        
        # Pulsing corners
        corner_size = 15 + pulse
        corners = [(min_x, min_y), (max_x, min_y), (min_x, max_y), (max_x, max_y)]
        
        for corner in corners:
            cv2.circle(frame, corner, corner_size // 2, self.colors['box'], -1)
            cv2.circle(frame, corner, corner_size // 2, self.colors['accent'], 2)
        
        # Center crosshair
        center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2
        cross_size = 10
        cv2.line(frame, (center_x - cross_size, center_y), 
                (center_x + cross_size, center_y), self.colors['accent'], 2)
        cv2.line(frame, (center_x, center_y - cross_size), 
                (center_x, center_y + cross_size), self.colors['accent'], 2)
        
        return frame
    
    def process_hands(self, frame):
        """Process hand detection"""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None, frame
        
        all_finger_tip_coords = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            if self.show_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Get finger tips
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            thumb_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_coords = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
            
            all_finger_tip_coords.extend([thumb_coords, index_coords])
            
            # Finger tip markers
            cv2.circle(frame, thumb_coords, 8, (255, 0, 0), -1)
            cv2.circle(frame, index_coords, 8, (0, 0, 255), -1)
        
        if all_finger_tip_coords:
            min_x = min(x for x, y in all_finger_tip_coords)
            min_y = min(y for x, y in all_finger_tip_coords)
            max_x = max(x for x, y in all_finger_tip_coords)
            max_y = max(y for x, y in all_finger_tip_coords)
            
            # Add padding
            padding = 30
            min_x = max(0, min_x - padding)
            min_y = max(150, min_y - padding)
            max_x = min(w, max_x + padding)
            max_y = min(h, max_y + padding)
            
            return (min_x, min_y, max_x, max_y), frame
        
        return None, frame
    
    def save_poems_to_file(self):
        """Save all generated poems to JSON file"""
        if not self.saved_poems:
            print("No poems to save!")
            return
        
        filename = f"poems_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.saved_poems, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(self.saved_poems)} poems to {filename}")
        except Exception as e:
            print(f"Error saving poems: {e}")
    
    def run(self):
        """Main application loop"""
        cap = cv2.VideoCapture(0)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Hand Poem Creator Started!")
        print("Controls:")
        print("  'p' - Generate poem")
        print("  't' - Change topic")
        print("  'l' - Toggle landmarks")
        print("  's' - Save poems to file")
        print("  'c' - Clear current poem")
        print("  'q' - Quit")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            self.calculate_fps()
            
            # Process hands
            box_coords, frame = self.process_hands(frame)
            
            if box_coords:
                # Stabilize coordinates
                stabilized_coords = self.stabilize_box(box_coords)
                min_x, min_y, max_x, max_y = stabilized_coords
                
                box_width = max_x - min_x
                box_height = max_y - min_y
                
                # Only process if box is reasonable size
                if box_width > 50 and box_height > 30:
                    # Draw enhanced bounding box
                    frame = self.draw_enhanced_box(frame, stabilized_coords)
                    
                    # Render poem if generated
                    if self.poem_generated and not self.generating_poem:
                        frame = self.render_poem_in_box(frame, stabilized_coords)
            
            # Draw UI panel
            frame = self.draw_ui_panel(frame)
            
            # Display frame
            cv2.imshow('Hand Poem Creator', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p') and box_coords and not self.generating_poem:
                # Check cooldown
                time_since_last = time.time() - self.last_generation_time
                if time_since_last >= self.generation_cooldown:
                    min_x, min_y, max_x, max_y = box_coords
                    box_width = max_x - min_x
                    box_height = max_y - min_y
                    
                    if box_width > 50 and box_height > 30:
                        # Calculate poem parameters based on box size
                        words_per_line = max(3, min(12, int(box_width / 35)))
                        num_lines = max(2, min(8, int(box_height / 25)))
                        total_words = words_per_line * num_lines
                        
                        current_topic = self.poem_topics[self.current_topic_index]
                        print(f"Generating poem about '{current_topic}' with ~{total_words} words, {num_lines} lines...")
                        
                        self.poem_generated = False
                        self.get_new_poem_async(total_words, num_lines, current_topic)
                    else:
                        print("Hand area too small! Make a larger gesture.")
                else:
                    remaining = self.generation_cooldown - time_since_last
                    print(f"Please wait {remaining:.1f} seconds before generating another poem.")
            
            elif key == ord('t'):
                # Change topic
                self.current_topic_index = (self.current_topic_index + 1) % len(self.poem_topics)
                current_topic = self.poem_topics[self.current_topic_index]
                print(f"Topic changed to: {current_topic.title()}")
            
            elif key == ord('l'):
                # Toggle landmarks
                self.show_landmarks = not self.show_landmarks
                print(f"Hand landmarks: {'ON' if self.show_landmarks else 'OFF'}")
            
            elif key == ord('s'):
                # Save poems
                self.save_poems_to_file()
            
            elif key == ord('c'):
                # Clear current poem
                self.poem_text = "Position your hand and press 'p' to generate a poem!"
                self.poem_generated = False
                print("Current poem cleared.")
            
            elif key == ord('r'):
                # Reset everything
                self.stabilization_buffer.clear()
                self.poem_text = "Position your hand and press 'p' to generate a poem!"
                self.poem_generated = False
                self.poems_generated = 0
                self.saved_poems.clear()
                print("Application reset!")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        # Save poems on exit if any were generated
        if self.saved_poems:
            print(f"\nYou generated {len(self.saved_poems)} poems this session.")
            save_choice = input("Save poems to file? (y/n): ").lower().strip()
            if save_choice == 'y':
                self.save_poems_to_file()
        
        print("Hand Poem Creator closed successfully!")

if __name__ == "__main__":
    try:
        creator = HandPoemCreator()
        creator.run()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please make sure you have set up your API key in a .env file")
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Please check your dependencies and camera connection")