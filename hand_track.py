import cv2
import mediapipe as mp
import numpy as np
import time

class HandTracker:
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
        
        # UI Configuration
        self.colors = {
            'box': (0, 255, 255),  # Cyan for a modern look
            'glow': (0, 255, 255),
            'text': (255, 255, 255), # White for high contrast
            'background': (20, 20, 20),
            'landmarks': (0, 255, 255),
            'connections': (255, 0, 255)
        }
        
        # State variables
        self.show_landmarks = False
        self.show_fps = True
        self.stabilization_buffer = []
        self.buffer_size = 5
        
        # FPS calculation
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
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
        
        if len(self.stabilization_buffer) == 0:
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
        cv2.putText(frame, "Enhanced Hand Tracker", (20, 45), 
                    cv2.FONT_HERSHEY_COMPLEX, 1.2, self.colors['box'], 2)
        
        # Instructions and info
        instructions = [
            "Controls: 'l' - Toggle landmarks | 'f' - Toggle FPS | 'r' - Reset | 'q' - Quit",
            f"FPS: {self.current_fps:.1f}" if self.show_fps else "",
            f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}"
        ]
        
        y_offset = 80
        for instruction in instructions:
            if instruction:
                cv2.putText(frame, instruction, (20, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
                y_offset += 25
        
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
        
        # Information box (now aligned to the top of the box)
        info_x, info_y = min_x, max(min_y - 60, 130)
        
        # Draw a semi-transparent background for the info text
        info_overlay = frame.copy()
        cv2.rectangle(info_overlay, (info_x, info_y), (info_x + 220, info_y + 40), (0, 0, 0), -1)
        cv2.addWeighted(info_overlay, 0.6, frame, 0.4, 0, frame)

        # Information text
        cv2.putText(frame, f'Width: {box_width}px', (info_x + 5, info_y + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        cv2.putText(frame, f'Height: {box_height}px', (info_x + 5, info_y + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        
        # Center point
        center_x, center_y = (min_x + max_x) // 2, (min_y + max_y) // 2
        cv2.circle(frame, (center_x, center_y), 5, self.colors['box'], -1)
    
    def process_hands(self, frame):
        """Process hand detection and return bounding box coordinates based on finger tips"""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None, frame
        
        all_relevant_tip_coords = []
        
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks if enabled
            if self.show_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Get only the coordinates for the thumb and index finger tips
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Convert to pixel coordinates and add to list
            all_relevant_tip_coords.append((int(thumb_tip.x * w), int(thumb_tip.y * h)))
            all_relevant_tip_coords.append((int(index_finger_tip.x * w), int(index_finger_tip.y * h)))
            
            # Draw finger tip markers
            cv2.circle(frame, all_relevant_tip_coords[-2], 8, (255, 0, 0), -1)  # Blue for thumb
            cv2.circle(frame, all_relevant_tip_coords[-1], 8, (0, 0, 255), -1)  # Red for index
        
        if all_relevant_tip_coords:
            # Calculate bounding box from only the selected finger tips
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
        
        print("Hand Tracker Started!")
        
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
            
            if box_coords:
                # Stabilize bounding box
                stabilized_coords = self.stabilize_box(box_coords)
                min_x, min_y, max_x, max_y = stabilized_coords
                
                box_width = max_x - min_x
                box_height = max_y - min_y
                
                # Draw enhanced bounding box
                if box_width > 10 and box_height > 10:  # Minimum size threshold
                    self.draw_enhanced_box(frame, stabilized_coords, box_width, box_height)
            
            # Draw UI panel
            frame = self.draw_ui_panel(frame)
            
            # Display frame
            cv2.imshow('Enhanced Hand Tracker', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
            elif key == ord('f'):
                self.show_fps = not self.show_fps
            elif key == ord('r'):
                self.stabilization_buffer.clear()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("Hand Tracker closed successfully!")

if __name__ == "__main__":
    tracker = HandTracker()
    tracker.run()
