import cv2
import mediapipe as mp
import google.generativeai as genai
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap
from dotenv import load_dotenv
load_dotenv()

api_key=os.getenv("API_KEY")
# --- Gemini API Configuration ---
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# --- MediaPipe and OpenCV Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)
cap = cv2.VideoCapture(0)

# --- Global variables for poem and font ---
poem_text = "Press 'p' to generate a poem!"
poem_generated = False
base_font_path = "arial.ttf"

def get_new_poem(words, lines, topic="nature"):
    prompt = (f"Write a short, creative poem about {topic}. "
              f"Keep it to approximately {words} words and around {lines} lines.")
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "Error generating poem. Check API key and internet connection."

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    min_x, min_y = w, h
    max_x, max_y = 0, 0
    box_found = False

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 1:
        all_finger_tip_coords = []
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            all_finger_tip_coords.append((int(thumb_tip.x * w), int(thumb_tip.y * h)))
            all_finger_tip_coords.append((int(index_finger_tip.x * w), int(index_finger_tip.y * h)))
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        if all_finger_tip_coords:
            min_x = min(x for x, y in all_finger_tip_coords)
            min_y = min(y for x, y in all_finger_tip_coords)
            max_x = max(x for x, y in all_finger_tip_coords)
            max_y = max(y for x, y in all_finger_tip_coords)
            box_found = True

    if box_found and max_x > min_x and max_y > min_y:
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        
        box_width = max_x - min_x
        box_height = max_y - min_y

        if poem_generated:
            # --- DYNAMIC FONT SIZING AND RENDERING ---
            
            # 1. Start with a reasonable font size
            dynamic_font_size = 30
            
            # 2. Iterate to find the best-fitting font size
            # We'll use a tolerance of a few pixels to prevent infinite loops
            text_area_width = box_width - 20
            
            # Loop to find the best font size
            # A hardcoded limit to prevent crashing on very small boxes
            for i in range(100): 
                try:
                    font = ImageFont.truetype(base_font_path, dynamic_font_size)
                    avg_char_width = font.getlength('A')
                    max_chars = int(text_area_width / avg_char_width)
                    
                    if max_chars <= 0: # Ensure we have at least one character
                        max_chars = 1
                    
                    wrapped_lines = textwrap.wrap(poem_text, width=max_chars)
                    
                    total_text_height = len(wrapped_lines) * font.getbbox('A')[3] * 1.2
                    
                    # If the text fits vertically, we can stop
                    if total_text_height < box_height:
                        break
                    
                    dynamic_font_size -= 1 # Reduce font size if it doesn't fit
                except (IOError, ValueError):
                    # Handle font errors or division by zero
                    font = ImageFont.load_default()
                    break

            # 3. Once the font size is found, draw the text
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            padding = 10
            
            if text_area_width > 0:
                text_y = min_y + padding
                for line in wrapped_lines:
                    draw.text((min_x + padding, text_y), line, font=font, fill=(0, 255, 0))
                    text_y += int(dynamic_font_size * 1.2)

            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    cv2.imshow('Hand Poem Creator', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p') and box_found and not poem_generated:
        box_width = max_x - min_x
        box_height = max_y - min_y
        
        words_per_line = max(1, int(box_width / 50))
        num_lines = max(1, int(box_height / 30))
        
        print(f"Generating poem with ~{words_per_line} words per line and ~{num_lines} lines...")
        poem_text = get_new_poem(words_per_line * num_lines, num_lines)
        poem_generated = True
        print("Poem generated.")


hands.close()
cap.release()
cv2.destroyAllWindows()