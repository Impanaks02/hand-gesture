import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

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
            # We are not drawing the full landmarks, but we still need the tips for the bounding box
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            all_finger_tip_coords.append((int(thumb_tip.x * w), int(thumb_tip.y * h)))
            all_finger_tip_coords.append((int(index_finger_tip.x * w), int(index_finger_tip.y * h)))
        
        # Now that we have all the coordinates, find the min/max across all of them
        if all_finger_tip_coords:
            min_x = min(x for x, y in all_finger_tip_coords)
            min_y = min(y for x, y in all_finger_tip_coords)
            max_x = max(x for x, y in all_finger_tip_coords)
            max_y = max(y for x, y in all_finger_tip_coords)
            box_found = True
    
    if box_found:
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        
        box_width = max_x - min_x
        box_height = max_y - min_y
        
        cv2.putText(frame, f'W: {box_width}, H: {box_height}', (min_x, min_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Hand Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()