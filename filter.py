import cv2
import mediapipe as mp
import numpy as np

# INITIALIZATION

# Initialize MediaPipe Hands and Face Mesh solutions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

# Set camera resolution (increase size)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1016)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 624)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- STATE & SETTINGS ---
current_filter = 1  # Start with the first filter

# Smoothing variables
previous_coords = None
smoothing_factor = 0.7  # Higher value = more smoothing (0.0 to 1.0)
filter_transition_alpha = 0.3  # For smooth filter transitions

# Hand gesture detection variables
pinch_threshold = 40  # Distance threshold for pinch detection
previous_pinch_state = False
gesture_cooldown = 0  # Cooldown to prevent rapid filter changes

# Mouse callback function for button clicks
def mouse_callback(event, x, y, flags, param):
    global current_filter
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is on any of the filter buttons
        h, w = param['frame_shape'][:2]
        num_buttons = 5
        button_radius = 20
        button_spacing = 60
        start_x = w // 2 - (button_spacing * (num_buttons - 1)) // 2

        for i in range(1, num_buttons + 1):
            center_x = start_x + (i - 1) * button_spacing
            center_y = h - 60

            # Calculate distance from click to button center
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5

            if distance <= button_radius:
                current_filter = i
                break

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def detect_pinch_gesture(hand_landmarks, w, h):
    """Detect pinch gesture (thumb tip close to any finger tip)."""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

    # Check distance from thumb to each finger tip
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]

    thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))

    # Find the closest finger to thumb
    min_distance = float('inf')
    closest_finger_pos = None

    for finger_tip in finger_tips:
        finger = hand_landmarks.landmark[finger_tip]
        finger_pos = (int(finger.x * w), int(finger.y * h))

        distance = calculate_distance(thumb_pos, finger_pos)

        if distance < min_distance:
            min_distance = distance
            closest_finger_pos = finger_pos

    # Return pinch state and positions
    is_pinching = min_distance < pinch_threshold
    return is_pinching, thumb_pos, closest_finger_pos

def check_hand_button_interaction(hand_landmarks, w, h):
    """Check if hand is performing pinch gesture over a filter button."""
    global current_filter, previous_pinch_state, gesture_cooldown

    # Decrease cooldown
    if gesture_cooldown > 0:
        gesture_cooldown -= 1
        return

    is_pinching, thumb_pos, closest_finger_pos = detect_pinch_gesture(hand_landmarks, w, h)

    # Calculate the center point between thumb and closest finger
    center_x = (thumb_pos[0] + closest_finger_pos[0]) // 2
    center_y = (thumb_pos[1] + closest_finger_pos[1]) // 2

    # Check if pinch just started (rising edge detection)
    if is_pinching and not previous_pinch_state:
        # Check if pinch is over any filter button
        num_buttons = 5
        button_radius = 20
        button_spacing = 60
        start_x = w // 2 - (button_spacing * (num_buttons - 1)) // 2

        for i in range(1, num_buttons + 1):
            button_center_x = start_x + (i - 1) * button_spacing
            button_center_y = h - 60

            # Calculate distance from pinch center to button center
            distance = calculate_distance((center_x, center_y), (button_center_x, button_center_y))

            if distance <= button_radius + 10:  # Add some tolerance
                current_filter = i
                gesture_cooldown = 30  # Set cooldown to prevent rapid changes
                break

    previous_pinch_state = is_pinching

def smooth_coordinates(current_coords, previous_coords, smoothing_factor):
    """Apply smoothing to hand tracking coordinates."""
    if previous_coords is None:
        return current_coords

    smoothed_coords = []
    for i, (curr, prev) in enumerate(zip(current_coords, previous_coords)):
        smoothed_x = prev[0] * smoothing_factor + curr[0] * (1 - smoothing_factor)
        smoothed_y = prev[1] * smoothing_factor + curr[1] * (1 - smoothing_factor)
        smoothed_coords.append((int(smoothed_x), int(smoothed_y)))

    return smoothed_coords

# --- FILTER FUNCTIONS ---

def apply_posterize(roi):
    """Applies a posterization effect to a region of interest (ROI)."""
    if roi.size == 0: return roi
    divisor = 64
    posterized_roi = (roi // divisor) * divisor
    return posterized_roi

def apply_grayscale(roi):
    """Applies a grayscale effect."""
    if roi.size == 0: return roi
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)

def apply_negative(roi):
    """Applies a color negative/inversion effect."""
    if roi.size == 0: return roi
    return cv2.bitwise_not(roi)

def apply_threshold(roi):
    """Applies a high-contrast black and white threshold."""
    if roi.size == 0: return roi
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(thresh_roi, cv2.COLOR_GRAY2BGR)

def apply_face_landmarks(roi, frame, box_coords):
    """Overlays face landmarks within the ROI with smooth application."""
    if roi.size == 0: return roi

    x1, y1, x2, y2 = box_coords
    # Process the full frame to find face landmarks
    results_face = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Create a copy of the ROI to draw on
    overlay_roi = roi.copy()

    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                # Get landmark coordinates in pixel values
                lx, ly = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

                # Check if the landmark is inside the hand-defined box
                if x1 < lx < x2 and y1 < ly < y2:
                    # Draw a small white circle for each landmark with some transparency
                    cv2.circle(overlay_roi, (lx - x1, ly - y1), 1, (255, 255, 255), -1)

    # Blend the overlay with the original ROI for smoother effect
    blended_roi = cv2.addWeighted(roi, 0.7, overlay_roi, 0.3, 0)
    return blended_roi


# --- UI FUNCTION ---

def draw_ui(frame, current_filter_index):
    """Draws only the filter selection buttons at the bottom of the frame."""
    h, w, _ = frame.shape

    # Draw buttons only (no text)
    num_buttons = 5
    button_radius = 20
    button_spacing = 60
    start_x = w // 2 - (button_spacing * (num_buttons - 1)) // 2

    for i in range(1, num_buttons + 1):
        center_x = start_x + (i - 1) * button_spacing
        center_y = h - 60

        # Highlight the active button
        if i == current_filter_index:
            # Outer ring for highlight
            cv2.circle(frame, (center_x, center_y), button_radius + 4, (0, 255, 0), -1)

        # Draw the button with hover effect (slightly brighter when active)
        button_color = (100, 100, 100) if i == current_filter_index else (80, 80, 80)
        cv2.circle(frame, (center_x, center_y), button_radius, button_color, -1)
        cv2.circle(frame, (center_x, center_y), button_radius, (120, 120, 120), 2)

        # Button number
        text_size = cv2.getTextSize(str(i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2
        cv2.putText(frame, str(i), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_pinch_indicator(frame, hand_landmarks, w, h):
    """Draw visual indicator for pinch gesture."""
    pass


# --- MAIN LOOP ---

# Create window and set up mouse callback
window_name = 'AR Filter Demo'
cv2.namedWindow(window_name)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Set up mouse callback after window is created
    cv2.setMouseCallback(window_name, mouse_callback, {'frame_shape': (h, w)})

    # Convert the BGR image to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find hands
    results_hands = hands.process(rgb_frame)

    # Check for hand gesture interactions with filter buttons
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Check for pinch gesture to interact with buttons
            check_hand_button_interaction(hand_landmarks, w, h)

            # Draw pinch indicator if hand is making pinch gesture
            draw_pinch_indicator(frame, hand_landmarks, w, h)

    # Draw the UI on the frame
    draw_ui(frame, current_filter)

    # Apply filters based on hand detection (works with 1 or 2 hands)
    if results_hands.multi_hand_landmarks:
        if len(results_hands.multi_hand_landmarks) == 2:
            # Two hands mode - use both hands to define frame
            landmarks = []
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # We use the tips of the index finger and thumb to define the frame
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                landmarks.extend([index_tip, thumb_tip])

            # Get pixel coordinates for the four points
            coords = np.array([(lm.x * w, lm.y * h) for lm in landmarks]).astype(int)

        elif len(results_hands.multi_hand_landmarks) == 1:
            # Single hand mode - use only thumb and index finger to create a frame
            hand_landmarks = results_hands.multi_hand_landmarks[0]

            # Get only thumb and index finger tips
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert to pixel coordinates
            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))

            # Create a rectangular area using thumb and index finger as diagonal corners
            coords = np.array([thumb_pos, index_pos])

        # Apply smoothing to coordinates
        if previous_coords is not None and len(previous_coords) == len(coords):
            coords = smooth_coordinates(coords, previous_coords, smoothing_factor)
        previous_coords = coords.copy()

        # Define bounding box from the coordinates
        x1, y1 = np.min(coords, axis=0)
        x2, y2 = np.max(coords, axis=0)

        # Add some padding for smoother transitions
        padding = 10
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(w, x2 + padding), min(h, y2 + padding)

        # Check if the box has a valid area
        if x1 < x2 and y1 < y2:
            # Extract the Region of Interest (ROI)
            roi = frame[y1:y2, x1:x2]
            original_roi = roi.copy()

            # Apply the selected filter
            if current_filter == 1:
                filtered_roi = apply_posterize(roi)
            elif current_filter == 2:
                filtered_roi = apply_grayscale(roi)
            elif current_filter == 3:
                filtered_roi = apply_negative(roi)
            elif current_filter == 4:
                filtered_roi = apply_threshold(roi)
            elif current_filter == 5:
                box_coords = (x1, y1, x2, y2)
                filtered_roi = apply_face_landmarks(roi, frame, box_coords)

            # Smooth filter transition
            final_roi = cv2.addWeighted(original_roi, filter_transition_alpha,
                                      filtered_roi, 1 - filter_transition_alpha, 0)

            # Place the filtered ROI back into the frame
            frame[y1:y2, x1:x2] = final_roi

    # Display the final frame
    cv2.imshow(window_name, frame)

    # Handle user input for changing filters
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif ord('1') <= key <= ord('5'):
        current_filter = int(chr(key))

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()