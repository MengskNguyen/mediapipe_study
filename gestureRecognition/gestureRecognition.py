import mediapipe as mp
import cv2

# =========================
# 1. Load gesture model
# =========================
model_path = "gesture_recognizer.task"

# =========================
# 2. Open webcam
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# =========================
# 3. MediaPipe shortcuts
# =========================
BaseOptions = mp.tasks.BaseOptions

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions

VisionRunningMode = mp.tasks.vision.RunningMode

# =========================
# 4. Hand skeleton connections
# =========================
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# =========================
# 5. Gesture recognizer config
# =========================
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# =========================
# 6. Create recognizer
# =========================
with GestureRecognizer.create_from_options(options) as recognizer:

    timestamp = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Cannot read camera feed")
            break

        # mirror effect
        frame = cv2.flip(frame, 1)

        # =========================
        # 7. BGR -> RGB
        # =========================
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # =========================
        # 8. Convert to MediaPipe Image
        # =========================
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # =========================
        # 9. Gesture recognition
        # =========================
        result = recognizer.recognize_for_video(
            mp_image,
            timestamp
        )

        h, w, _ = frame.shape

        # =========================
        # 10. Draw results
        # =========================
        if result.hand_landmarks:

            for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):

                # -----------------
                # Draw landmarks
                # -----------------
                for lm in hand_landmarks:

                    cx = int(lm.x * w)
                    cy = int(lm.y * h)

                    cv2.circle(
                        frame,
                        (cx, cy),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # -----------------
                # Draw skeleton
                # -----------------
                for start_idx, end_idx in HAND_CONNECTIONS:

                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]

                    x1 = int(start.x * w)
                    y1 = int(start.y * h)

                    x2 = int(end.x * w)
                    y2 = int(end.y * h)

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        2
                    )

                # -----------------
                # Handedness
                # -----------------
                handedness = result.handedness[hand_idx][0].category_name

                # -----------------
                # Gesture name
                # -----------------
                gesture_name = "Unknown"

                if result.gestures:
                    if len(result.gestures[hand_idx]) > 0:
                        gesture_name = result.gestures[hand_idx][0].category_name

                # -----------------
                # Text position
                # -----------------
                x = int(hand_landmarks[0].x * w)
                y = int(hand_landmarks[0].y * h) - 20

                # -----------------
                # Draw text
                # -----------------
                cv2.putText(
                    frame,
                    f"{handedness} | {gesture_name}",
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

        # =========================
        # 11. Show output
        # =========================
        cv2.imshow("Gesture Recognition", frame)

        # ~30 FPS
        timestamp += 33

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# =========================
# 12. Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()