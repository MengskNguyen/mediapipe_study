import mediapipe as mp
import cv2

# =========================
# 1. Load face model
# =========================
model_path = "face_landmarker.task"

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

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

VisionRunningMode = mp.tasks.vision.RunningMode

# =========================
# 4. Face Landmarker config
# =========================
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),

    running_mode=VisionRunningMode.VIDEO,

    # detect tối đa bao nhiêu khuôn mặt
    num_faces=1,

    # optional features
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)

# =========================
# 5. Create detector
# =========================
with FaceLandmarker.create_from_options(options) as landmarker:

    timestamp = 0

    while True:

        # =========================
        # 6. Read webcam frame
        # =========================
        ret, frame = cap.read()

        if not ret:
            print("Cannot read camera feed")
            break

        # mirror effect
        frame = cv2.flip(frame, 1)

        # =========================
        # 7. Convert BGR -> RGB
        # =========================
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # =========================
        # 8. Convert to MediaPipe Image
        # =========================
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # =========================
        # 9. Face landmark detection
        # =========================
        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        h, w, _ = frame.shape

        # =========================
        # 10. Draw face landmarks
        # =========================
        if result.face_landmarks:

            # loop từng khuôn mặt
            for face_landmarks in result.face_landmarks:

                # loop từng landmark
                for lm in face_landmarks:

                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    # vẽ landmark
                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

                # =========================
                # 11. Draw face bounding box
                # =========================

                xs = [lm.x for lm in face_landmarks]
                ys = [lm.y for lm in face_landmarks]

                x_min = int(min(xs) * w)
                y_min = int(min(ys) * h)

                x_max = int(max(xs) * w)
                y_max = int(max(ys) * h)

                cv2.rectangle(
                    frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    (255, 0, 0),
                    2
                )

                # =========================
                # 12. Show landmark count
                # =========================
                cv2.putText(
                    frame,
                    f"Face Mesh: {len(face_landmarks)} pts",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

        # =========================
        # 13. Show output
        # =========================
        cv2.imshow(
            "Face Landmark Detection",
            frame
        )

        # ~30 FPS
        timestamp += 33

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# =========================
# 14. Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()