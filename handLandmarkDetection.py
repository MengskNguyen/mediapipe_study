import mediapipe as mp
import cv2

# =========================
# 1. Load model file
# =========================
model_path = 'hand_landmark_webcam.task'

# =========================
# 2. Mở webcam (camera index 0 = camera mặc định)
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# =========================
# 3. Khai báo MediaPipe API shortcuts
# =========================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# =========================
# 4. Định nghĩa các kết nối giữa các điểm trên bàn tay
# (21 landmarks → nối thành skeleton)
# =========================
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # ngón cái
    (0,5),(5,6),(6,7),(7,8),          # ngón trỏ
    (5,9),(9,10),(10,11),(11,12),     # ngón giữa
    (9,13),(13,14),(14,15),(15,16),   # ngón áp út
    (13,17),(17,18),(18,19),(19,20),  # ngón út
    (0,17)                            # lòng bàn tay
]

# =========================
# 5. Cấu hình model HandLandmarker
# =========================
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,  # dùng cho video/webcam
    num_hands=2                            # detect tối đa 2 tay
)

# =========================
# 6. Khởi tạo detector (landmarker)
# =========================
with HandLandmarker.create_from_options(options) as landmarker:

    # timestamp dùng để sync frame trong video mode
    timestamp = 0

    # =========================
    # 7. Vòng lặp đọc webcam liên tục
    # =========================
    while True:

        # Đọc 1 frame từ camera
        ret, frame = cap.read()

        if not ret:
            print("Cannot read camera feed")
            break

        # Lật ngang ảnh (giống gương selfie)
        frame = cv2.flip(frame, 1)

        # =========================
        # 8. Convert BGR (OpenCV) → RGB (MediaPipe)
        # =========================
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # =========================
        # 9. Chuyển frame sang MediaPipe Image object
        # =========================
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # =========================
        # 10. Detect hand landmarks
        # =========================
        result = landmarker.detect_for_video(mp_image, timestamp)

        # Lấy kích thước frame để convert tọa độ (0-1 → pixel)
        h, w, _ = frame.shape

        # =========================
        # 11. Vẽ kết quả nếu detect được tay
        # =========================
        if result.hand_landmarks:

            # duyệt từng bàn tay
            for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):

                # ---------
                # Vẽ điểm landmarks (21 điểm)
                # ---------
                for lm in hand_landmarks:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)

                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                # ---------
                # Vẽ đường nối giữa các landmarks
                # ---------
                for start_idx, end_idx in HAND_CONNECTIONS:
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]

                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)

                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # ---------
                # Hiển thị Left / Right hand
                # ---------
                handedness = result.handedness[hand_idx][0].category_name

                # lấy vị trí landmark 0 (wrist) để đặt text
                x = int(hand_landmarks[0].x * w)
                y = int(hand_landmarks[0].y * h) - 20

                cv2.putText(
                    frame,
                    handedness,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

        # =========================
        # 12. Hiển thị output
        # =========================
        cv2.imshow("Hand Detection", frame)

        # tăng timestamp ~33ms/frame (≈30 FPS)
        timestamp += 33

        # nhấn q để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# =========================
# 13. Giải phóng tài nguyên
# =========================
cap.release()
cv2.destroyAllWindows()