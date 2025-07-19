import cv2
import time
import mediapipe as mp

holistic_module = mp.solutions.holistic
holistic_detector = holistic_module.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

drawing_utils = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

prev_time = 0
curr_time = 0
pose_landmarks_data = []

cv2.namedWindow("Landmarks Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Landmarks Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while camera.isOpened():
    is_frame_valid, frame = camera.read()
    if not is_frame_valid:
        break

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_image.flags.writeable = False
    model_results = holistic_detector.process(rgb_image)
    rgb_image.flags.writeable = True
    output_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    if model_results.face_landmarks:
        drawing_utils.draw_landmarks(
            output_image,
            model_results.face_landmarks,
            holistic_module.FACEMESH_CONTOURS,
            drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
            drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
        )

    if model_results.right_hand_landmarks:
        drawing_utils.draw_landmarks(
            output_image,
            model_results.right_hand_landmarks,
            holistic_module.HAND_CONNECTIONS
        )

    if model_results.left_hand_landmarks:
        drawing_utils.draw_landmarks(
            output_image,
            model_results.left_hand_landmarks,
            holistic_module.HAND_CONNECTIONS
        )

    pose_landmarks_data.clear()
    if model_results.pose_landmarks:
        drawing_utils.draw_landmarks(
            output_image,
            model_results.pose_landmarks,
            holistic_module.POSE_CONNECTIONS
        )
        for idx, landmark in enumerate(model_results.pose_landmarks.landmark):
            if landmark.visibility > 0.5:
                pose_landmarks_data.append(
                    (mp.solutions.pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z)
                )

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(output_image, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Landmarks Detection", output_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

print("Pose Landmarks Detected:")
for name, x, y, z in pose_landmarks_data:
    print(f"{name}: (x={x}, y={y}, z={z})")

print("Hand Landmarks for Reference:")
for landmark in mp.solutions.hands.HandLandmark:
    print(f"{landmark} : {landmark.value}")
