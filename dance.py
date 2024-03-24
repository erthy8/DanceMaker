# pip install mediapipe opencv-python
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

from mediapipe.framework.formats import landmark_pb2


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


def delay(image):
    cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
    cv2.putText(image, 'Starting in ....', (15, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, str(3),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    time.sleep(1)
    cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
    cv2.putText(image, str(2),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    time.sleep(1)
    cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
    cv2.putText(image, str(1),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    time.sleep(1)
    cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
    cv2.putText(image, str(0),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
    time.sleep(1)
    cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
    cv2.putText(image, "Dance Now!",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)


cap = cv2.VideoCapture(0)

df_points = pd.DataFrame({i: [] for i in range(32)})


# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    _, startFrame = cap.read()
    delay(startFrame)  # pauses program and gives the user a 3,2,1 countdown

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()

        curr_timestamp = int((time.time() - start_time()) * 1000)

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        if landmarks is not None:
            lm_builder = landmark_pb2.NormalizedLandmarkList()
            lm_builder.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks
            ])
            self.df_points.loc[curr_timestamp] = lm_builder

            # print(str(round(coords[0][1], 2)))
            for lmd in lm_list.values():
                cv2.putText(image, str(round(lmd.x, 2)) + ", " + str(round(lmd.y, 2)),
                            tuple(np.multiply([lmd.x, lmd.y], [
                                640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                                255, 255, 255), 2, cv2.LINE_AA
                            )
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(
                                          color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )
        else:
            self.df_points.loc[curr_timestamp] = [
                None for _ in range(32)]

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
