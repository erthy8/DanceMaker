# pip install mediapipe opencv-python
import cv2
import numpy as np
import pandas as pd
import time
from utils import create_folder_if_not_exists

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def get_live_data(filename: str):
    cap = cv2.VideoCapture(0)

    df_points = pd.DataFrame({i: [] for i in range(32)})

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        countdown_start_time = time.time()
        real_start_time = countdown_start_time
        countdown_time = 7
        elapsed_time = (time.time() - countdown_start_time)
        while cap.isOpened():
            ret, frame = cap.read()

            if elapsed_time < countdown_time:
                image = frame
                cv2.putText(image, "Starting in {} seconds".format(int(countdown_time - elapsed_time)), (10, 50), cv2.FONT_HERSHEY_PLAIN,
                            3, (15, 225, 215), 2)
                elapsed_time = (time.time() - countdown_start_time)
                real_start_time = time.time()
            else:
                curr_timestamp = int((time.time() - real_start_time) * 1000)

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                landmarks = results.pose_landmarks
                if landmarks is not None:
                    landmarks = landmarks.landmark

                    lm_builder = landmark_pb2.NormalizedLandmarkList()
                    lm_builder.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in landmarks
                    ])
                    df_points.loc[curr_timestamp] = lm_builder

                    # print(str(round(coords[0][1], 2)))
                    visual_points = lm_builder.landmark[11:17] + \
                        lm_builder.landmark[23:29]
                    for lmd in visual_points:
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
                    df_points.loc[curr_timestamp] = [
                        None for _ in range(32)]

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    create_folder_if_not_exists(f"final-{title}")
    df_points.to_hdf(
        f"data/live-{title}/final-{title}.hdf5", key="livedata")
    print(f"data saved in: data/live-{title}.hdf5")


if __name__ == "__main__":
    title = ""
    get_live_data(title)
