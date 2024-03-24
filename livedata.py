# pip install mediapipe opencv-python
import cv2
import numpy as np
import pandas as pd
import time
from utils import create_folder_if_not_exists

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import threading
import soundfile as sf
import sounddevice as sd

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def play_audio_soundfile(audio_file):
    # Load the audio file
    y, sr = sf.read(audio_file)

    # Play the audio using sounddevice
    sd.play(y, sr)
    sd.wait()


def play_audio_in_thread(audio_file):
    thread = threading.Thread(target=play_audio_soundfile, args=(audio_file,))
    thread.start()


def get_live_data(title: str, df_dance):

    cap = cv2.VideoCapture(0)

    df_points = pd.DataFrame({i: [] for i in range(33)})

    audio_flag = False
    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        countdown_start_time = time.time()
        real_start_time = countdown_start_time
        countdown_time = 5
        index = 0
        countdown_elapsed_time = (time.time() - countdown_start_time)

        while cap.isOpened():
            ret, frame = cap.read()

            if countdown_elapsed_time < countdown_time:
                image = frame
                cv2.putText(image, "Starting in {} seconds".format(int(countdown_time - countdown_elapsed_time)), (10, 50), cv2.FONT_HERSHEY_PLAIN,
                            3, (15, 225, 215), 2)
                countdown_elapsed_time = (time.time() - countdown_start_time)
                real_start_time = time.time()
            else:
                if not audio_flag:
                    # Adjust the path as needed
                    audio_file = f"audio\{title}.mp3"
                    play_audio_in_thread(audio_file)
                    audio_flag = True
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
                    df_points.loc[curr_timestamp] = lm_builder.landmark

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
                else:
                    df_points.loc[curr_timestamp] = [
                        None for _ in range(33)]

                # Render detections from video
                df_dance_len = len(df_dance)
                if index < df_dance_len and curr_timestamp > df_dance.index[index]:
                    while index < df_dance_len - 1 and curr_timestamp > df_dance.index[index + 1]:
                        index += 1
                    if df_dance.iloc[index, 0] != None:
                        cur_pose_landmarks = landmark_pb2.NormalizedLandmarkList()
                        cur_pose_landmarks.landmark.extend(
                            df_dance.iloc[index].tolist())
                        solutions.drawing_utils.draw_landmarks(
                            image,
                            cur_pose_landmarks,
                            solutions.pose.POSE_CONNECTIONS,
                            solutions.drawing_styles.get_default_pose_landmarks_style())
                    index += 1

                if index >= df_dance_len:
                    break

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    create_folder_if_not_exists(str(title))
    with open("logs.txt", "w") as file:
        file.write(f"final-{title}")

    df_points.to_hdf(
        f"data/{title}/livedata.hdf5", key="livedata")
    print(f"data saved in: data/final-{title}/livedata.hdf5")


if __name__ == "__main__":
    title = "The Robot- Fortnite Emote"
    df_dance = pd.read_hdf(
        './data/final-The Robot- Fortnite Emote/videodata.hdf5', key='videodata')
    get_live_data(title, df_dance)
