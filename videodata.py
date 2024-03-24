import numpy as np
import math
import cv2
import copy
import pytube
from moviepy.editor import VideoFileClip
from abc import ABC, abstractmethod
import pandas as pd
from utils import create_folder_if_not_exists
from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


class VideoProcessor(ABC):

    def __init__(self):
        self.df_points = pd.DataFrame({i: [] for i in range(37)})

    def __call__(self, video: pytube.YouTube):
        self.video = video
        self.title = video.title
        try:
            self.video_source = self.video.streams.first().download("sources/")
        except Exception as e:
            print(e, "Error downloading video. Check the URL and your connection.")
            quit()

        self.clip = VideoFileClip(self.video_source)
        self.timestamps = [round(tstamp * 1000) for tstamp,
                           frame in self.clip.iter_frames(with_times=True)]
        self.total_frames = len(self.timestamps)
        self.frame_count = 0
        self._process_video()

    def _process_video(self):
        # Process the video frame by frame
        output_clip = self.clip.fl_image(self._process_frame)
        output_clip.write_videofile(
            f"outputs/final-{self.title}.mp4", fps=self.clip.fps)
        print(f"Output video saved as outputs/{self.title}.mp4")
        audio_clip = output_clip.audio
        audio_dir = "audio/"
        audio_clip.write_audiofile(audio_dir + f"{self.title}.mp3")
        print(f"Audio clip saved as audio/{self.title}.mp3")
        output_clip.close()
        self.frame_count = 0

        create_folder_if_not_exists(f"final-{self.title}")
        with open("logs.txt", "w") as file:
            file.write(f"final-{self.title}")

        self.df_points.to_hdf(
            f"data/final-{self.title}/videodata.hdf5", key="videodata")
        print(
            f"data saved in: data/final-{self.title}/videodata.hdf5")

    @abstractmethod
    def _process_frame():
        pass


class VideoSegmenter(VideoProcessor):

    BG_COLOR = (192, 192, 192)  # gray
    MASK_COLOR = (255, 255, 255)  # white

    def __init__(self, segmenter: vision.ImageSegmenter):
        super(VideoSegmenter, self).__init__()
        self.segmenter = segmenter

    def _process_frame(self, frame_data: np.ndarray):
        if self.frame_count < self.total_frames:
            pred_img = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=frame_data)
            segmentation_result = self.segmenter.segment_for_video(
                pred_img, self.timestamps[self.frame_count])
            category_mask = segmentation_result.category_mask
            self.frame_count += 1

            # Generate output frame data (where condition is met, etc.)
            fg_image = np.zeros(frame_data.shape, dtype=np.uint8)
            fg_image[:] = self.MASK_COLOR
            bg_image = np.zeros(frame_data.shape, dtype=np.uint8)
            bg_image[:] = self.BG_COLOR

            condition = np.stack(
                (category_mask.numpy_view(),) * 3, axis=-1) > 0.1
            output_frame = np.where(condition, fg_image, frame_data)
            return output_frame
        else:
            return frame_data


class VideoPoser(VideoProcessor):

    def __init__(self, poser: vision.ImageSegmenter, yolo):
        super(VideoPoser, self).__init__()
        self.poser = poser
        self.yolo = yolo

    def _process_frame(self, frame_data: np.ndarray):
        if self.frame_count < self.total_frames:
            height, width = frame_data.shape[:2]
            curr_timestamp = self.timestamps[self.frame_count]
            
            pred_img = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=frame_data)
            landmarks = self.poser.detect_for_video(
                pred_img, curr_timestamp)
            self.frame_count += 1

            # Annotate Frame
            output_frame = np.copy(frame_data)
            poses = landmarks.pose_landmarks
            if len(poses) > 0:
                pose = poses[0]
                lm_builder = landmark_pb2.NormalizedLandmarkList()
                lm_builder.landmark.extend([landmark_pb2.NormalizedLandmark(
                    x=lmd.x, y=lmd.y, z=lmd.z) for lmd in pose])
                    
                bbox = self.yolo.track(output_frame)
                bbox = bbox[0]
                x1, y1, x2, y2 = None, None, None, None
                if len(bbox.boxes.xyxy) > 0:
                    x1, y1, x2, y2 = bbox.boxes.xyxy[0].tolist()
                    x1, y1, x2, y2 = x1 / width, y1 / height, x2 / width, y2 / height 

                    color = (0, 255, 0)  # Green
                    thickness = 2
                    cv2.rectangle(output_frame, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), color, thickness)

                solutions.drawing_utils.draw_landmarks(
                    output_frame,
                    lm_builder,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style())
                
                self.df_points.loc[curr_timestamp] = list(lm_builder.landmark) + [x1, y1, x2, y2]
            else:
                self.df_points.loc[curr_timestamp] = [None for _ in range(37)]
            return output_frame
        else:
            return frame_data


if __name__ == "__main__":
    mode = 'POSE'  # SEGMENT or POSE
    if mode == 'SEGMENT':
        # Create the options that will be used for InteractiveSegmenter
        base_options = python.BaseOptions(
            model_asset_path='models/selfie_multiclass_256x256.tflite')
        options = vision.ImageSegmenterOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO,
                                               output_category_mask=True)
        segmenter = vision.ImageSegmenter.create_from_options(options)

        processor = VideoSegmenter(segmenter)

    elif mode == 'POSE':
        base_options = python.BaseOptions(
            model_asset_path='models/pose_landmarker_heavy.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False)
        detector = vision.PoseLandmarker.create_from_options(options)
        yolo = YOLO("YOLOv9/best.pt")
        processor = VideoPoser(detector, yolo)

    else:
        quit("Invalid Mode")

    youtube_url = input("Enter YouTube video URL: ")
    # https://www.youtube.com/watch?v=cs9FbcqSKSg&ab_channel=FortniteCentral
    video = pytube.YouTube(youtube_url)
    # video = VideoFileClip("")
    print(f"Found: {video.title}")

    processor(video)
