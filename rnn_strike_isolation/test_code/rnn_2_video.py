import time
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import cv2
from collections import deque
import os
import json
from pathlib import Path

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is being used:", physical_devices[0])
else:
    print("No GPU found. Using CPU.")

print("Num GPUs Available:", len(physical_devices))

columns = [
    "nose_y", "nose_x",
    "left_shoulder_y", "left_shoulder_x",
    "right_shoulder_y", "right_shoulder_x",
    "left_elbow_y", "left_elbow_x",
    "right_elbow_y", "right_elbow_x",
    "left_wrist_y", "left_wrist_x",
    "right_wrist_y", "right_wrist_x",
    "left_hip_y", "left_hip_x",
    "right_hip_y", "right_hip_x",
    "left_knee_y", "left_knee_x",
    "right_knee_y", "right_knee_x",
    "left_ankle_y", "left_ankle_x",
    "right_ankle_y", "right_ankle_x"
]

class ShotCounter:
    MIN_FRAMES_BETWEEN_SHOTS = 60

    def __init__(self):
        self.nb_history = 30
        self.probs = np.zeros(4)
        self.nb_forehands = 0
        self.nb_backhands = 0
        self.nb_serves = 0
        self.last_shot = "neutral"
        self.frames_since_last_shot = self.MIN_FRAMES_BETWEEN_SHOTS
        self.results = []
        
    def update(self, probs, frame_id):
        if len(probs) == 4:
            self.probs = probs
        else:
            self.probs[0:3] = probs

        if (probs[0] > 0.98 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS):
            self.nb_backhands += 1
            self.last_shot = "backhand"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
        elif (probs[1] > 0.98 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS):
            self.nb_forehands += 1
            self.last_shot = "forehand"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
        elif (len(probs) > 3 and probs[3] > 0.98 and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS):
            self.nb_serves += 1
            self.last_shot = "serve"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})

        self.frames_since_last_shot += 1

class VideoClipMaker:
    VIDEO_FRAMES = 60
    
    def __init__(self, fps, frame_width, frame_height, folder_path):
        self.fps = fps 
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.counter = 1
        self.frame_buffer = deque(maxlen=self.VIDEO_FRAMES)
        self.feature_buffer = deque(maxlen=self.VIDEO_FRAMES)
        self.folder_path = folder_path
        self.json_data = {}
    
    def addFrameAndFeature(self, frame, feature): 
        self.frame_buffer.append(frame)
        self.feature_buffer.append(feature)
        
    def translateLastFrameToBeginTime(self, frame, to_seconds=False):
        total_seconds = (frame - self.VIDEO_FRAMES) / self.fps
        if to_seconds:
            return total_seconds
        minutes, seconds = divmod(total_seconds, 60)
        return f"{int(minutes):02}-{int(seconds):02}"
    
    def createVideoClip(self, frame_id, shot_type):
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        clip_writer = cv2.VideoWriter_fourcc(*'MJPG')

        file_name = f'{self.folder_path}/clip_{self.counter}_{self.translateLastFrameToBeginTime(frame_id)}.mp4'
        print(f"fps is {self.fps} and frame_size is ({self.frame_width},{self.frame_height})")
        out = cv2.VideoWriter(
            file_name,
            clip_writer,
            self.fps,
            (self.frame_width, self.frame_height))

        for frame in self.frame_buffer:
            out.write(frame)
        out.release()
        print(f"Has create {file_name}.")

        shots_df = pd.DataFrame(
            np.concatenate(self.feature_buffer, axis=0),
            columns=columns
        )
        shots_df["shot"] = np.full(self.VIDEO_FRAMES, shot_type)
        outpath = Path(self.folder_path).joinpath(f"clip_{self.counter}_{shot_type}.csv")

        outpath.parent.mkdir(parents=True, exist_ok=True)

        shots_df.to_csv(outpath, index=False)
        print(f"saving csv to {outpath}")

        start_seconds = self.translateLastFrameToBeginTime(frame_id, to_seconds=True)
        end_seconds = self.translateLastFrameToBeginTime(frame_id, to_seconds=True) + 2

        trick_data = {
            "start": start_seconds,
            "end": end_seconds,
            "channel": 0,
            "labels": [shot_type.capitalize() + " Strike"]
        }

        if "tricks" not in self.json_data:
            self.json_data["tricks"] = []

        self.json_data["tricks"].append(trick_data)

        self.counter += 1

def process_keypoints(csv_file_path, video_file_path, dest_path, model):
    shot_counter = ShotCounter()
    
    df = pd.read_csv(csv_file_path)
    
    cap = cv2.VideoCapture(video_file_path)
    assert cap.isOpened()

    video_clip_maker = VideoClipMaker(
        30, 
        cap.get(cv2.CAP_PROP_FRAME_WIDTH), 
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        dest_path
    )
    video_clip_maker.video_file_path = video_file_path

    video_clip_maker.json_data = {
        "video_url": os.path.basename(video_file_path),
        "id": 1,
        "tricks": []
    }

    NB_IMAGES = 30
    FRAME_ID = 0
    features_pool = []

    for index, row in df.iterrows():
        FRAME_ID += 1
        
        features = row.values.reshape(1, 26)
        
        ret, frame = cap.read()
        if not ret:
            break

        video_clip_maker.addFrameAndFeature(frame.copy(), features)
        features_pool.append(features)

        if len(features_pool) == NB_IMAGES:
            features_seq = np.array(features_pool).reshape(1, NB_IMAGES, 26)
            assert features_seq.shape == (1, 30, 26)
            probs = model(features_seq)[0]
            shot_counter.update(probs, FRAME_ID)
            
            if shot_counter.frames_since_last_shot == 30:
                start_time = video_clip_maker.translateLastFrameToBeginTime(FRAME_ID - 30)
                end_time = video_clip_maker.translateLastFrameToBeginTime(FRAME_ID)
                
                if shot_counter.last_shot == "forehand":
                    print(f"Detected forehand shot from {start_time} to {end_time}")
                    video_clip_maker.createVideoClip(FRAME_ID, "forehand")
                elif shot_counter.last_shot == "backhand":
                    print(f"Detected backhand shot from {start_time} to {end_time}")
                    video_clip_maker.createVideoClip(FRAME_ID, "backhand")
                elif shot_counter.last_shot == "serve":
                    print(f"Detected serve shot from {start_time} to {end_time}")
                    video_clip_maker.createVideoClip(FRAME_ID, "serve")
            
            features_pool = features_pool[1:]

    cap.release()
    
    return video_clip_maker

def scan_through_folder(src, dest, model):
    if not os.path.exists(src):
        print("Source directory does not exist.")
        return
    
    if not os.path.exists(dest):
        os.makedirs(dest)
        print(f"Destination directory {dest} created.")

    for file in os.listdir(src):
        if file.endswith("_keypoints.csv"):
            csv_file_path = os.path.join(src, file)
            video_file_name = file.replace("_keypoints.csv", ".mp4")
            video_file_path = os.path.join(src, video_file_name)
            
            if not os.path.exists(video_file_path):
                print(f"Video file not found for {csv_file_path}. Skipping.")
                continue
            
            dest_final_folder = os.path.join(dest, os.path.splitext(video_file_name)[0])
            if not os.path.exists(dest_final_folder):
                os.makedirs(dest_final_folder)
            
            # Process keypoints and create video clips
            video_clip_maker = process_keypoints(csv_file_path, video_file_path, dest_final_folder, model)
            
            # Save the combined JSON file for the video
            json_file_name = f'{dest_final_folder}/{os.path.splitext(video_file_name)[0]}_timestamp.json'
            with open(json_file_name, 'w') as json_file:
                json.dump([video_clip_maker.json_data], json_file, indent=2)
            print(f"saving json to {json_file_name}")

if __name__ == "__main__":
    source_folder = "C:/Users/Lenovo/Desktop/RNN/new/2-2"  # This should point to where your CSV files are
    output_folder = "C:/Users/Lenovo/Desktop/RNN/new/2-2"
    model_file = "C:/Users/Lenovo/Desktop/RNN/tennis_rnn_rafa.keras"
    
    m1 = keras.models.load_model(model_file)
    scan_through_folder(source_folder, output_folder, m1)