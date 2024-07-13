import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import os
import json

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
    def __init__(self, fps):
        self.MIN_FRAMES_BETWEEN_SHOTS = 2 * fps
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

def process_keypoints(csv_file_path, model, fps):
    shot_counter = ShotCounter(fps)
    
    df = pd.read_csv(csv_file_path)

    json_data = {
        "video_url": os.path.basename(csv_file_path).replace("_keypoints.csv", ".mp4"),
        "id": 1,
        "tricks": []
    }

    NB_IMAGES = 30
    FRAME_ID = 0
    features_pool = []

    for index, row in df.iterrows():
        FRAME_ID += 1
        
        features = row.values.reshape(1, 26)
        features_pool.append(features)

        if len(features_pool) == NB_IMAGES:
            features_seq = np.array(features_pool).reshape(1, NB_IMAGES, 26)
            assert features_seq.shape == (1, 30, 26)
            probs = model(features_seq)[0]
            shot_counter.update(probs, FRAME_ID)
            
            if shot_counter.frames_since_last_shot == 30:
                start_time = (FRAME_ID - 2*fps) / fps  # Start 1 second before the detection
                end_time = FRAME_ID / fps  # End at the current frame
                
                if shot_counter.last_shot in ["forehand", "backhand", "serve"]:
                    print(f"Detected {shot_counter.last_shot} shot from {start_time:.2f}s to {end_time:.2f}s")
                    json_data["tricks"].append({
                        "start": start_time,
                        "end": end_time,
                        "channel": 0,
                        "labels": [shot_counter.last_shot.capitalize() + " Strike"]
                    })
            
            features_pool = features_pool[1:]
    
    return json_data

def process_csv(csv_file_path, output_folder, model, fps):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Destination directory {output_folder} created.")

    # Process keypoints and create JSON data
    json_data = process_keypoints(csv_file_path, model, fps)
    
    # Save the JSON file
    json_file_name = os.path.join(output_folder, os.path.basename(csv_file_path).replace("_keypoints.csv", "_timestamp.json"))
    with open(json_file_name, 'w') as json_file:
        json.dump([json_data], json_file, indent=2)
    print(f"Saving JSON to {json_file_name}")

if __name__ == "__main__":
    csv_file_path = "C:/Users/Lenovo/Desktop/RNN/new/output/2_keypoints.csv"  # This should point to your specific CSV file
    output_folder = "C:/Users/Lenovo/Desktop/RNN/new/2"
    model_file = "C:/Users/Lenovo/Desktop/RNN/tennis_rnn_rafa.keras"
    fps = 30  # You can adjust this value as needed
    
    m1 = keras.models.load_model(model_file)
    process_csv(csv_file_path, output_folder, m1, fps)