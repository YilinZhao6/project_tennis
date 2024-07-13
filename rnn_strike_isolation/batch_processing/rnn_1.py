import cv2
import numpy as np
import pandas as pd
import os
import time
from extract_human_pose import HumanPoseExtractor

def is_video_corrupt(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return True
    ret, frame = cap.read()
    cap.release()
    return not ret

def extract_keypoints_to_csv(video_file_path, output_folder):
    if is_video_corrupt(video_file_path):
        print(f"Skipping corrupt video: {video_file_path}")
        return None

    cap = cv2.VideoCapture(video_file_path)
    assert cap.isOpened()

    ret, frame = cap.read()
    human_pose_extractor = HumanPoseExtractor(frame.shape)

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

    all_features = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        human_pose_extractor.extract(frame)
        human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])

        features = human_pose_extractor.keypoints_with_scores.reshape(17, 3)
        features = features[features[:, 2] > 0][:, 0:2].reshape(1, 13 * 2)
        all_features.append(features[0])

        human_pose_extractor.roi.update(human_pose_extractor.keypoints_pixels_frame)

    cap.release()

    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_features, columns=columns)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    csv_filename = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_file_path))[0]}_keypoints.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Keypoints saved to {csv_filename}")

    return csv_filename

def process_videos(input_folder, output_folder):
    processed_videos = set()

    print("Starting video detection and processing...")
    
    try:
        while True:
            print("Detecting videos...")
            new_videos_found = False
            
            # Check for new videos
            for filename in os.listdir(input_folder):
                if filename.endswith(('.mp4', '.avi', '.mov')):  # Add more video extensions if needed
                    video_path = os.path.join(input_folder, filename)
                    if video_path not in processed_videos:
                        new_videos_found = True
                        print(f"Processing video: {video_path}")
                        csv_file = extract_keypoints_to_csv(video_path, output_folder)
                        if csv_file:
                            processed_videos.add(video_path)
                            print(f"Finished processing: {video_path}")

            if not new_videos_found:
                print("No new videos detected. Continuing to monitor...")
            
            time.sleep(5)  # Wait for 5 seconds before checking for new videos again

    except KeyboardInterrupt:
        print("\nProcess stopped by user. Exiting...")

if __name__ == "__main__":
    input_folder = "C:/Users/Lenovo/Desktop/RNN/new/videos"
    output_folder = "C:/Users/Lenovo/Desktop/RNN/new/output"
    process_videos(input_folder, output_folder)