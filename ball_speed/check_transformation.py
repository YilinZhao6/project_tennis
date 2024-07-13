import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import time

def get_boundaries_and_center(frame, message):
    points = []
    center = None
    def mouse_callback(event, x, y, flags, param):
        nonlocal center
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            elif center is None:
                center = (x, y)
                cv2.drawMarker(frame, (x, y), (0, 0, 255), cv2.MARKER_STAR, 20, 2)
            cv2.imshow(message, frame)
            if len(points) == 4 and center is not None:
                cv2.destroyAllWindows()

    cv2.namedWindow(message, cv2.WINDOW_NORMAL)
    cv2.imshow(message, frame)
    cv2.setMouseCallback(message, mouse_callback)
    
    print(f"Please click on the four corners of the {message} in this order:")
    print("1. Top-left  2. Top-right  3. Bottom-right  4. Bottom-left")
    print("Then click on the center of the court (red star).")
    print("You can resize the window if needed.")
    
    while len(points) < 4 or center is None:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

    return np.array(points, dtype=np.float32), center

def detect_and_transform_tennis_balls(input_video, output_video, transformed_output_video, fps, skip_frames=3, confidence=0.25):
    model = YOLO('yolov8n.pt')
    #model = YOLO('best.pt')

    cap = cv2.VideoCapture(input_video)
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the video")
        return

    court_boundaries, court_center = get_boundaries_and_center(first_frame, "Tennis Court")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Original video resolution: {width}x{height}")
    print(f"FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps / (skip_frames + 1), (width, height))
    out_transformed = cv2.VideoWriter(transformed_output_video, fourcc, fps / (skip_frames + 1), (800, 400))

    # Standard tennis court dimensions (in pixels)
    std_court_width, std_court_height = 700, 350
    std_court = np.array([
        [50, 25], [750, 25], [750, 375], [50, 375]
    ], dtype=np.float32)

    # Calculate transformation matrix
    M = cv2.getPerspectiveTransform(court_boundaries, std_court)

    pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        pbar.update(1)

        if frame_count % (skip_frames + 1) != 0:
            continue

        results = model(frame, imgsz=(height, width), conf=confidence)

        annotated_frame = np.zeros_like(frame)
        transformed_frame = np.zeros((400, 800, 3), dtype=np.uint8)
        cv2.polylines(transformed_frame, [std_court.astype(int)], True, (0, 255, 255), 2)

        ball_position = None
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == 32:  # Sports ball class
                    ball_position = (center_x, center_y)

        if ball_position is not None:
            cv2.circle(annotated_frame, ball_position, 5, (0, 255, 255), -1)
            
            # Transform ball position
            transformed_ball = cv2.perspectiveTransform(np.array([[ball_position]], dtype=np.float32), M)
            transformed_ball_pos = tuple(map(int, transformed_ball[0][0]))
            cv2.circle(transformed_frame, transformed_ball_pos, 5, (0, 255, 255), -1)

        cv2.polylines(annotated_frame, [court_boundaries.astype(int)], True, (0, 255, 255), 2)

        out.write(annotated_frame)
        out_transformed.write(transformed_frame)

    pbar.close()
    cap.release()
    out.release()
    out_transformed.release()

    print(f"Tennis ball detection completed. Output saved to {output_video}")
    print(f"Transformed output saved to {transformed_output_video}")

# Usage
input_video = "7.mp4"
output_video = "7-output-mask.mp4"
transformed_output_video = "7-transformed-output.mp4"
fps = 28.9

detect_and_transform_tennis_balls(input_video, output_video, transformed_output_video, fps, skip_frames=0, confidence=0.10)