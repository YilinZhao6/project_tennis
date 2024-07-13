import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import csv

def get_boundaries(frame, message):
    points = []
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(message, frame)
            if len(points) == 4:
                cv2.destroyAllWindows()

    cv2.namedWindow(message, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(message, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(message, frame)
    cv2.setMouseCallback(message, mouse_callback)
    
    print(f"Please click on the four corners of the {message} in this order:")
    print("1. Top-left  2. Top-right  3. Bottom-right  4. Bottom-left")
    print("You can resize the window if needed.")
    
    while len(points) < 4:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

    return np.array(points, dtype=np.float32)

def get_court_boundaries_and_center(frame):
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
            cv2.imshow("Tennis Court", frame)
            if len(points) == 4 and center is not None:
                cv2.destroyAllWindows()

    cv2.namedWindow("Tennis Court", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Tennis Court", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Tennis Court", frame)
    cv2.setMouseCallback("Tennis Court", mouse_callback)
    
    print("Please click on the four corners of the Tennis Court in this order:")
    print("1. Top-left  2. Top-right  3. Bottom-right  4. Bottom-left")
    print("Then click on the center of the court (red star).")
    
    while len(points) < 4 or center is None:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

    return np.array(points, dtype=np.float32), center

def get_horizontal_line(frame, message):
    point = []
    preview_frame = frame.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal preview_frame
        if event == cv2.EVENT_MOUSEMOVE:
            preview_frame = frame.copy()
            cv2.line(preview_frame, (0, y), (frame.shape[1], y), (0, 255, 0), 2)
            cv2.imshow(message, preview_frame)
        elif event == cv2.EVENT_LBUTTONDOWN:
            point.append((x, y))
            cv2.line(frame, (0, y), (frame.shape[1], y), (0, 255, 0), 2)
            cv2.imshow(message, frame)
            cv2.destroyAllWindows()
    
    cv2.namedWindow(message, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(message, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(message, frame)
    cv2.setMouseCallback(message, mouse_callback)
    
    print(f"Please click to draw a horizontal line separating Player 1 (below) and Player 2 (above)")
    
    while len(point) < 1:
        cv2.waitKey(1)

    return point[0][1]  # Return the y-coordinate of the line

def compute_homography(court_boundaries, court_center):
    # Define the corners of a regulation tennis court (in meters)
    real_court = np.array([
        [-5.485, -11.885],
        [5.485, -11.885],
        [5.485, 11.885],
        [-5.485, 11.885]
    ], dtype=np.float32)

    # Compute the perspective transform
    return cv2.getPerspectiveTransform(court_boundaries, real_court)

def transform_point(point, homography):
    p = np.array([point[0], point[1], 1])
    tp = np.dot(homography, p)
    return tp[:2] / tp[2]

def calculate_speed_improved(prev_pos, curr_pos, time_diff, homography):
    # Transform points to top-down view
    prev_pos_transformed = transform_point(prev_pos, homography)
    curr_pos_transformed = transform_point(curr_pos, homography)

    # Calculate distance in meters
    distance = np.linalg.norm(curr_pos_transformed - prev_pos_transformed)
    
    # Calculate speed
    speed = distance / time_diff if time_diff > 0 else 0
    return min(speed, 50)  # Cap the speed at 50 m/s

def point_inside_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def detect_tennis_balls_and_players(input_video, output_video, fps, player_ball_proximity_threshold=150, skip_frames=3, confidence=0.25):
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(input_video)
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the video")
        return

    detection_area = get_boundaries(first_frame, "Detection Area")
    court_boundaries, court_center = get_court_boundaries_and_center(first_frame)
    player_separation_y = get_horizontal_line(first_frame, "Player Separation Line")

    homography = compute_homography(court_boundaries, court_center)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Original video resolution: {width}x{height}")
    print(f"FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps / (skip_frames + 1), (width, height))

    pbar = tqdm(total=total_frames, desc="Processing frames", unit="frame")

    frame_count = 0
    prev_ball_pos = None
    prev_ball_time = None
    current_turn = "Player 2"
    current_speed = 0
    max_speed_in_turn = 0
    turn_start_time = 0
    ball_hit = False

    # Create CSV file for velocity data
    with open('velocity_data.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Frame', 'Time', 'Turn', 'Velocity'])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            pbar.update(1)

            if frame_count % (skip_frames + 1) != 0:
                continue

            current_time = frame_count / fps

            results = model(frame, imgsz=(height, width), conf=confidence)

            annotated_frame = frame.copy()

            player1_pos = None
            player2_pos = None
            ball_position = None
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    if not point_inside_polygon(center_x, center_y, detection_area):
                        continue

                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls == 0:  # Person class
                        if center_y > player_separation_y:
                            player1_pos = (center_x, center_y)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f"Player 1: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        else:
                            player2_pos = (center_x, center_y)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(annotated_frame, f"Player 2: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    
                    elif cls == 32:  # Sports ball class
                        ball_pos = np.array([center_x, center_y])
                        ball_position = ball_pos
                        if prev_ball_pos is not None and prev_ball_time is not None:
                            time_diff = current_time - prev_ball_time
                            current_speed = calculate_speed_improved(prev_ball_pos, ball_pos, time_diff, homography)
                            
                            if current_speed > 5 and not ball_hit:
                                ball_hit = True
                                max_speed_in_turn = max(max_speed_in_turn, current_speed)
                        
                        prev_ball_pos = ball_pos
                        prev_ball_time = current_time

            if ball_position is not None:
                ball_close_to_player1 = player1_pos and np.linalg.norm(np.array(ball_position) - np.array(player1_pos)) < player_ball_proximity_threshold
                ball_close_to_player2 = player2_pos and np.linalg.norm(np.array(ball_position) - np.array(player2_pos)) < player_ball_proximity_threshold
                
                if ball_close_to_player1 and current_turn == "Player 2":
                    current_turn = "Player 1"
                    max_speed_in_turn = 0
                    turn_start_time = current_time
                    ball_hit = False
                elif ball_close_to_player2 and current_turn == "Player 1":
                    current_turn = "Player 2"
                    max_speed_in_turn = 0
                    turn_start_time = current_time
                    ball_hit = False
                
                color = (0, 0, 255) if (ball_close_to_player1 or ball_close_to_player2) else (0, 255, 255)
                cv2.circle(annotated_frame, tuple(ball_position.astype(int)), 5, color, -1)
            
            cv2.putText(annotated_frame, f"Turn: {current_turn}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Current Speed: {current_speed:.2f} m/s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Max Speed in Turn: {max_speed_in_turn:.2f} m/s", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.polylines(annotated_frame, [court_boundaries.astype(int)], True, (0, 255, 255), 2)
            cv2.polylines(annotated_frame, [detection_area.astype(int)], True, (0, 255, 0), 2)
            cv2.line(annotated_frame, (0, player_separation_y), (width, player_separation_y), (255, 255, 0), 2)
            cv2.drawMarker(annotated_frame, tuple(map(int, court_center)), (0, 0, 255), cv2.MARKER_STAR, 20, 2)

            out.write(annotated_frame)

            # Write velocity data to CSV
            csv_writer.writerow([frame_count, current_time, current_turn, current_speed])

            # Update max_speed_in_turn
            max_speed_in_turn = max(max_speed_in_turn, current_speed)

    pbar.close()
    cap.release()
    out.release()

    print(f"Tennis ball and player detection completed. Output saved to {output_video}")
    print(f"Velocity data saved to velocity_data.csv")

# Usage
input_video = "7.mp4"
output_video = "7-output.mp4"
fps = 28.9

detect_tennis_balls_and_players(input_video, output_video, fps, player_ball_proximity_threshold=150, skip_frames=2, confidence=0.15)