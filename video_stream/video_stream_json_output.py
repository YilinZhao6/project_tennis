import cv2
import datetime
import time
import os
import json
from threading import Thread, Lock

def record_rtsp(rtsp_url, start_delay, clip_duration, output_folder):
    print(f"Waiting {start_delay} seconds before starting recording for {rtsp_url}")
    time.sleep(start_delay)
    
    print(f"Creating output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Opening RTSP stream: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream: {rtsp_url}")
        return

    # Set fps to 25 as specified
    fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Start recording: {rtsp_url}")
    
    # Create JSON file for timing information
    json_file = os.path.join(output_folder, "video_times.json")
    video_times = []
    json_lock = Lock()
    
    while True:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_folder, f"output_{timestamp}.mp4")
        
        print(f"Initializing video writer for: {output_file}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        start_time = time.time()
        start_datetime = datetime.datetime.now().isoformat()
        
        while time.time() - start_time < clip_duration:
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame from {rtsp_url}")
                break
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(frame)
        
        out.release()
        end_datetime = datetime.datetime.now().isoformat()
        print(f"Completed 20-second clip: {output_file}")
        
        # Update JSON file with timing information
        video_info = {
            "filename": os.path.basename(output_file),
            "start_time": start_datetime,
            "end_time": end_datetime
        }
        video_times.append(video_info)
        
        with json_lock:
            with open(json_file, 'w') as f:
                json.dump(video_times, f, indent=4)
        
        # Check if the stream is still available
        if not cap.isOpened():
            print(f"Error: RTSP stream closed for {rtsp_url}")
            break

    cap.release()
    print(f"Recording stopped for: {rtsp_url}")

# RTSP URLs
rtsp_urls = [
    "rtsp://admin:123456@192.168.1.102:8554/profile0",
    "rtsp://admin:123456@192.168.1.102:8554/profile0",
    "rtsp://admin:123456@192.168.1.102:8554/profile0",
    "rtsp://admin:123456@192.168.1.102:8554/profile0"
]

print("Starting recording threads")
threads = []
for i, url in enumerate(rtsp_urls):
    start_delay = 5 + i * 3  # 5, 8, 11, 14 seconds
    output_folder = f"rtsp_output_{i+1}"
    thread = Thread(target=record_rtsp, args=(url, start_delay, 20, output_folder))
    threads.append(thread)
    thread.start()

print("All recording threads started. Press Ctrl+C to stop.")

try:
    # Keep the main thread alive
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping all recordings...")

print("Waiting for all recordings to complete")
for thread in threads:
    thread.join()

print("All recordings completed.")