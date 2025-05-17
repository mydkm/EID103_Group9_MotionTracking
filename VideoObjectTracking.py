import cv2
import numpy as np
import sys
import math # Added for atan2
import time # Added for time difference (dt)
import collections # Added for deque (efficient fixed-size list)
import csv # Added for saving data
import datetime # Added for unique filename

# --- Ask user for video file path ---
video_path = input("Please enter the full path to your video file: ").strip()

if not video_path:
    print("No video path entered. Exiting.")
    sys.exit()

# --- Global variables for origin ---
origin_x, origin_y = None, None
origin_set = False
window_name = 'Frame - Click to Set Origin' # Define window name globally

# --- Mouse callback function ---
def set_origin_callback(event, x, y, flags, param):
    """Handles mouse clicks to set the origin point."""
    global origin_x, origin_y, origin_set
    # Capture only the first left button down click
    if event == cv2.EVENT_LBUTTONDOWN and not origin_set:
        origin_x, origin_y = x, y
        origin_set = True
        print(f"Origin set at: ({origin_x}, {origin_y})")
        # Update window title after setting origin
        cv2.setWindowTitle(window_name, 'Frame - Tracking Relative to Origin')

# --- Video File Setup ---
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    print("Please ensure the path is correct and the file is a valid video format.")
    sys.exit()

print(f"Video file '{video_path}' opened successfully.")

# Get frame dimensions (useful for text placement)
ret, frame_for_dims = cap.read()
if not ret:
    print("Error: Could not read initial frame for dimensions from video.")
    cap.release()
    sys.exit()
frame_height, frame_width = frame_for_dims.shape[:2]
print(f"Frame dimensions: {frame_width}x{frame_height}")

# Reset video to the beginning for actual processing
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


# --- HSV Range & Min Area (Pink) ---
# --- TUNING REQUIRED based on actual conditions and video ---
L_limit = np.array([98, 50, 50])  # Lower H, S, V - STARTING POINT
U_limit = np.array([139, 255, 255]) # Upper H, S, V - STARTING POINT
MIN_AREA = 500 # Adjust this value

# --- Variables for Velocity Calculation ---
previous_angle = None
previous_time = None
angular_velocity = 0.0 # rad/s
history_length = 150 # Number of data points for the graph history
velocity_history = collections.deque(maxlen=history_length)
# Initialize deque with zeros
for _ in range(history_length):
    velocity_history.append(0)

# --- Data Logging Setup ---
# Generate a unique filename using timestamp
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Create a "safe" version of the video path for the filename (optional)
safe_video_name_part = "".join(c if c.isalnum() else "_" for c in video_path.split('/')[-1].split('\\')[-1])
csv_filename = f"tracking_data_{safe_video_name_part}_{timestamp_str}.csv"
data_log = [] # List to store data rows before writing
csv_header = [
    'Timestamp_VideoTime_s', 'Abs_Center_X', 'Abs_Center_Y',
    'Rel_Center_X', 'Rel_Center_Y', 'Area', 'Angular_Velocity_RadS'
]

# --- OpenCV Graph Setup ---
graph_height = 200 # Pixel height of graph window
graph_width = 450  # Pixel width (adjust as needed)
graph_canvas = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)
max_expected_velocity = np.pi * 2.5 # Example: +/- 2.5*pi rad/s range
y_scale = graph_height / (2 * max_expected_velocity) if max_expected_velocity > 0 else 1
y_offset = graph_height // 2


# --- Window Setup & Mouse Callback ---
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, set_origin_callback)
cv2.namedWindow('Angular Velocity Graph')
cv2.namedWindow('Pink Mask')


print("-----------------------------------------------------")
print(f"Processing video: {video_path}")
print("Please LEFT-CLICK on the window to set the reference origin point.")
print("The video will play. Click on a frame to set the origin.")
print("Press ESC to exit.")
print("-----------------------------------------------------")


# --- Wait for Origin Setup Phase ---
while not origin_set:
    ret, frame = cap.read()
    if not ret:
        print("Video ended before origin was set or error reading frame. Exiting...")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()

    cv2.putText(frame, "LEFT-CLICK to set Origin", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2, cv2.LINE_AA)
    cv2.imshow(window_name, frame)

    key = cv2.waitKey(30) & 0xFF # Wait 30ms
    if key == 27:
        print("ESC pressed during origin setup. Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()

print("Origin set. Starting object tracking relative to origin.")

# --- Main Processing Loop ---
frame_count = 0
start_processing_time = time.time()

while True:
    current_processing_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("End of video file reached or error reading frame. Exiting loop...")
        break

    frame_count += 1
    center = None
    relative_coords = None
    current_angle = None
    max_area = 0

    if origin_set:
        cv2.drawMarker(frame, (origin_x, origin_y), (255, 0, 0),
                         cv2.MARKER_CROSS, markerSize=15, thickness=2)

    into_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pink_mask = cv2.inRange(into_hsv, L_limit, U_limit)
    kernel = np.ones((5,5), np.uint8)
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_OPEN, kernel)
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = None

    if contours:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_contour = contour

        if largest_contour is not None and max_area > MIN_AREA:
            x, y, w, h = cv2.boundingRect(largest_contour)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center = (cX, cY)

                if origin_set:
                    relative_x = cX - origin_x
                    relative_y = origin_y - cY
                    relative_coords = (relative_x, relative_y)
                    current_angle = math.atan2(relative_y, relative_x) if (relative_x != 0 or relative_y != 0) else 0.0

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if center: cv2.circle(frame, center, 5, (0, 0, 255), -1)
            if center and origin_set: cv2.line(frame, (origin_x, origin_y), center, (255, 255, 0), 1)
            if relative_coords:
                coord_text = f"Rel:({relative_coords[0]}, {relative_coords[1]})"
                cv2.putText(frame, coord_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if current_angle is not None and previous_angle is not None and previous_time is not None:
        delta_t = current_processing_time - previous_time
        if delta_t > 1e-6:
            delta_angle = current_angle - previous_angle
            if delta_angle > math.pi: delta_angle -= 2 * math.pi
            elif delta_angle < -math.pi: delta_angle += 2 * math.pi
            angular_velocity = delta_angle / delta_t
    else:
        angular_velocity = 0.0

    velocity_history.append(angular_velocity)
    if current_angle is not None:
        previous_angle = current_angle
        previous_time = current_processing_time

    if center is not None and origin_set:
        rel_x_val = relative_coords[0] if relative_coords is not None else None
        rel_y_val = relative_coords[1] if relative_coords is not None else None
        video_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        log_timestamp = video_timestamp_ms / 1000.0 if video_timestamp_ms > 0 else current_processing_time - start_processing_time
        data_row = [log_timestamp, center[0], center[1], rel_x_val, rel_y_val, max_area, angular_velocity]
        data_log.append(data_row)

    if center:
        abs_coord_text = f"F: {frame_count} Abs:({center[0]},{center[1]}) A:{max_area:.0f}"
        print(abs_coord_text, end='')
        if relative_coords: print(f" | Rel:({relative_coords[0]},{relative_coords[1]})", end='')
        else: print(" | Rel: (Origin not set)", end='')
        print(f" | AVel: {angular_velocity:.2f} r/s")

    vel_text = f"AngVel: {angular_velocity:.2f} rad/s"
    cv2.putText(frame, vel_text, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    graph_canvas.fill(0)
    cv2.line(graph_canvas, (0, y_offset), (graph_width, y_offset), (0, 80, 0), 1)
    points = []
    for i, vel in enumerate(list(velocity_history)):
        x_pos = int(i * (graph_width / history_length))
        clamped_vel = np.clip(vel, -max_expected_velocity, max_expected_velocity)
        y_pos = int(y_offset - clamped_vel * y_scale)
        y_pos = np.clip(y_pos, 0, graph_height - 1)
        points.append((x_pos, y_pos))
    if len(points) > 1:
        cv2.polylines(graph_canvas, [np.array(points)], isClosed=False, color=(0, 255, 255), thickness=1)
    cv2.putText(graph_canvas, f"{angular_velocity:.2f} r/s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200),1)
    cv2.putText(graph_canvas, f"+/-{max_expected_velocity:.1f}", (10,graph_height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,150,0),1)

    cv2.imshow(window_name, frame)
    cv2.imshow('Angular Velocity Graph', graph_canvas)
    cv2.imshow('Pink Mask', pink_mask)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        print("ESC key pressed. Exiting...")
        break

# --- Cleanup ---
print("Releasing video and closing windows.")
cap.release()
cv2.destroyAllWindows()

# --- Save Collected Data to CSV ---
if data_log:
    print(f"\nSaving collected data to {csv_filename}...")
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)
            writer.writerows(data_log)
        print("Data saved successfully.")
    except IOError as e:
        print(f"Error saving data to CSV: {e}")
else:
    print("\nNo data logged.")

print("Exited.")