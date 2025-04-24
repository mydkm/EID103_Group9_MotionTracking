import cv2
import numpy as np
import sys
import math # Added for atan2
import time # Added for time difference (dt)
import collections # Added for deque (efficient fixed-size list)

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

# --- Camera Setup ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Using DirectShow backend

if not cap.isOpened():
    print("Error: Could not open video device.")
    sys.exit()

print("Camera opened successfully.")

# Get frame dimensions (useful for text placement)
ret, frame_for_dims = cap.read()
if not ret:
    print("Error: Could not read initial frame for dimensions.")
    cap.release()
    sys.exit()
frame_height, frame_width = frame_for_dims.shape[:2]
print(f"Frame dimensions: {frame_width}x{frame_height}")

# --- HSV Range & Min Area ---
# --- TUNING REQUIRED based on actual conditions ---
L_limit = np.array([165, 60, 70])  # Lower H, S, V - STARTING POINT
U_limit = np.array([179, 255, 255]) # Upper H, S, V - STARTING POINT
MIN_AREA = 500 # Adjust this value

# --- Variables for Velocity Calculation (Appended) ---
previous_angle = None
previous_time = None
angular_velocity = 0.0 # rad/s
history_length = 150 # Number of data points for the graph history
velocity_history = collections.deque(maxlen=history_length)
# Initialize deque with zeros
for _ in range(history_length):
    velocity_history.append(0)

# --- OpenCV Graph Setup (Appended) ---
graph_height = 200 # Pixel height of graph window
graph_width = 450  # Pixel width (adjust as needed)
# Create black canvas [height, width, 3 channels], unsigned 8-bit integer
graph_canvas = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)
# Y-axis scaling: map velocity range to pixel height
max_expected_velocity = np.pi * 2.5 # Example: +/- 2.5*pi rad/s range
# Pixels per rad/s
y_scale = 2 * graph_height / (2 * max_expected_velocity) if max_expected_velocity > 0 else 1
# Y pixel coordinate corresponding to 0 velocity (middle)
y_offset = graph_height // 2


# --- Window Setup & Mouse Callback ---
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, set_origin_callback)
# Create window for the graph (Appended)
cv2.namedWindow('Angular Velocity Graph')


print("-----------------------------------------------------")
print("Please LEFT-CLICK on the window to set the reference origin point.")
print("Press ESC to exit.")
print("-----------------------------------------------------")


# --- Wait for Origin Setup Phase ---
while not origin_set:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame while waiting for origin. Exiting...")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()

    # Display instructions on the frame
    cv2.putText(frame, "LEFT-CLICK to set Origin", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2, cv2.LINE_AA) # Changed color

    cv2.imshow(window_name, frame)

    # Wait for key press or mouse click (handled by callback)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # Allow Esc to exit during setup
         print("ESC pressed during origin setup. Exiting.")
         cap.release()
         cv2.destroyAllWindows()
         sys.exit()

print("Origin set. Starting object tracking relative to origin.")

# --- Main Processing Loop ---
while True:
    # Get current time at the beginning of the frame processing
    current_time = time.time() # Moved here
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting loop...")
        break

    # Reset variables for this frame
    center = None
    relative_coords = None
    current_angle = None # Initialize angle for this frame

    # --- Draw Origin Marker ---
    if origin_set:
        # Draw a blue cross marker at the selected origin
        cv2.drawMarker(frame, (origin_x, origin_y), (255, 0, 0), # Blue color
                       cv2.MARKER_CROSS, markerSize=15, thickness=2)

    # --- Color Detection ---
    into_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pink_mask = cv2.inRange(into_hsv, L_limit, U_limit)
    # Optional morphological operations (uncomment to use)
    # kernel = np.ones((5,5), np.uint8)
    # pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_OPEN, kernel)
    # pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel)


    # --- Contour Finding ---
    contours, hierarchy = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    max_area = 0

    # --- Find Largest Contour ---
    if contours:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_contour = contour

        # --- Process Largest Contour if Valid ---
        if largest_contour is not None and max_area > MIN_AREA:
            # Calculate Bounding Box
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Calculate Center (Absolute Coordinates)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center = (cX, cY)

                # --- Calculate Relative Coordinates ---
                if origin_set: # Only calculate if origin has been defined
                    relative_x = cX - origin_x
                    # Using inverted Y relative to origin (as in original code)
                    relative_y = origin_y - cY
                    relative_coords = (relative_x, relative_y)

                    # --- Calculate Current Angle (Appended Logic) ---
                    # Calculate angle using atan2 - safe for all quadrants
                    # Returns angle in radians, range (-pi, pi]
                    # Consistent with relative_y = origin_y - cY (Cartesian-like angle)
                    if relative_x != 0 or relative_y != 0:
                         current_angle = math.atan2(relative_y, relative_x)
                    else:
                         current_angle = 0.0 # Define angle as 0 if exactly at origin


            # --- Draw Visualizations ---
            # Bounding box (Green)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Center point (Absolute - Red)
            if center:
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
            # Line from origin to center (Cyan)
            if center and origin_set:
                cv2.line(frame, (origin_x, origin_y), center, (255, 255, 0), 1)

            # --- Display Relative Coordinates Text ---
            if relative_coords:
                 coord_text = f"Rel:({relative_coords[0]}, {relative_coords[1]})"
                 # Position text near the top-left of the bounding box
                 cv2.putText(frame, coord_text, (x, y - 10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) # Yellow text


    # --- Calculate Angular Velocity (Appended Logic) ---
    # Check if we have data from the previous frame and a valid angle this frame
    if current_angle is not None and previous_angle is not None and previous_time is not None:
        delta_t = current_time - previous_time
        if delta_t > 1e-6: # Ensure time difference is sufficient
            # Calculate difference in angle
            delta_angle = current_angle - previous_angle
            # Handle Angle Wrap-Around (-pi to pi)
            if delta_angle > math.pi: delta_angle -= 2 * math.pi
            elif delta_angle < -math.pi: delta_angle += 2 * math.pi
            # Calculate angular velocity (radians per second)
            angular_velocity = delta_angle / delta_t
        # else: Keep previous velocity if dt is too small or zero
    else: # Reset velocity if object not detected or first detection
        angular_velocity = 0.0

    # --- Store Velocity & Update State for Next Frame (Appended Logic) ---
    velocity_history.append(angular_velocity) # Add current value to history deque
    # Update previous state only if object was detected this frame
    if current_angle is not None:
        previous_angle = current_angle
        previous_time = current_time
    # Else: Keep previous angle/time, velocity will be recalculated based on last known


    # --- Print Coordinates to Console ---
    if center:
        abs_coord_text = f"Abs Center:({center[0]}, {center[1]}) Area:{max_area:.0f}"
        print(abs_coord_text, end='')
        if relative_coords:
             print(f" | Rel Center:({relative_coords[0]}, {relative_coords[1]})", end='') # Print relative coords
        else:
             print(" | Rel Center: (Origin not set)", end='') # Indicate if origin isn't set yet
        # Print velocity on the same line (Appended)
        print(f" | AngVel: {angular_velocity:.2f} rad/s")
    # else:
        # print("Pink object not found or too small.") # Optional message


    # --- Display Angular Velocity Text on Main Frame (Appended) ---
    vel_text = f"AngVel: {angular_velocity:.2f} rad/s"
    cv2.putText(frame, vel_text, (10, frame_height - 10), # Bottom-left using frame_height
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2) # Cyan


    # --- Draw Graph on OpenCV Canvas (Appended Logic) ---
    graph_canvas.fill(0) # Clear canvas (fill with black)
    # Draw Y=0 axis (center line)
    cv2.line(graph_canvas, (0, y_offset), (graph_width, y_offset), (0, 80, 0), 1) # Dark green
    # Prepare points for the velocity polyline
    points = []
    for i, vel in enumerate(list(velocity_history)):
        # Calculate x position based on history index
        x_pos = int(i * (graph_width / history_length))
        # Calculate y position based on velocity, scale, and offset
        # Clamp velocity to prevent extreme values going off canvas visually
        clamped_vel = np.clip(vel, -max_expected_velocity, max_expected_velocity)
        # Invert y-axis for drawing: positive velocity goes up (lower y_pos)
        y_pos = int(y_offset - clamped_vel * y_scale)
        # Ensure y_pos stays within canvas bounds after calculation
        y_pos = np.clip(y_pos, 0, graph_height - 1)
        points.append((x_pos, y_pos))
    # Draw the velocity graph line
    if len(points) > 1:
        cv2.polylines(graph_canvas, [np.array(points)], isClosed=False, color=(0, 255, 255), thickness=1) # Cyan line
    # Draw text labels on graph canvas
    cv2.putText(graph_canvas, f"{angular_velocity:.2f} rad/s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(graph_canvas, f"+/-{max_expected_velocity:.1f}", (10, graph_height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 0), 1)


    # --- Display the main frame ---
    cv2.imshow(window_name, frame)
    # --- Display the graph frame (Appended) ---
    cv2.imshow('Angular Velocity Graph', graph_canvas)
    # cv2.imshow('Pink Mask', pink_mask) # Keep mask window if needed for tuning

    # --- Exit Condition ---
    if cv2.waitKey(1) == 27: # Check for ESC key press
        print("ESC key pressed. Exiting...")
        break

# --- Cleanup ---
print("Releasing camera and closing windows.")
cap.release()
cv2.destroyAllWindows() # Destroys all OpenCV windows
print("Exited.")
