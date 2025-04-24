# EID103_Group9_MotionTracking

# Pink Object Tracker with Relative Coordinates and Velocity

## Description

This Python script uses OpenCV to track a pink-colored object detected by a connected camera. It allows the user to define a reference origin point by clicking on the video feed. The script then calculates and displays:

1.  The absolute coordinates (center) of the detected pink object.
2.  The coordinates of the object's center *relative* to the user-defined origin.
3.  The angular (rotational) velocity of the object's center around the defined origin.

The script displays the live camera feed with visualizations (origin marker, object bounding box, center point, line from origin to object) and relevant coordinate/velocity text. It also shows a separate window with a real-time graph of the calculated angular velocity, drawn using OpenCV.

## Features

* Real-time video capture from a connected camera.
* Color detection based on HSV (Hue, Saturation, Value) ranges (tuned for pink).
* User-defined origin point via mouse click.
* Calculation of absolute and relative object coordinates.
* Calculation of angular velocity relative to the origin.
* Visualization of tracking results on the main video feed.
* Real-time graphing of angular velocity using OpenCV.
* Option to exit the application using the ESC key.

## Requirements

* Python 3.x
* Required Python packages (listed in `requirements.txt`):
    * `opencv-python`
    * `numpy`
* A connected webcam compatible with OpenCV.

## Setup

1.  **Clone or Download:** Obtain the Python script file (e.g., `tracker.py`).
2.  **Install Dependencies:** Open a terminal or command prompt in the directory containing the script and `requirements.txt`, then run:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Camera Connection:** Ensure your webcam is properly connected to your computer and recognized by the operating system.
    * **Windows:** Drivers are usually handled automatically. Ensure camera privacy settings allow desktop apps to access the camera (Settings -> Privacy & Security -> Camera). The script uses `cv2.CAP_DSHOW`, which often works well on Windows.
    * **Linux:** You might need to add your user to the `video` group: `sudo usermod -a -G video $USER`. You may need to log out and back in for this change to take effect.
    * **macOS:** Ensure applications have permission to access the camera in System Settings -> Privacy & Security -> Camera.

## Running the Script

1.  Open a terminal or command prompt.
2.  Navigate to the directory containing the script.
3.  Run the script using:
    ```bash
    python your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual filename).
4.  **Set Origin:** Two windows will appear (the main frame and the graph). The main frame window will initially prompt you to "LEFT-CLICK to set Origin". Click anywhere on this window to define the `(0, 0)` point for relative measurements.
5.  **Tracking:** Once the origin is set, the script will start tracking the largest pink object it finds. You'll see the visualizations on the main frame and the velocity graph update in the second window. Coordinate and velocity information will also be printed to the console.
6.  **Exit:** Press the `ESC` key while one of the OpenCV windows is active to stop the script and close all windows.

## Tuning

The accuracy of the color detection heavily depends on the lighting conditions and the specific shade of the target object. You will likely need to adjust these values near the top of the script:

* `L_limit = np.array([165, 60, 70])`: The **Lower** HSV bounds.
* `U_limit = np.array([179, 255, 255])`: The **Upper** HSV bounds.
* `MIN_AREA = 500`: The minimum pixel area for a detected contour to be considered the target object (helps filter noise).

To tune the HSV values, you can uncomment the `cv2.imshow('Pink Mask', pink_mask)` line in the main loop. This will show a black and white image where white pixels represent areas matching the current HSV range. Adjust `L_limit` and `U_limit` until the target object appears clearly white in the mask window under your typical operating conditions, with minimal white noise elsewhere.
