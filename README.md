Smart Light Control with YOLOv8 and Camera

This project uses a camera and YOLOv8 object detection to implement a smart light system. The light automatically turns on when a person is detected and off when no person is present.

Features

Real-time person detection using YOLOv8.

Controls a light via HTTP request to a networked relay (on/off).

Threaded design for smooth camera capture and frame processing.

Displays camera feed with bounding boxes showing detected people.

Prevents repeated commands by storing the current light state.

Hardware & Setup
Component	Description / Connection
Camera	USB or built-in camera connected to PC
Relay / Light	Connected via network API at 192.168.1.111

Note: You need a relay that can receive HTTP GET requests for turning the light on/off.

Software Requirements

Python 3.x

Libraries:

opencv-python

ultralytics (for YOLOv8)

requests

Hardware: Any PC/laptop with a webcam.

Install required libraries via pip:

pip install opencv-python ultralytics requests

Usage

Run the script:

python smart_light_yolo.py


The program runs two threads:

Camera Thread: continuously captures frames from the webcam.

Processing Thread: detects people with YOLOv8, controls the light, and displays annotated video.

Controls the light automatically:

Turns on if at least one person is detected.

Turns off if no person is detected.

Press 'q' to exit the program.

Code Overview
# Pseudo-overview
# 1. Camera Thread: captures frames into shared variable
# 2. Processing Thread: runs YOLOv8 detection
# 3. Count number of "person" detected
# 4. Send HTTP request to light relay if needed
# 5. Display annotated frames


Uses threading.Lock() to safely share frames between threads.

YOLOv8 model used: yolov8n.pt (lightweight and fast).

Prevents repeated light toggling by storing the current_light_state.

Example Output (Serial / Console)
Detected 1 person
Light turned ON
Detected 0 persons
Light turned OFF


Bounding boxes of detected persons are shown in the camera window.

Notes

Ensure the relay’s IP and API matches your network configuration.

Adjust YOLOv8 model (yolov8n.pt) if you need higher accuracy or speed.

Program is thread-safe and designed for real-time operation.

License

Open-source and free to use under MIT License.
