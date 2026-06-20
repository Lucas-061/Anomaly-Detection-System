<div align="center">

# Anomaly-Detection-System

**Camera-Based Indoor Abnormal Behavior Recognition System**

[![English](https://img.shields.io/badge/README-English-blue)](./README_en.md)
[![Simplified Chinese](https://img.shields.io/badge/README-Simplified%20Chinese-red)](./README.md)

</div>

## 1. Project Overview

Anomaly-Detection-System is an indoor safety monitoring system for camera-based abnormal behavior recognition. It is designed for smart homes, indoor equipment rooms, laboratories, children's activity areas, industrial workstations, and other indoor safety-control scenarios.

The system captures video streams in real time with OpenCV, detects human targets with YOLOv8, and combines target tracking, virtual fence analysis, and rule-based behavior recognition to identify and warn about abnormal behaviors.

Compared with traditional physical fences, which are fixed, less flexible, and difficult to manage intelligently, this system allows users to draw virtual safety zones in software. The monitored boundary can be adjusted for different scenarios, and the system can detect abnormal behaviors such as illegal intrusion, long-term lingering, fence crossing, falling, running/fighting, and climbing.


## System Demo

### Fall Detection

![Fall detection demo](./docs/images/show_fall_down.jpg)

### Intrusion and Fence-Crossing Detection

![Intrusion and fence-crossing detection demo](./docs/images/show_intrusion.jpg)

## 2. Design Goals

The system is designed to provide the following main functions:

1. Real-time camera video capture and display.
2. Human target detection based on an object detection model.
3. User-defined virtual safety fence boundaries.
4. Real-time detection of whether a person enters the virtual fence area.
5. Target tracking with person IDs and movement trajectories.
6. Abnormal behavior recognition, including:
   - Illegal intrusion
   - Fence crossing
   - Long-term lingering
   - Fall detection
   - Running or fighting
   - Climbing
7. Warning prompts when abnormal behavior is detected.
8. Alarm screenshot saving and alarm record logging.
9. Optional extended functions:
   - Video saving and playback
   - Multi-level warnings
   - Graphical user interface

## 3. System Architecture

The overall workflow is shown below:

```text
+------------------+
| Camera/Video Src |
+--------+---------+
         |
         v
+------------------+
| OpenCV Capture   |
+--------+---------+
         |
         v
+------------------+
| YOLO Detection   |
+--------+---------+
         |
         v
+------------------+
| Tracking/Traj.   |
+--------+---------+
         |
         v
+------------------+
| Virtual Fence    |
+--------+---------+
         |
         v
+------------------+
| Behavior Rules   |
+--------+---------+
         |
         v
+------------------+
| Alarm/Save/Log   |
+------------------+
```

Core processing flow:

```text
Read camera frame
    -> Detect human targets
    -> Assign person ID
    -> Record movement trajectory
    -> Check virtual fence status
    -> Detect abnormal behavior
    -> Display alarm message and save records
```

## 4. Project Structure

```text
Anomaly-Detection-System/
|
|-- main.py                  # Program entry point
|-- ui_main.py               # PyQt6 main interface
|-- detector.py              # YOLO human detection module
|-- tracker.py               # Target tracking module
|-- fence.py                 # Virtual fence drawing and region checking
|-- behavior.py              # Abnormal behavior recognition module
|-- alarm.py                 # Alarm, screenshot, and logging module
|-- video_clip.py            # Alarm video clip recording module
|
|-- models/                  # Model files
|   |-- yolov8n.pt
|
|-- records/                 # Runtime records
|   |-- screenshots/         # Alarm screenshots
|   |-- videos/              # Alarm video clips
|   |-- alarm_log.csv        # Alarm log
|
|-- TrainVedio/              # Default video selection directory
|-- README.md                # English documentation
|-- README_en.md             # English documentation
|-- requirements.txt         # Python dependencies
```

## 5. Functional Modules

### 5.1 Video Capture

OpenCV is used to open a camera or read a local video file and provide video frames to the detection module.

Main functions:

- Open a camera device.
- Read real-time frames.
- Support local video testing.
- Pass frames to the detection pipeline.

### 5.2 Human Detection

YOLO is used to detect human targets in each video frame. Only the `person` class is retained.

Main functions:

- Load the YOLO model.
- Detect human targets frame by frame.
- Output bounding boxes, confidence scores, and class information.
- Draw detection boxes on the video frame.

### 5.3 Virtual Fence

Users can draw a virtual safety area with mouse clicks.

Main functions:

- Left-click to add fence vertices.
- Right-click to clear the fence.
- Connect multiple points into a polygon.
- Determine whether the center point of a person is inside the fence.

### 5.4 Target Tracking

Detected people are assigned IDs and tracked over time.

Main functions:

- Assign a unique ID to each person.
- Record historical center points.
- Estimate movement speed.
- Determine whether a person crosses the fence boundary.

### 5.5 Abnormal Behavior Recognition

Rules are applied based on bounding boxes, trajectories, speed, and stay duration.

| Abnormal Behavior | Rule Basis |
| --- | --- |
| Illegal intrusion | The human center point enters the virtual fence area. |
| Fence crossing | A person moves from outside to inside and the trajectory crosses the fence boundary. |
| Long-term lingering | A person stays inside the fence area longer than the threshold. |
| Fall detection | The bounding-box aspect ratio becomes abnormal and remains in a low posture for several frames. |
| Running or fighting | Movement speed exceeds the threshold, or multiple people move quickly while staying too close. |
| Climbing | The body position continues to move upward, or the upper-body height changes significantly. |

### 5.6 Alarm Module

When abnormal behavior is detected, the system triggers warning feedback.

Main functions:

- Display alarm text on the video frame.
- Save alarm screenshots.
- Record alarm time, type, person ID, warning level, screenshot path, and video clip path.
- Support multi-level warnings.

### 5.7 PyQt6 GUI

The graphical interface improves usability and provides basic operation controls.

The interface can include:

- Real-time monitoring view.
- Recognition mode selection buttons.
- Play button.
- Exit recognition button.
- Alarm record viewer.
- Alarm record table.
- Current alarm status display.

## 6. Technology Stack

| Technology | Purpose |
| --- | --- |
| Python | Main development language |
| OpenCV | Camera capture, image processing, video display, and region drawing |
| PyQt6 | GUI development |
| YOLOv8 | Human target detection |
| NumPy | Coordinate calculation and array processing |
| CSV | Alarm record storage |

Potential extensions:

| Technology | Extended Function |
| --- | --- |
| OpenCV VideoWriter | Save videos with detection boxes |
| SQLite / MySQL | Alarm record querying, statistics, and long-term storage |
| MediaPipe Pose / YOLOv8-Pose | Keypoint-based fall and climbing recognition |

## 7. Dependencies

Main dependencies in `requirements.txt`:

```text
opencv-python
numpy
PyQt6
ultralytics
```

## 8. Implementation Steps

Recommended development order:

1. Implement real-time camera reading.
2. Integrate YOLO for human detection.
3. Implement mouse-based virtual fence drawing.
4. Detect whether a person enters the fence.
5. Add target tracking with person IDs and trajectories.
6. Implement long-term lingering and fence-crossing detection.
7. Implement fall, running, and climbing detection.
8. Add alarm screenshot and alarm record saving.
9. Build the PyQt6 graphical interface.

## 9. Minimum Viable Version

A minimum viable version should include:

```text
Real-time camera reading
    + YOLO human detection
    + Mouse-drawn virtual fence
    + Intrusion alarm
    + Alarm screenshot saving
```

After the minimum version is completed, the system can be extended with:

```text
Target tracking
Long-term lingering detection
Fence-crossing detection
Fall detection
Running detection
Climbing detection
PyQt6 graphical interface
```

The main functions above have already been implemented. Future extensions may include:

```text
Processed video saving
Historical video playback
Multi-level warnings
Pose-keypoint-based action recognition
Database storage and querying
```

## 10. Expected Results

After completion, the system should be able to:

- Display camera frames in real time.
- Detect and mark people in the scene.
- Allow users to define virtual safety fences.
- Trigger alarms when a person enters a dangerous area.
- Recognize abnormal behaviors through rule-based analysis.
- Save alarm screenshots and alarm records.
- Provide basic operations through a graphical interface.

## 11. Current Minimum System

The current code implements a PyQt6-based minimum abnormal behavior recognition system with the following functions:

- Camera or video file input.
- YOLOv8 human detection, with automatic fallback to OpenCV HOG if YOLO fails to load.
- Target tracking based on position, size, and color histogram features.
- Person ID assignment and continuous localization of the same person.
- Fall detection.
- Running detection.
- Climbing detection.
- Multi-level warning display.
- Alarm screenshot saving.
- Alarm log recording.
- Initial interface prompt: `Please select a recognition mode`.
- Right-side buttons for camera recognition or folder video recognition.
- Folder video recognition opens the `TrainVedio` directory by default, allowing the user to select a single video file.

## 12. Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the graphical interface:

```bash
python main.py
```

## 13. Operation Guide

```text
After startup: the video display area shows "Please select a recognition mode"
Recognition mode -> Camera recognition: open the camera for real-time recognition
Recognition mode -> Folder video recognition: open the TrainVedio folder and manually select a video
After selecting a video: the first frame is displayed, but playback does not start automatically
Left-click inside the video area: add virtual fence points
Right-click inside the video area: clear the virtual fence
Play: start recognition for the selected video
After video recognition finishes: click Play again to re-run recognition on the same video
Alarm Records: open a graphical table window to view historical alarm CSV records
Exit Recognition: stop the current recognition task and return to the home screen
Top-right X: exit the entire program
```

Folder video recognition rules:

```text
Video directory: TrainVedio/
The user selects one video file each time
Only the selected video is recognized each time
After recognition finishes, click Play to recognize the current video again
After recognition finishes, click Exit Recognition to return to the home screen
Click Folder Video Recognition again to choose another video
```

Supported video formats:

```text
.mp4
.avi
.mov
.mkv
.wmv
.flv
.m4v
```

## 14. Detection Backend

The system uses YOLOv8 for human detection first. If `ultralytics` is not installed or the model fails to load, the program automatically falls back to OpenCV HOG human detection.

Recommended YOLOv8 installation:

```bash
pip install ultralytics
```

To use a local model, place the model file at:

```text
models/yolov8n.pt
```

If the following file exists in the project root, it will also be loaded automatically:

```text
yolov8n.pt
```

## 15. Same-Person Tracking

The tracking module no longer relies only on center-point distance. It combines the following information to determine whether detections belong to the same person:

```text
Human center-point distance
Human bounding-box size change
HSV color histogram features of the human region
```

This makes person ID assignment more stable when multiple people appear at the same time, short occlusion occurs, or targets move quickly.

## 16. Output Files

Alarm screenshot directories:

```text
records/screenshots/intrusion/      # Illegal intrusion screenshots
records/screenshots/cross_fence/    # Fence-crossing screenshots
records/screenshots/fall_down/      # Fall screenshots
```

Alarm video clip directories:

```text
records/videos/intrusion/           # Illegal intrusion clips
records/videos/cross_fence/         # Fence-crossing clips
records/videos/fall_down/           # Fall clips
```

Alarm log file:

```text
records/alarm_log.csv
```

Alarm log fields:

```text
time,source,track_id,alarm_type,alarm_name,level,screenshot,video_clip
```

## 17. Multi-Level Warnings

The system divides abnormal behaviors into three warning levels according to risk severity:

| Warning Level | Trigger Behavior | Interface Display |
| --- | --- | --- |
| Level 1 | Running, long-term lingering | Yellow warning |
| Level 2 | Illegal intrusion, climbing | Orange warning |
| Level 3 | Fence crossing, fall detection | Red warning |

Multi-level warnings are displayed in the real-time status label, the right-side data panel, and the alarm record table. They are also written to the `level` field in the alarm log.

The system currently saves screenshots, alarm video clips, and CSV records for the following alarm types:

```text
intrusion    Illegal intrusion
cross_fence  Fence crossing
fall_down    Fall detection
```

The screenshot is the recognition frame at the moment the alarm is triggered, including the human bounding box, person ID, virtual fence, and alarm text.

Alarm video clips are saved using a "3 seconds before alarm + 5 seconds after alarm" strategy. If the video starts or ends before the full duration can be collected, the actual available segment is saved.

To reduce UI stutter and memory usage during long-running sessions, alarm clip recording uses the following protection strategies:

```text
Video clips are written in a background thread to avoid blocking the PyQt6 UI
At most 3 alarm clips can be recorded at the same time
Alarm clip FPS is capped at 20 FPS
If screenshot, CSV, or clip saving fails, the right-side data panel displays an error message
```

## 18. Current Code Files

```text
main.py          # Program entry point
ui_main.py       # PyQt6 graphical interface
detector.py      # Human detection, with YOLO and HOG fallback
tracker.py       # Simple centroid-based target tracking
fence.py         # Virtual fence drawing and region checking
behavior.py      # Abnormal behavior rule detection
alarm.py         # Alarm screenshot and log saving
video_clip.py    # Alarm video clip saving before and after alarms
```

