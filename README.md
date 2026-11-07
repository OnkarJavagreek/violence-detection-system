# AI Violence Detection System

A real-time violence detection web app built as my final-year Computer Engineering project. It reads a live webcam feed, runs a trained YOLO model on every frame, and highlights violent activity in the browser. Frames where violence is detected are saved automatically as screenshots.

## Features
- Live webcam streaming in the browser (Django `StreamingHttpResponse`, MJPEG)
- Frame-by-frame inference with a custom-trained YOLO model (`best.pt`) using Ultralytics
- Red bounding box and confidence label, plus a "VIOLENCE DETECTED!" banner on the video
- Detection threshold of 60% confidence to reduce false alarms
- Automatic screenshot saving to `media/` (at most one per second)
- Simple dark UI with a start page and a live detection page

## Tech stack
Python, Django, OpenCV, Ultralytics YOLO, HTML/CSS, SQLite (Django default)

## Project structure
```
violence-detection-system/
├── best.pt                     # trained YOLO model weights
├── manage.py
├── violence_detection_project/ # Django settings and URLs
└── detection_app/
    ├── views.py                # model loading, frame processing, streaming
    ├── urls.py                 # / , /predict/ , /video_feed/
    └── templates/              # home.html, prediction.html
```

## How it works
1. `home` page: "Start Real-Time Detection" button.
2. `predict` page: shows the live stream from `/video_feed/`.
3. `generate_frames()` captures each webcam frame with OpenCV and runs the YOLO model.
4. Detections of the violence class above the confidence threshold are drawn on the frame and saved to `media/`.
5. Frames are streamed back to the browser as an MJPEG stream.

## Setup and run
```bash
# 1. Clone the repo
git clone https://github.com/OnkarJavagreek/violence-detection-system.git
cd violence-detection-system

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server from the project root (best.pt must be in this folder)
python manage.py runserver
```
Open http://127.0.0.1:8000/ and allow webcam access.

Optional: set your own secret key with the `DJANGO_SECRET_KEY` environment variable.

## Configuration
In `detection_app/views.py`:
- `CONF_THRESHOLD`: minimum confidence to flag violence (default 0.60)
- `SAVE_INTERVAL`: minimum seconds between saved screenshots (default 1)
- `VIOLENCE_CLASS`: class index of "violence" in the trained model

## Model and results
- Model: YOLO (custom trained, weights in `best.pt`)
- Dataset: _add dataset name and source link here (the dataset is not included in this repo)_
- Results: _add precision / recall / mAP and a short note on test conditions_

## Screenshots
_Add 1-2 screenshots of the detection page here._

## Limitations and future work
- Runs on a single webcam stream and processes frames on the machine running the server
- Detection quality depends on lighting, camera angle and training data
- Ideas: video file upload, alert notifications (SMS / email), multi-camera support, a dashboard for saved detections

## Note
Built for academic and learning purposes. Violence detection models can produce false positives and negatives, so they should support, not replace, human judgement.
