import cv2
import os
from datetime import datetime
from django.shortcuts import render
from django.http import StreamingHttpResponse
from ultralytics import YOLO


model = YOLO("best.pt")

VIOLENCE_CLASS = 1
CONF_THRESHOLD = 0.60


MEDIA_FOLDER = "media"
os.makedirs(MEDIA_FOLDER, exist_ok=True)

last_saved_time = 0
SAVE_INTERVAL = 1   


def home(request):
    return render(request, "home.html")


def generate_frames():
    global last_saved_time

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        detected = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == VIOLENCE_CLASS and conf > CONF_THRESHOLD:
                    detected = True

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    label = f"Violence {conf*100:.1f}%"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)

        if detected:
            current_time = datetime.now().timestamp()

            if current_time - last_saved_time > SAVE_INTERVAL:
                filename = datetime.now().strftime("violence_%Y%m%d_%H%M%S.jpg")
                filepath = os.path.join(MEDIA_FOLDER, filename)

                cv2.imwrite(filepath, frame)
                print(f"[INFO] Screenshot saved: {filepath}")

                last_saved_time = current_time

            cv2.putText(frame, "VIOLENCE DETECTED!",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def video_feed(request):
    return StreamingHttpResponse(
        generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


def predict(request):
    return render(request, "prediction.html")