import threading

import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse


app = FastAPI(title="Smart Traffic Video Stream")

# آخر Frame فقط
latest_frame = None
frame_id = 0

# مزامنة بين Detector والمتصفح
frame_condition = threading.Condition()


def publish_frame(frame):
    """
    استقبال آخر Frame بعد رسم HUD وتحويله إلى JPEG.
    لا نحتفظ بفريمات قديمة.
    """
    global latest_frame, frame_id

    # تصغير الفريم لتقليل حجم البيانات
    if frame.shape[1] > 960:
        frame = cv2.resize(
            frame,
            (960, 540),
            interpolation=cv2.INTER_AREA
        )

    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            65
        ]
    )

    if not success:
        return

    with frame_condition:
        latest_frame = buffer.tobytes()
        frame_id += 1

        # إبلاغ المتصفح بوصول Frame جديد
        frame_condition.notify_all()


def generate_frames():
    """
    إرسال كل Frame جديد مرة واحدة فقط.
    """
    last_sent_id = -1

    while True:

        with frame_condition:

            # انتظار وصول Frame جديد
            frame_condition.wait_for(
                lambda: (
                    latest_frame is not None
                    and frame_id != last_sent_id
                ),
                timeout=10
            )

            if latest_frame is None:
                continue

            frame = latest_frame
            current_id = frame_id

        last_sent_id = current_id

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
            + frame
            + b"\r\n"
        )


@app.get("/")
def home():
    return {
        "status": "Video stream server is running",
        "stream": "/video_feed"
    }


@app.get("/video_feed")
def video_feed():

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


def start_stream_server():

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="warning"
    )

    server = uvicorn.Server(config)

    thread = threading.Thread(
        target=server.run,
        daemon=True
    )

    thread.start()

    print("[STREAM] Server running on http://localhost:8001")

    return thread