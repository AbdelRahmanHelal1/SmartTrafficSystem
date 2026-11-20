"""
Client لإرسال العربيات المؤكدة للـ Backend API.

لو الـ API واقع أو الشبكة قطعت: بيحاول كذا مرة، ولو فشل بيحفظ العربية على الـ disk
(مجلد PENDING_DIR) وبيبعتها أول ما الاتصال يرجع، فمفيش عربية بتضيع.
"""
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import cv2
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
PENDING_DIR = Path(os.getenv("PENDING_DIR", "pending"))
MAX_RETRIES = int(os.getenv("SEND_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("SEND_RETRY_BACKOFF", "2"))   # ثواني، بتتضاعف كل محاولة
REQUEST_TIMEOUT = float(os.getenv("SEND_TIMEOUT", "5"))

OK, RETRY, DROP = "ok", "retry", "drop"


def _encode_jpg(image) -> bytes:
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Failed to encode image")
    return buf.tobytes()


def _post(data: dict, vehicle_jpg: bytes, plate_jpg: bytes, timeout: float):
    """محاولة إرسال واحدة. يرجّع (الحالة، الـ response أو None)."""
    files = {
        "vehicle_image": ("vehicle.jpg", vehicle_jpg, "image/jpeg"),
        "plate_image": ("plate.jpg", plate_jpg, "image/jpeg"),
    }
    try:
        response = requests.post(
            f"{API_URL}/vehicles/upload", data=data, files=files, timeout=timeout
        )
    except requests.RequestException as e:           # انقطاع شبكة / timeout / السيرفر واقع
        print(f"[api_client] connection problem: {e}")
        return RETRY, None

    if response.status_code < 300:
        return OK, response.json()
    if response.status_code >= 500:                  # مشكلة مؤقتة في السيرفر
        print(f"[api_client] server error {response.status_code}")
        return RETRY, None
    # 4xx: الطلب نفسه غلط، ومهما كررناه مش هينجح
    print(f"[api_client] rejected {response.status_code}: {response.text[:200]}")
    return DROP, None


def _spool(data: dict, vehicle_jpg: bytes, plate_jpg: bytes) -> None:
    folder = PENDING_DIR / f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "vehicle.jpg").write_bytes(vehicle_jpg)
    (folder / "plate.jpg").write_bytes(plate_jpg)
    (folder / "data.json").write_text(json.dumps(data), encoding="utf-8")   # آخر حاجة: يعني الفولدر كامل
    print(f"[api_client] saved to pending: {folder.name}")


def flush_pending() -> int:
    """يبعت العربيات المحفوظة. يقف عند أول فشل اتصال. يرجّع عدد اللي اتبعت."""
    if not PENDING_DIR.exists():
        return 0

    sent = 0
    for folder in sorted(p for p in PENDING_DIR.iterdir() if p.is_dir()):
        data_file = folder / "data.json"
        if not data_file.exists():                   # فولدر ناقص (اتقطع وقت الحفظ)
            continue
        data = json.loads(data_file.read_text(encoding="utf-8"))
        status, _ = _post(
            data,
            (folder / "vehicle.jpg").read_bytes(),
            (folder / "plate.jpg").read_bytes(),
            REQUEST_TIMEOUT,
        )
        if status == RETRY:
            break                                    # السيرفر لسه واقع، نحاول بعدين
        shutil.rmtree(folder, ignore_errors=True)    # اتبعت (أو اترفض نهائيًا)
        if status == OK:
            sent += 1

    if sent:
        print(f"[api_client] resent {sent} pending vehicle(s)")
    return sent


def send_vehicle(
    plate_number: str,
    plate_confidence: float,
    camera_id: str,
    track_id: int,
    vehicle_img,
    plate_img,
    vehicle_type: str | None = None,
) -> dict | None:
    """يرسل سيارة مؤكدة للـ API. يرجّع الـ response أو None لو ما وصلتش (بتتحفظ للإرسال بعدين)."""
    data = {
        "plate_number": plate_number,
        "plate_confidence": str(plate_confidence),
        "camera_id": camera_id,
        "track_id": str(track_id),
    }
    if vehicle_type:
        data["vehicle_type"] = vehicle_type

    vehicle_jpg, plate_jpg = _encode_jpg(vehicle_img), _encode_jpg(plate_img)

    flush_pending()                                  # الأقدم الأول

    delay = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        status, result = _post(data, vehicle_jpg, plate_jpg, REQUEST_TIMEOUT)
        if status == OK:
            return result
        if status == DROP:
            return None
        if attempt < MAX_RETRIES:
            time.sleep(delay)
            delay *= 2

    _spool(data, vehicle_jpg, plate_jpg)
    return None