"""
ANPR: YOLO (عربيات) + YOLO (لوحات) + fast-plate-ocr + HUD.

القاعدة: العربية مبتظهرش في الـ HUD غير لما لوحتها تكون واضحة
ورقمها يتقرا نفس القراءة CONFIRM_READS مرات.

+ إرسال كل عربية مؤكدة للـ Backend API (في thread منفصل).
"""
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from stream_server import publish_frame, start_stream_server
import cv2
import numpy as np
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer
from api_client import send_vehicle
from pathlib import Path


# ============================ Settings ============================

# Project root directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Models directory
MODELS_DIR = PROJECT_DIR/"vision" / "models"

VEHICLE_MODEL = str(MODELS_DIR / "yolov8n.pt")
PLATE_MODEL = str(MODELS_DIR / "best.pt")
FP_MODEL = "cct-s-v1-global-model"
VIDEO_PATH = str(MODELS_DIR / "2103099-uhd_3840_2160_30fps.mp4")


# Backend API
CAMERA_ID = "cam1"
SEND_TO_API = True             # False: يشتغل زي الأول من غير ما يبعت للـ API

FRAME_SIZE = (960, 540)
VEHICLE_CLASSES = [2, 3, 5, 7]

# Tracker (ByteTrack): بيستخدم الكشوفات الضعيفة عشان يكمّل نفس العربية بنفس الـ ID
TRACK_MIN_CONF = 0.10          # لازم واطية؛ لو رفعتها الـ tracker بيضيّع العربية ويديها ID جديد
TRACKER_FILE = "my_tracker.yaml"
TRACKER_CFG = """tracker_type: bytetrack
track_high_thresh: 0.40
track_low_thresh: 0.10
new_track_thresh: 0.50
track_buffer: 60
match_thresh: 0.80
fuse_score: True
"""
ORI_POINT = (475, 520)         # النقطة المرجعية (أسفل المنتصف)
MAX_DISTANCE = 220             # العربيات الأبعد من كده بتتجاهل
SHOW_DISTANCE = True

# اللوحة الواضحة (بكسل في الفريم الأصلي)
PLATE_CONF = 0.40
PLATE_MIN_W, PLATE_MIN_H = 70, 18
PLATE_ASPECT = (1.8, 6.5)      # عرض/ارتفاع مقبول
PLATE_MIN_SHARP = 60.0         # قلّلها لو مفيش حاجة بتعدّي، وارفعها لو بتعدّي لوحات مش واضحة
PLATE_EDGE_PAD = 2             # لوحة لازقة في حافة قصة العربية = مقطوعة

# الرقم
OCR_INTERVAL = 3               # أقل عدد فريمات بين قراءتين لنفس العربية
OCR_MAX_TRIES = 12
OCR_MIN_CONF = 0.50
TEXT_LEN = (5, 10)
NEED_LETTER_AND_DIGIT = True
CONFIRM_READS = 2              # نفس الرقم كام مرة عشان العربية تظهر في الـ HUD
LOCK_READS = 4                 # بعدها نقفل الرقم ونوقف الكشف والـ OCR للعربية دي

# HUD
MAX_HISTORY = 8
THUMB_V, THUMB_P = (192, 140), (240, 80)
SHOW_V, SHOW_P = (96, 70), (84, 28)
PANEL_W, MARGIN, HEADER_H = 300, 16, 44
CARD_H, CARD_GAP, CARD_PAD = 130, 8, 12

DEBUG = False                  # True: يطبع أسباب رفض اللوحات كل 120 فريم

YELLOW, WHITE, BLACK, GRAY = (0, 215, 255), (255, 255, 255), (0, 0, 0), (110, 110, 110)
CARD_BG = (45, 40, 35)
FONT = cv2.FONT_HERSHEY_DUPLEX

rejects = Counter()


# ============================ Detection ============================

def get_vehicles(car_model, image):
    """العربيات (ليها tracker id) اللي جوه MAX_DISTANCE من النقطة المرجعية."""
    H, W = image.shape[:2]
    result = car_model.track(image, persist=True, classes=VEHICLE_CLASSES,
                             verbose=False, conf=TRACK_MIN_CONF, tracker=TRACKER_FILE)[0]
    out = []
    for b in (result.boxes if result.boxes is not None else []):
        if b.id is None:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        dist = int(math.dist(ORI_POINT, ((x1 + x2) // 2, (y1 + y2) // 2)))
        if dist < MAX_DISTANCE:
            out.append({"id": int(b.id[0].item()), "box": (x1, y1, x2, y2), "dist": dist,
                        "conf": float(b.conf[0]), "name": car_model.names[int(b.cls[0])]})
    return out


def sharpness(img, width=128):
    """الحدة (variance of Laplacian) بعد توحيد العرض، عشان تتقارن بين لوحات بأحجام مختلفة."""
    h, w = img.shape[:2]
    img = cv2.resize(img, (width, max(1, int(width * h / w))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def find_clear_plate(plate_model, vehicle):
    """أوضح لوحة في قصة العربية، أو None لو مفيش لوحة عدّت شروط الوضوح."""
    res = plate_model.predict(vehicle, conf=PLATE_CONF, imgsz=320, device="cpu", verbose=False)[0]
    vh, vw = vehicle.shape[:2]
    boxes = res.boxes if res.boxes is not None else []
    if len(boxes) == 0:
        rejects["no_plate"] += 1

    best = None
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(vw, x2), min(vh, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop, w, h = vehicle[y1:y2, x1:x2], x2 - x1, y2 - y1

        reason, sharp = None, 0.0
        if w < PLATE_MIN_W or h < PLATE_MIN_H:
            reason = "too_small"
        elif not PLATE_ASPECT[0] <= w / h <= PLATE_ASPECT[1]:
            reason = "bad_aspect"
        elif (x1 <= PLATE_EDGE_PAD or y1 <= PLATE_EDGE_PAD or
              x2 >= vw - PLATE_EDGE_PAD or y2 >= vh - PLATE_EDGE_PAD):
            reason = "cut_by_edge"
        else:
            sharp = sharpness(crop)
            if sharp < PLATE_MIN_SHARP:
                reason = "blurry"

        if reason:
            rejects[reason] += 1
            continue
        score = (sharp + 1) * math.sqrt(w * h) * float(b.conf[0])
        if best is None or score > best["score"]:
            best = {"crop": crop, "box": (x1, y1, x2, y2), "score": score}
    return best


# ============================ OCR ============================

def read_plate(ocr, crop):
    """(نص, ثقة). النص فاضي لو القراءة مش منطقية (قصير/طويل/مفيهوش حروف وأرقام/ثقة واطية)."""
    h, w = crop.shape[:2]
    px, py = int(w * 0.06), int(h * 0.06)
    img = cv2.copyMakeBorder(crop, py, py, px, px, cv2.BORDER_REPLICATE)
    mode = ocr.config.image_color_mode
    if mode == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == "grayscale":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pred = ocr.run_one(img)
    text = str(pred.plate).replace("_", "").strip().upper()

    probs = getattr(pred, "char_probs", None)      # لو النسخة بترجّع ثقة لكل حرف
    conf = 1.0
    if probs is not None and len(probs) and text:
        conf = float(np.mean(np.asarray(probs, dtype=float).reshape(-1)[:len(text)]))

    ok = TEXT_LEN[0] <= len(text) <= TEXT_LEN[1] and text.isalnum() and conf >= OCR_MIN_CONF
    if ok and NEED_LETTER_AND_DIGIT:
        ok = any(c.isalpha() for c in text) and any(c.isdigit() for c in text)
    return (text if ok else ""), conf


def vote(track, text, conf):
    """تصويت على القراءات. بيأكّد العربية بعد CONFIRM_READS وبيقفل بعد LOCK_READS."""
    track["votes"][text] = track["votes"].get(text, 0.0) + conf
    track["counts"][text] = track["counts"].get(text, 0) + 1
    track["text"] = max(track["votes"], key=track["votes"].get)
    n = track["counts"][track["text"]]
    track["confirmed"] = track["confirmed"] or n >= CONFIRM_READS
    track["locked"] = track["locked"] or n >= LOCK_READS


# ============================ Drawing ============================

def draw_corners(frame, box, color=WHITE, thickness=2, frac=0.22):
    x1, y1, x2, y2 = box
    L = max(8, int(min(x2 - x1, y2 - y1) * frac))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, thickness, cv2.LINE_AA)


def draw_label(frame, text, x, y):
    cv2.putText(frame, text, (x, max(25, y - 10)), FONT, 0.55, WHITE, 2, cv2.LINE_AA)


def fit(img, size):
    """تصغير مع الحفاظ على النسبة + padding (من غير تشويه)."""
    tw, th = size
    h, w = img.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    canvas = np.full((th, tw, 3), CARD_BG, np.uint8)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = r
    return canvas


def paste(dst, src, x, y):
    """لصق صورة جوه أخرى مع قص اللي يطلع بره الحدود."""
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 < x2 and y1 < y2:
        dst[y1:y2, x1:x2] = src[y1 - y:y2 - y, x1 - x:x2 - x]


def draw_hud(frame, history, active_ids, total):
    """بانل على يمين الفريم: كارت لكل عربية اتأكدت (صورة العربية + بياناتها + اللوحة + الرقم)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = w - PANEL_W - MARGIN, MARGIN, w - MARGIN, h - MARGIN
    ph, pw = y2 - y1, x2 - x1

    panel = np.full((ph, pw, 3), (30, 25, 20), np.uint8)
    cv2.putText(panel, "VEHICLE ANALYTICS HUD - REALTIME", (12, 24), FONT, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, f"{total} vehicles", (12, 40), FONT, 0.35, YELLOW, 1, cv2.LINE_AA)
    cv2.line(panel, (12, HEADER_H - 2), (pw - 12, HEADER_H - 2), GRAY, 1)

    cards = panel[HEADER_H:]                      # view على الجزء الخاص بالكروت
    cw = pw - CARD_PAD * 2
    for i, rec in enumerate(history):
        rec["y"] += (i * (CARD_H + CARD_GAP) + 6 - rec["y"]) * 0.25      # حركة انزلاق ناعمة
        y, x = int(rec["y"]), CARD_PAD
        active = rec["display_id"] in active_ids
        border = YELLOW if active else GRAY

        cv2.rectangle(cards, (x, y), (x + cw, y + CARD_H), CARD_BG, -1)
        cv2.rectangle(cards, (x, y), (x + cw, y + CARD_H), border, 2 if active else 1)
        paste(cards, fit(rec["vehicle"], SHOW_V), x + 8, y + 8)
        cv2.rectangle(cards, (x + 8, y + 8), (x + 8 + SHOW_V[0], y + 8 + SHOW_V[1]), border, 1)

        info = [f"ID: {rec['display_id']:03d}", f"Type: {rec['name']}", f"Conf: {rec['conf']:.0%}"]
        for k, t in enumerate(info):
            cv2.putText(cards, t, (x + 114, y + 26 + k * 22), FONT, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

        sy = y + 86
        cv2.rectangle(cards, (x + 8, sy), (x + cw - 8, sy + 36), YELLOW, -1)
        scale = 0.6
        while scale > 0.35 and cv2.getTextSize(rec["text"], FONT, scale, 1)[0][0] > cw - 26 - SHOW_P[0]:
            scale -= 0.05
        cv2.putText(cards, rec["text"], (x + 14, sy + 24), FONT, scale, BLACK, 1, cv2.LINE_AA)
        paste(cards, fit(rec["plate"], SHOW_P), x + cw - 8 - SHOW_P[0], sy + 4)

    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = cv2.addWeighted(panel, 0.8, roi, 0.2, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), GRAY, 1)


# ============================ Main ============================

def main():
    start_stream_server()
    car_model, plate_model = YOLO(VEHICLE_MODEL), YOLO(PLATE_MODEL)
    ocr = LicensePlateRecognizer(FP_MODEL, device="cpu")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video!")

    with open(TRACKER_FILE, "w") as f:           # إعدادات الـ tracker
        f.write(TRACKER_CFG)

    tracks, history, frame_count = {}, [], 0
    sender = ThreadPoolExecutor(max_workers=1)   # الإرسال للـ API في thread منفصل عشان الفيديو ما يهنجش
    plates_seen = {}                             # الرقم -> ID العرض (يمنع تكرار نفس العربية)
    next_id = 1                                  # ID العرض: 1، 2، 3... بالترتيب

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        image = cv2.resize(frame, FRAME_SIZE)
        sx, sy = frame.shape[1] / FRAME_SIZE[0], frame.shape[0] / FRAME_SIZE[1]
        vehicles = get_vehicles(car_model, image)

        for v in vehicles:
            x1, y1, x2, y2 = v["box"]
            draw_corners(image, v["box"], YELLOW)
            if SHOW_DISTANCE:
                cv2.putText(image, f"{v['dist']}px", ((x1 + x2) // 2, (y1 + y2) // 2),
                            FONT, 0.45, WHITE, 1, cv2.LINE_AA)

            # 1) سجل العربية
            if v["id"] not in tracks:
                tracks[v["id"]] = {"id": v["id"], "name": v["name"], "conf": v["conf"],
                                   "votes": {}, "counts": {}, "text": "", "tries": 0,
                                   "last_ocr": -OCR_INTERVAL, "score": 0.0, "y": -CARD_H, "display_id": None,
                                   "vehicle": None, "plate": None,
                                   "vehicle_raw": None, "plate_raw": None,
                                   "confirmed": False, "locked": False, "shown": False}
            t = tracks[v["id"]]
            t["last_seen"] = frame_count

            # 2) الرقم اتقفل: نعرضه من غير كشف ولا OCR
            if t["locked"]:
                draw_label(image, f"{t['display_id']}- {t['text']}", x1, y1)
                continue

            # 3) لوحة واضحة؟ (القص من الفريم الأصلي 4K) لو لأ: متتحسبش
            vehicle = frame[int(y1 * sy):int(y2 * sy), int(x1 * sx):int(x2 * sx)]
            if vehicle.size == 0:
                continue
            plate = find_clear_plate(plate_model, vehicle)
            if plate is None:
                continue

            # 4) نحتفظ بأوضح لحظة (صورة اللوحة + العربية في نفس اللحظة)
            #    + نسخ أصلية عالية الجودة للإرسال للـ API (.copy() عشان ما نمسكش فريم 4K كامل في الذاكرة)
            if plate["score"] > t["score"]:
                t.update(score=plate["score"], conf=v["conf"],
                         plate=fit(plate["crop"], THUMB_P), vehicle=fit(vehicle, THUMB_V),
                         plate_raw=plate["crop"].copy(), vehicle_raw=vehicle.copy())

            # 5) OCR + تصويت
            if t["tries"] < OCR_MAX_TRIES and frame_count - t["last_ocr"] >= OCR_INTERVAL:
                t["tries"] += 1
                t["last_ocr"] = frame_count
                try:
                    text, conf = read_plate(ocr, plate["crop"])
                    if text and len(text)==7 :
                        vote(t, text, conf)
                except Exception as e:
                    print("[OCR ERROR]", e)

            # 6) لما الرقم يتأكد: ID عرض جديد، أو نفس ID لو الرقم ده اتشاف قبل كده
            if t["confirmed"] and not t["shown"]:
                t["shown"] = True
                if t["text"] in plates_seen:             # نفس العربية رجعت بـ tracker id تاني
                    t["display_id"] = plates_seen[t["text"]]
                else:
                    t["display_id"] = plates_seen[t["text"]] = next_id
                    next_id += 1
                    history.insert(0, t)
                    del history[MAX_HISTORY:]

                    # إرسال العربية المؤكدة للـ Backend (مرة واحدة لكل رقم جديد)
                    if SEND_TO_API:
                        sender.submit(
                            send_vehicle,
                            plate_number=t["text"],
                            plate_confidence=min(1.0, t["votes"][t["text"]] / t["counts"][t["text"]]),
                            camera_id=CAMERA_ID,
                            track_id=t["id"],
                            vehicle_img=t["vehicle_raw"],
                            plate_img=t["plate_raw"],
                            vehicle_type=t["name"],
                        )

            # 7) رسم اللوحة على الفيديو
            px1, py1, px2, py2 = plate["box"]
            draw_corners(image, (x1 + int(px1 / sx), y1 + int(py1 / sy),
                                 x1 + int(px2 / sx), y1 + int(py2 / sy)), WHITE, 2, 0.25)
            draw_label(image, f"{t['display_id']}- {t['text']}" if t["shown"] else "Reading...",
                       x1 + int(px1 / sx), y1 + int(py1 / sy))

        cv2.circle(image, ORI_POINT, 6, YELLOW, -1, cv2.LINE_AA)
        cv2.circle(image, ORI_POINT, 10, YELLOW, 1, cv2.LINE_AA)
        active = {tracks[v["id"]]["display_id"] for v in vehicles}
        draw_hud(image, history, active, next_id - 1)

        if frame_count % 150 == 0:          # تنضيف السجلات القديمة
            tracks = {i: t for i, t in tracks.items() if frame_count - t["last_seen"] < 600}
        if DEBUG and frame_count % 120 == 0:
            print("[REJECT]", dict(rejects))

        publish_frame(image)

    cap.release()
    sender.shutdown(wait=True)               # يستنى أي إرسال لسه شغال قبل الخروج
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()