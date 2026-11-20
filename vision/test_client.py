import cv2
import numpy as np

from api_client import send_vehicle

# صور وهمية للاختبار (مكان فريم السيارة وقصاصة اللوحة الحقيقيين)
vehicle_img = np.full((300, 400, 3), 120, dtype=np.uint8)
plate_img = np.full((60, 200, 3), 255, dtype=np.uint8)
cv2.putText(plate_img, "TEST123", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

result = send_vehicle(
    plate_number="TEST123",
    plate_confidence=0.91,
    camera_id="cam1",
    track_id=99,
    vehicle_img=vehicle_img,
    plate_img=plate_img,
    vehicle_type="car",
)
print(result)