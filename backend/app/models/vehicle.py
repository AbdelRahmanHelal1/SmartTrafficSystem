from datetime import datetime, timezone

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class VehicleBase(BaseModel):
    plate_number: str = Field(..., min_length=1, max_length=20)
    plate_confidence: float = Field(..., ge=0.0, le=1.0)
    vehicle_type: str | None = None  # car, truck, bus, motorcycle
    camera_id: str = Field(..., min_length=1)
    track_id: int = Field(..., ge=0)
    detected_at: datetime = Field(default_factory=utc_now)


class VehicleCreate(VehicleBase):
    vehicle_image_path: str | None = None
    plate_image_path: str | None = None


class VehicleResponse(VehicleBase):
    id: str
    vehicle_image_path: str | None = None
    plate_image_path: str | None = None