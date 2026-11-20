from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status

from app.models.vehicle import VehicleCreate, VehicleResponse
from app.services import storage_service, vehicle_service

router = APIRouter(prefix="/vehicles", tags=["Vehicles"])


@router.post("", response_model=VehicleResponse, status_code=status.HTTP_201_CREATED)
async def create_vehicle(vehicle: VehicleCreate):
    return await vehicle_service.create_vehicle(vehicle)


@router.post(
    "/upload", response_model=VehicleResponse, status_code=status.HTTP_201_CREATED
)
async def create_vehicle_with_images(
    plate_number: str = Form(..., min_length=1, max_length=20),
    plate_confidence: float = Form(..., ge=0.0, le=1.0),
    camera_id: str = Form(..., min_length=1),
    track_id: int = Form(..., ge=0),
    vehicle_type: str | None = Form(None),
    vehicle_image: UploadFile = File(...),
    plate_image: UploadFile = File(...),
):
    vehicle_path = await storage_service.save_image(vehicle_image, "vehicles")
    plate_path = await storage_service.save_image(plate_image, "plates")

    vehicle = VehicleCreate(
        plate_number=plate_number,
        plate_confidence=plate_confidence,
        camera_id=camera_id,
        track_id=track_id,
        vehicle_type=vehicle_type,
        vehicle_image_path=vehicle_path,
        plate_image_path=plate_path,
    )
    return await vehicle_service.create_vehicle(vehicle)



@router.get("", response_model=list[VehicleResponse])
async def list_vehicles(
    plate_number: str | None = Query(None, description="بحث جزئي في رقم اللوحة"),
    camera_id: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = Query(100, ge=1, le=1000),
):
    return await vehicle_service.list_vehicles(
        plate_number, camera_id, start_time, end_time, limit
    )


@router.get("/{vehicle_id}", response_model=VehicleResponse)
async def get_vehicle(vehicle_id: str):
    vehicle = await vehicle_service.get_vehicle(vehicle_id)
    if vehicle is None:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return vehicle