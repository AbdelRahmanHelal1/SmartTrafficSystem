import re
from datetime import datetime

from bson import ObjectId
from bson.errors import InvalidId

from app.database.mongodb import get_database
from app.models.vehicle import VehicleCreate, VehicleResponse


def _collection():
    return get_database()["vehicles"]


def _to_response(doc: dict) -> VehicleResponse:
    doc["id"] = str(doc.pop("_id"))
    return VehicleResponse(**doc)


async def create_vehicle(vehicle: VehicleCreate) -> VehicleResponse:
    doc = vehicle.model_dump()
    await _collection().insert_one(doc)  # بيضيف _id للـ doc تلقائيًا
    return _to_response(doc)


async def list_vehicles(
    plate_number: str | None = None,
    camera_id: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = 100,
) -> list[VehicleResponse]:
    query: dict = {}

    if plate_number:
        query["plate_number"] = {"$regex": re.escape(plate_number), "$options": "i"}
    if camera_id:
        query["camera_id"] = camera_id

    time_filter: dict = {}
    if start_time:
        time_filter["$gte"] = start_time
    if end_time:
        time_filter["$lte"] = end_time
    if time_filter:
        query["detected_at"] = time_filter

    cursor = _collection().find(query).sort("detected_at", -1).limit(limit)
    return [_to_response(doc) async for doc in cursor]


async def get_vehicle(vehicle_id: str) -> VehicleResponse | None:
    try:
        oid = ObjectId(vehicle_id)
    except InvalidId:
        return None
    doc = await _collection().find_one({"_id": oid})
    return _to_response(doc) if doc else None