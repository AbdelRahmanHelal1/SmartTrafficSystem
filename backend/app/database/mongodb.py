from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from app.config import settings

client: AsyncMongoClient | None = None


async def connect_to_mongo() -> None:
    global client
    client = AsyncMongoClient(
        settings.mongodb_url,
        serverSelectionTimeoutMS=3000,
        tz_aware=True,  # عشان التواريخ ترجع بـ UTC صريح مش naive
    )
    await client.admin.command("ping")

    vehicles = client[settings.mongodb_db_name]["vehicles"]
    await vehicles.create_index("plate_number")
    await vehicles.create_index("camera_id")
    await vehicles.create_index([("detected_at", -1)])

    
async def close_mongo_connection() -> None:
    global client
    if client is not None:
        await client.close()
        client = None


def get_database() -> AsyncDatabase:
    if client is None:
        raise RuntimeError("MongoDB client is not initialized")
    return client[settings.mongodb_db_name]