from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent  # مجلد backend


class Settings(BaseSettings):
    app_name: str = "Smart Traffic System API"
    app_version: str = "1.0.0"
    app_description: str = "AI-powered Vehicle and License Plate Recognition System"

    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "smart_traffic"

    storage_dir: Path = BASE_DIR.parent / "storage"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )



settings = Settings()