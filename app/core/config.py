from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./velocity_alerts.db"
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    llm_model: str = ""
    instagram_session_id: str = ""
    firebase_credentials_path: str = ""

    # Apify token for the Instagram profile-scraper fallback. When IG
    # rate-limits the public web_profile_info endpoint (which it always
    # does from Railway's cloud IPs), the verifier falls back to Apify's
    # `apify/instagram-profile-scraper` actor for residential-IP lookups.
    apify_token: str = ""

    polling_interval_minutes: int = 30
    velocity_spike_threshold: float = 2.5
    alert_cooldown_hours: int = 6

    # How many hours of data to consider "recent" for velocity calc
    velocity_window_hours: int = 6
    # Minimum views before a post is worth tracking
    min_views_threshold: int = 1000
    # How many historical posts to use for baseline
    baseline_post_count: int = 20

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
