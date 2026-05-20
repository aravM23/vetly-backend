from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./velocity_alerts.db"

    # LLM credentials. Either provider works; OpenRouter takes precedence
    # because it speaks the OpenAI-compatible API with one key giving access to
    # every model. `llm_model` is fully qualified (e.g. "openai/gpt-4o-mini",
    # "anthropic/claude-3.5-sonnet") when using OpenRouter.
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    llm_model: str = ""  # if blank we pick a sane default based on the active provider

    instagram_session_id: str = ""
    firebase_credentials_path: str = ""

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
