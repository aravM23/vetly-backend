from datetime import datetime
from pydantic import BaseModel, Field


class ContentPillars(BaseModel):
    primary_narrative: str = Field(description="The creator's overarching story/brand")
    topics: list[str] = Field(default_factory=list, description="Core content topics")
    tone: str = Field(default="", description="Brand voice/tone")
    audience: str = Field(default="", description="Target audience description")


class UserCreate(BaseModel):
    username: str
    instagram_handle: str | None = None
    content_pillars: ContentPillars | None = None
    niche_tags: list[str] = Field(default_factory=list)
    push_token: str | None = None


class UserResponse(BaseModel):
    id: int
    username: str
    instagram_handle: str | None
    content_pillars: dict | None
    niche_tags: list | None
    notification_enabled: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TrackedCreatorCreate(BaseModel):
    instagram_handle: str


class TrackedCreatorResponse(BaseModel):
    id: int
    instagram_handle: str
    display_name: str | None
    follower_count: int | None
    avg_views: float | None
    avg_likes: float | None
    avg_comments: float | None
    last_scraped_at: datetime | None
    is_active: bool

    model_config = {"from_attributes": True}


class PostVelocity(BaseModel):
    post_id: str
    post_url: str | None
    creator_handle: str
    views: int
    likes: int
    comments: int
    hours_since_post: float
    view_velocity: float
    velocity_multiplier: float
    detected_format: str | None
    is_spike: bool


class AlertResponse(BaseModel):
    id: int
    creator_handle: str
    velocity_multiplier: float
    views_at_detection: int
    hours_since_post: float
    detected_format: str | None
    alert_headline: str
    alert_body: str
    draft_hook: str | None
    draft_structure: dict | None
    rewrite_rationale: str | None
    urgency: str
    status: str
    estimated_peak_hours: float | None
    created_at: datetime
    sent_at: datetime | None

    model_config = {"from_attributes": True}


class AlertFeedResponse(BaseModel):
    alerts: list[AlertResponse]
    total: int
    pending_count: int


class VelocityFeedItem(BaseModel):
    creator_handle: str
    creator_name: str | None
    post_url: str | None
    caption_preview: str | None
    views: int
    velocity_multiplier: float
    hours_since_post: float
    detected_format: str | None
    is_spike: bool
    alert_generated: bool


class VelocityFeedResponse(BaseModel):
    items: list[VelocityFeedItem]
    spike_count: int
    last_scan_at: datetime | None


class TriggerScanRequest(BaseModel):
    user_id: int


class TriggerScanResponse(BaseModel):
    posts_scanned: int
    spikes_detected: int
    alerts_generated: int
    alerts: list[AlertResponse]
