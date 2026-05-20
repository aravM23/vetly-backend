from datetime import datetime
from sqlalchemy import (
    String, Integer, Float, Text, Boolean, DateTime, ForeignKey, JSON, Enum as SAEnum
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from app.core.database import Base


class AlertUrgency(str, enum.Enum):
    CRITICAL = "critical"   # 5x+ multiplier, first 2 hours
    HIGH = "high"           # 3x+ multiplier, first 6 hours
    MEDIUM = "medium"       # 2.5x+ multiplier
    LOW = "low"             # notable but not urgent


class AlertStatus(str, enum.Enum):
    PENDING = "pending"
    SENT = "sent"
    OPENED = "opened"
    ACTED_ON = "acted_on"
    DISMISSED = "dismissed"
    EXPIRED = "expired"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    instagram_handle: Mapped[str | None] = mapped_column(String(255))
    content_pillars: Mapped[dict | None] = mapped_column(JSON)
    niche_tags: Mapped[list | None] = mapped_column(JSON)
    push_token: Mapped[str | None] = mapped_column(Text)
    notification_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    tracked_creators: Mapped[list["TrackedCreator"]] = relationship(back_populates="user")
    alerts: Mapped[list["VelocityAlert"]] = relationship(back_populates="user")


class TrackedCreator(Base):
    """A competitor/inspiration creator that a user is monitoring."""
    __tablename__ = "tracked_creators"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    instagram_handle: Mapped[str] = mapped_column(String(255), index=True)
    display_name: Mapped[str | None] = mapped_column(String(255))
    follower_count: Mapped[int | None] = mapped_column(Integer)
    avg_views: Mapped[float | None] = mapped_column(Float)
    avg_likes: Mapped[float | None] = mapped_column(Float)
    avg_comments: Mapped[float | None] = mapped_column(Float)
    last_scraped_at: Mapped[datetime | None] = mapped_column(DateTime)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="tracked_creators")
    posts: Mapped[list["CreatorPost"]] = relationship(back_populates="creator")


class CreatorPost(Base):
    """Individual post snapshot with engagement metrics over time."""
    __tablename__ = "creator_posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    creator_id: Mapped[int] = mapped_column(ForeignKey("tracked_creators.id"), index=True)
    instagram_post_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    post_url: Mapped[str | None] = mapped_column(Text)
    caption: Mapped[str | None] = mapped_column(Text)
    post_type: Mapped[str | None] = mapped_column(String(50))  # reel, carousel, image
    posted_at: Mapped[datetime | None] = mapped_column(DateTime)

    # Engagement at time of scrape
    views: Mapped[int] = mapped_column(Integer, default=0)
    likes: Mapped[int] = mapped_column(Integer, default=0)
    comments: Mapped[int] = mapped_column(Integer, default=0)
    shares: Mapped[int | None] = mapped_column(Integer)

    # Velocity metrics (calculated)
    view_velocity: Mapped[float | None] = mapped_column(Float)   # views per hour
    velocity_multiplier: Mapped[float | None] = mapped_column(Float)  # vs creator's avg
    hours_since_post: Mapped[float | None] = mapped_column(Float)
    is_spike: Mapped[bool] = mapped_column(Boolean, default=False)

    # Content analysis
    detected_format: Mapped[str | None] = mapped_column(String(100))  # listicle, storytime, etc.
    detected_hook_type: Mapped[str | None] = mapped_column(String(100))
    content_analysis: Mapped[dict | None] = mapped_column(JSON)

    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    creator: Mapped["TrackedCreator"] = relationship(back_populates="posts")
    snapshots: Mapped[list["PostSnapshot"]] = relationship(back_populates="post")


class PostSnapshot(Base):
    """Point-in-time engagement capture for velocity calculation."""
    __tablename__ = "post_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    post_id: Mapped[int] = mapped_column(ForeignKey("creator_posts.id"), index=True)
    views: Mapped[int] = mapped_column(Integer, default=0)
    likes: Mapped[int] = mapped_column(Integer, default=0)
    comments: Mapped[int] = mapped_column(Integer, default=0)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    post: Mapped["CreatorPost"] = relationship(back_populates="snapshots")


class VelocityAlert(Base):
    """The core output: a push-ready alert when a trend spike is detected."""
    __tablename__ = "velocity_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    post_id: Mapped[int] = mapped_column(ForeignKey("creator_posts.id"))
    creator_handle: Mapped[str] = mapped_column(String(255))

    # Spike data
    velocity_multiplier: Mapped[float] = mapped_column(Float)
    views_at_detection: Mapped[int] = mapped_column(Integer)
    hours_since_post: Mapped[float] = mapped_column(Float)
    detected_format: Mapped[str | None] = mapped_column(String(100))

    # The generated draft content
    alert_headline: Mapped[str] = mapped_column(Text)
    alert_body: Mapped[str] = mapped_column(Text)
    draft_hook: Mapped[str | None] = mapped_column(Text)
    draft_structure: Mapped[dict | None] = mapped_column(JSON)
    rewrite_rationale: Mapped[str | None] = mapped_column(Text)

    urgency: Mapped[str] = mapped_column(SAEnum(AlertUrgency), default=AlertUrgency.MEDIUM)
    status: Mapped[str] = mapped_column(SAEnum(AlertStatus), default=AlertStatus.PENDING)

    # Estimated window before the wave peaks
    estimated_peak_hours: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime)

    user: Mapped["User"] = relationship(back_populates="alerts")
    source_post: Mapped["CreatorPost"] = relationship()
