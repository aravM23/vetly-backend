"""
Timing models — self-contained schema for the niche timing chart demo.

Kept independent of the velocity-alerts CreatorPost pipeline so the two systems
can evolve on their own. Every row here represents a post we've observed with
enough history to compute its 24h normalized engagement lift.
"""
from datetime import datetime

from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Niche(Base):
    __tablename__ = "niches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    label: Mapped[str] = mapped_column(String(128))
    color_hex: Mapped[str] = mapped_column(String(16))
    blurb: Mapped[str] = mapped_column(String(256), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    posts: Mapped[list["NichePost"]] = relationship(back_populates="niche")


class NichePost(Base):
    """
    A single post observation. `lift_24h` is the only engagement number we trust:
    views at the +24h checkpoint, divided by the creator's own 30-day baseline.
    """
    __tablename__ = "niche_posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    niche_id: Mapped[int] = mapped_column(ForeignKey("niches.id"), index=True)

    creator_handle: Mapped[str] = mapped_column(String(128))
    publish_hour_local: Mapped[int] = mapped_column(Integer, index=True)   # 0-23
    publish_day_type: Mapped[str] = mapped_column(String(8), index=True)   # "weekday"/"weekend"

    lift_24h: Mapped[float] = mapped_column(Float, index=True)
    views_at_24h: Mapped[int] = mapped_column(Integer, default=0)
    baseline_used: Mapped[float] = mapped_column(Float, default=0.0)

    posted_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    niche: Mapped["Niche"] = relationship(back_populates="posts")


Index("ix_niche_posts_bucket", NichePost.niche_id, NichePost.publish_hour_local, NichePost.publish_day_type)


class NicheHourlyStat(Base):
    """
    Precomputed percentile distribution per (niche, hour, day_type) bucket.
    Recomputed hourly by the background scheduler (or after simulate_batch()).
    """
    __tablename__ = "niche_hourly_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    niche_id: Mapped[int] = mapped_column(ForeignKey("niches.id"), index=True)
    hour: Mapped[int] = mapped_column(Integer)
    day_type: Mapped[str] = mapped_column(String(8))

    p10: Mapped[float] = mapped_column(Float)
    p25: Mapped[float] = mapped_column(Float)
    p50: Mapped[float] = mapped_column(Float)
    p75: Mapped[float] = mapped_column(Float)
    p90: Mapped[float] = mapped_column(Float)
    sample_size: Mapped[int] = mapped_column(Integer)

    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


Index(
    "ix_hourly_stats_bucket",
    NicheHourlyStat.niche_id,
    NicheHourlyStat.hour,
    NicheHourlyStat.day_type,
    unique=True,
)
