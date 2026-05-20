"""
Discovery models — automated creator sourcing pipeline for Club Stanley.

Kept distinct from the velocity-alerts schema (TrackedCreator / CreatorPost) so
discovery can produce many low-confidence candidates without polluting the
tracked-creator graph. A candidate gets promoted into TrackedCreator only when
the user explicitly approves it.

Flow:
  DiscoverySettings (per user)
       │
       ▼
  DiscoveryRun  ──┐
       │         │ (many candidates per run, one row per (run, handle))
       ▼         ▼
  CreatorCandidate
       │
       │ approve() → creates a TrackedCreator row, marks candidate PROMOTED
       │ reject()  → marks candidate REJECTED, never surfaced again
       ▼
  (existing velocity-alerts pipeline takes over)
"""
import enum
from datetime import datetime

from sqlalchemy import (
    String,
    Integer,
    Float,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    Enum as SAEnum,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class CandidateStatus(str, enum.Enum):
    PENDING = "pending"       # discovered, not yet reviewed
    APPROVED = "approved"     # user approved, promoted into TrackedCreator
    REJECTED = "rejected"     # user rejected, hide from future runs
    DUPLICATE = "duplicate"   # already in TrackedCreator at discovery time
    ERRORED = "errored"       # hydration or scoring failed


class DiscoveryRunStatus(str, enum.Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CandidateSource(str, enum.Enum):
    HASHTAG = "hashtag"
    BRAND_MENTION = "brand_mention"
    LLM_BRAINSTORM = "llm_brainstorm"
    SIMILAR_ACCOUNT = "similar_account"
    MOCK = "mock"


class DiscoverySettings(Base):
    """Per-user ICP + discovery seeds. One row per user."""
    __tablename__ = "discovery_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), unique=True, index=True
    )

    # ICP — free-form description fed to the LLM scorer.
    # Mirrors the Club Stanley sourcing guide: emerging social-media coaches,
    # 10k-100k followers, NORAM / EMEA preferred, talking-head content, 3x+/week
    # cadence, tapped-in audience, brand-ready but not over-saturated.
    icp_description: Mapped[str] = mapped_column(
        Text,
        default=(
            "Club Stanley target Creators: EMERGING social-media coaches on "
            "Instagram (people who teach content strategy, IG growth, UGC, "
            "creator-economy tactics, monetization, hooks/storytelling, etc.). "
            "Sweet spot 10k-100k followers; sub-10k OK only as an outlier when "
            "the audience is unusually tapped-in. Prefer talking-head / "
            "voiceover content with a clear POV over generic 'growth reels' "
            "(b-roll + text overlay). Want consistent posting (3x+/week), "
            "real comment conversation (questions, 'I tried this', Creator "
            "replies), and a bio that clearly states niche + who they help + "
            "proof points. Geo preference: NORAM and UK-adjacent EMEA. Avoid "
            "ad-saturated profiles."
        ),
    )

    # Discovery seeds — JSON arrays the user can edit.
    hashtag_seeds: Mapped[list | None] = mapped_column(JSON)
    brand_account_seeds: Mapped[list | None] = mapped_column(JSON)
    competitor_handle_seeds: Mapped[list | None] = mapped_column(JSON)

    # Filters applied before scoring. Per the sourcing guide:
    #   - 10k-100k is the sweet spot.
    #   - Sub-10k allowed when "exceptional potential" — handled via
    #     allow_sub_floor_outliers below rather than hard-dropping.
    follower_min: Mapped[int] = mapped_column(Integer, default=10_000)
    follower_max: Mapped[int] = mapped_column(Integer, default=100_000)
    min_engagement_rate: Mapped[float] = mapped_column(Float, default=0.02)  # 2%
    allow_sub_floor_outliers: Mapped[bool] = mapped_column(Boolean, default=True)

    # Geo preferences fed to the scorer.
    preferred_geo_tags: Mapped[list | None] = mapped_column(JSON)   # ["NORAM", "UK", "EMEA"]
    deprioritized_geo_tags: Mapped[list | None] = mapped_column(JSON)  # ["Philippines"]

    # Run sizing.
    candidates_per_source: Mapped[int] = mapped_column(Integer, default=20)
    digest_size: Mapped[int] = mapped_column(Integer, default=15)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class DiscoveryRun(Base):
    """One execution of the discovery pipeline."""
    __tablename__ = "discovery_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)

    status: Mapped[str] = mapped_column(
        SAEnum(DiscoveryRunStatus), default=DiscoveryRunStatus.RUNNING
    )

    # Telemetry — how many candidates each source returned + how many survived.
    sources_used: Mapped[list | None] = mapped_column(JSON)
    raw_count: Mapped[int] = mapped_column(Integer, default=0)
    deduped_count: Mapped[int] = mapped_column(Integer, default=0)
    hydrated_count: Mapped[int] = mapped_column(Integer, default=0)
    scored_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)

    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)

    candidates: Mapped[list["CreatorCandidate"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class CreatorCandidate(Base):
    """A discovered handle awaiting review."""
    __tablename__ = "creator_candidates"
    __table_args__ = (
        # A user shouldn't see the same handle twice across runs — we upsert
        # into the latest run rather than re-creating.
        UniqueConstraint("user_id", "platform", "handle", name="uq_user_platform_handle"),
        Index("ix_candidate_user_status_score", "user_id", "status", "score_overall"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    run_id: Mapped[int | None] = mapped_column(
        ForeignKey("discovery_runs.id"), index=True
    )

    platform: Mapped[str] = mapped_column(String(32), default="instagram")
    handle: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str | None] = mapped_column(String(255))
    biography: Mapped[str | None] = mapped_column(Text)
    follower_count: Mapped[int | None] = mapped_column(Integer)
    following_count: Mapped[int | None] = mapped_column(Integer)
    post_count: Mapped[int | None] = mapped_column(Integer)

    # Derived signals from the hydrated recent-post sample.
    avg_views: Mapped[float | None] = mapped_column(Float)
    avg_likes: Mapped[float | None] = mapped_column(Float)
    avg_comments: Mapped[float | None] = mapped_column(Float)
    engagement_rate: Mapped[float | None] = mapped_column(Float)
    recent_post_caption_sample: Mapped[str | None] = mapped_column(Text)
    last_post_at: Mapped[datetime | None] = mapped_column(DateTime)

    # ─── Club Stanley green/red flag signals ───────────────────────────────
    # Each is surfaced independently so a reviewer can sanity-check the
    # bundled score against the underlying observation.
    posts_per_week: Mapped[float | None] = mapped_column(Float)         # cadence; green ≥3, red ≤1
    like_to_comment_ratio: Mapped[float | None] = mapped_column(Float)  # pod / weak-convo signal when very high
    ad_density: Mapped[float | None] = mapped_column(Float)             # 0-1: fraction of captions that look like sponcon
    country_guess: Mapped[str | None] = mapped_column(String(128))
    timezone_bucket: Mapped[str | None] = mapped_column(String(64))     # NORAM / UK / EMEA / APAC / PHILIPPINES / UNKNOWN
    # LLM-judged signals (0-100 each).
    talking_head_signal: Mapped[int | None] = mapped_column(Integer)
    bio_quality_signal: Mapped[int | None] = mapped_column(Integer)
    comment_quality_signal: Mapped[int | None] = mapped_column(Integer)
    is_outlier_flagged: Mapped[bool] = mapped_column(Boolean, default=False)  # Mehr-Rajput case
    green_flags: Mapped[list | None] = mapped_column(JSON)
    red_flags: Mapped[list | None] = mapped_column(JSON)

    # Sourcing trail.
    discovered_via: Mapped[str] = mapped_column(SAEnum(CandidateSource))
    discovery_seed: Mapped[str | None] = mapped_column(String(255))  # e.g. the hashtag or brand handle

    # Scoring axes — mapped onto the Club Stanley rubric:
    #   score_fit        → niche fit (social-media coaching alignment)
    #   score_engagement → engagement quality + comment health
    #   score_audience   → audience size + demographic / geo fit
    #   score_recency    → consistency (posts/week + recency of last post)
    # The weighted overall lives in score_overall.
    score_fit: Mapped[int | None] = mapped_column(Integer)
    score_engagement: Mapped[int | None] = mapped_column(Integer)
    score_audience: Mapped[int | None] = mapped_column(Integer)
    score_recency: Mapped[int | None] = mapped_column(Integer)
    score_overall: Mapped[int | None] = mapped_column(Integer, index=True)
    score_reasoning: Mapped[str | None] = mapped_column(Text)
    scored_at: Mapped[datetime | None] = mapped_column(DateTime)

    status: Mapped[str] = mapped_column(
        SAEnum(CandidateStatus), default=CandidateStatus.PENDING, index=True
    )
    promoted_tracked_creator_id: Mapped[int | None] = mapped_column(
        ForeignKey("tracked_creators.id")
    )

    # ─── Club Stanley shortlist ────────────────────────────────────────────
    # Whether a reviewer has explicitly picked this creator for the
    # Club Stanley incubator cohort. Independent of approve/reject — the
    # user might keep tracking a creator without shortlisting them, or
    # shortlist someone they don't want in the velocity-alerts pipeline.
    is_shortlisted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    shortlisted_at: Mapped[datetime | None] = mapped_column(DateTime)

    first_seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime)

    run: Mapped["DiscoveryRun"] = relationship(back_populates="candidates")
