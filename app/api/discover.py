"""
Discovery REST endpoints.

  POST   /api/users/{user_id}/discover/run                — kick off a discovery run
  GET    /api/users/{user_id}/discover/runs               — list recent runs
  GET    /api/users/{user_id}/discover/candidates         — list candidates with filters
  POST   /api/users/{user_id}/discover/candidates/{cid}/approve  — promote to TrackedCreator
  POST   /api/users/{user_id}/discover/candidates/{cid}/reject   — mark rejected
  GET    /api/users/{user_id}/discover/settings           — read ICP + seeds
  PUT    /api/users/{user_id}/discover/settings           — update ICP + seeds
"""
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session, get_db
from app.models.discovery_models import (
    CandidateStatus,
    CreatorCandidate,
    DiscoveryRun,
    DiscoverySettings,
)
from app.models.models import User
from app.services.discovery.runner import (
    get_or_create_settings,
    promote_candidate,
    run_discovery,
)

router = APIRouter(prefix="/users/{user_id}/discover", tags=["discover"])


# ─── Schemas ────────────────────────────────────────────────────────────────


class RunRequest(BaseModel):
    use_scrapers: bool = True
    per_source_limit: int | None = Field(default=None, ge=1, le=100)
    run_sync: bool = Field(default=False, description="Block until run completes")


class RunResponse(BaseModel):
    id: int
    status: str
    sources_used: list[str] | None
    raw_count: int
    deduped_count: int
    hydrated_count: int
    scored_count: int
    started_at: datetime
    completed_at: datetime | None
    error_message: str | None = None

    model_config = {"from_attributes": True}


class CandidateResponse(BaseModel):
    id: int
    handle: str
    display_name: str | None
    biography: str | None
    follower_count: int | None
    engagement_rate: float | None
    avg_views: float | None
    last_post_at: datetime | None

    # Club Stanley signals — surfaced flat for easy table rendering.
    posts_per_week: float | None
    like_to_comment_ratio: float | None
    ad_density: float | None
    country_guess: str | None
    timezone_bucket: str | None
    talking_head_signal: int | None
    bio_quality_signal: int | None
    comment_quality_signal: int | None
    is_outlier_flagged: bool
    green_flags: list[str] | None
    red_flags: list[str] | None

    discovered_via: str
    discovery_seed: str | None
    score_fit: int | None
    score_engagement: int | None
    score_audience: int | None
    score_recency: int | None
    score_overall: int | None
    score_reasoning: str | None
    status: str
    # Club Stanley cohort shortlist
    is_shortlisted: bool = False
    shortlisted_at: datetime | None = None
    first_seen_at: datetime

    model_config = {"from_attributes": True}


class SettingsResponse(BaseModel):
    icp_description: str
    hashtag_seeds: list[str] | None
    brand_account_seeds: list[str] | None
    competitor_handle_seeds: list[str] | None
    follower_min: int
    follower_max: int
    min_engagement_rate: float
    allow_sub_floor_outliers: bool
    preferred_geo_tags: list[str] | None
    deprioritized_geo_tags: list[str] | None
    candidates_per_source: int
    digest_size: int

    model_config = {"from_attributes": True}


class SettingsUpdate(BaseModel):
    icp_description: str | None = None
    hashtag_seeds: list[str] | None = None
    brand_account_seeds: list[str] | None = None
    competitor_handle_seeds: list[str] | None = None
    follower_min: int | None = Field(default=None, ge=0)
    follower_max: int | None = Field(default=None, ge=0)
    min_engagement_rate: float | None = Field(default=None, ge=0, le=1)
    allow_sub_floor_outliers: bool | None = None
    preferred_geo_tags: list[str] | None = None
    deprioritized_geo_tags: list[str] | None = None
    candidates_per_source: int | None = Field(default=None, ge=1, le=100)
    digest_size: int | None = Field(default=None, ge=1, le=100)


# ─── Helpers ────────────────────────────────────────────────────────────────


async def _require_user(db: AsyncSession, user_id: int) -> User:
    res = await db.execute(select(User).where(User.id == user_id))
    user = res.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")
    return user


async def _background_run(user_id: int, use_scrapers: bool, per_source_limit: int | None):
    # Background tasks need their own session (the request-scoped one is closed).
    async with async_session() as db:
        try:
            await run_discovery(
                db,
                user_id=user_id,
                use_scrapers=use_scrapers,
                per_source_limit=per_source_limit,
            )
        except Exception:
            # `run_discovery` already records the failure on the DiscoveryRun row.
            pass


# ─── Routes ─────────────────────────────────────────────────────────────────


@router.post("/run", response_model=RunResponse, status_code=202)
async def kickoff_run(
    user_id: int,
    payload: RunRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    await _require_user(db, user_id)

    if payload.run_sync:
        run = await run_discovery(
            db,
            user_id=user_id,
            use_scrapers=payload.use_scrapers,
            per_source_limit=payload.per_source_limit,
        )
        return run

    # Async path: insert a placeholder run so the client gets an id immediately.
    placeholder = DiscoveryRun(user_id=user_id, sources_used=[])
    db.add(placeholder)
    await db.commit()
    await db.refresh(placeholder)

    background_tasks.add_task(
        _background_run, user_id, payload.use_scrapers, payload.per_source_limit
    )
    return placeholder


@router.get("/runs", response_model=list[RunResponse])
async def list_runs(
    user_id: int,
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    await _require_user(db, user_id)
    res = await db.execute(
        select(DiscoveryRun)
        .where(DiscoveryRun.user_id == user_id)
        .order_by(desc(DiscoveryRun.started_at))
        .limit(limit)
    )
    return res.scalars().all()


@router.get("/candidates", response_model=list[CandidateResponse])
async def list_candidates(
    user_id: int,
    status: Literal["pending", "approved", "rejected", "all"] = "pending",
    min_score: int = Query(default=0, ge=0, le=100),
    # Cap raised to 2000 so the Sourcing metrics dashboard can pull the full
    # lifetime candidate set to compute funnel + score distribution.
    limit: int = Query(default=50, ge=1, le=2000),
    shortlisted: bool | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    await _require_user(db, user_id)

    filters = [CreatorCandidate.user_id == user_id, CreatorCandidate.score_overall >= min_score]
    if status != "all":
        filters.append(CreatorCandidate.status == CandidateStatus(status))
    if shortlisted is not None:
        filters.append(CreatorCandidate.is_shortlisted == shortlisted)

    res = await db.execute(
        select(CreatorCandidate)
        .where(and_(*filters))
        .order_by(desc(CreatorCandidate.score_overall), desc(CreatorCandidate.first_seen_at))
        .limit(limit)
    )
    return res.scalars().all()


# ─── Club Stanley shortlist ────────────────────────────────────────────────


@router.post("/candidates/{candidate_id}/shortlist", response_model=CandidateResponse)
async def shortlist(
    user_id: int,
    candidate_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Pick a candidate for the Club Stanley cohort (toggleable)."""
    await _require_user(db, user_id)
    res = await db.execute(
        select(CreatorCandidate).where(
            and_(
                CreatorCandidate.id == candidate_id,
                CreatorCandidate.user_id == user_id,
            )
        )
    )
    candidate = res.scalar_one_or_none()
    if not candidate:
        raise HTTPException(404, "Candidate not found")
    candidate.is_shortlisted = True
    candidate.shortlisted_at = datetime.utcnow()
    await db.commit()
    await db.refresh(candidate)
    return candidate


@router.delete("/candidates/{candidate_id}/shortlist", response_model=CandidateResponse)
async def unshortlist(
    user_id: int,
    candidate_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Remove a candidate from the Club Stanley cohort."""
    await _require_user(db, user_id)
    res = await db.execute(
        select(CreatorCandidate).where(
            and_(
                CreatorCandidate.id == candidate_id,
                CreatorCandidate.user_id == user_id,
            )
        )
    )
    candidate = res.scalar_one_or_none()
    if not candidate:
        raise HTTPException(404, "Candidate not found")
    candidate.is_shortlisted = False
    candidate.shortlisted_at = None
    await db.commit()
    await db.refresh(candidate)
    return candidate


@router.post("/candidates/{candidate_id}/approve", response_model=CandidateResponse)
async def approve(
    user_id: int,
    candidate_id: int,
    db: AsyncSession = Depends(get_db),
):
    await _require_user(db, user_id)
    try:
        await promote_candidate(db, user_id=user_id, candidate_id=candidate_id)
    except ValueError as e:
        raise HTTPException(404, str(e))

    res = await db.execute(
        select(CreatorCandidate).where(CreatorCandidate.id == candidate_id)
    )
    return res.scalar_one()


@router.post("/candidates/{candidate_id}/reject", response_model=CandidateResponse)
async def reject(
    user_id: int,
    candidate_id: int,
    db: AsyncSession = Depends(get_db),
):
    await _require_user(db, user_id)
    res = await db.execute(
        select(CreatorCandidate).where(
            and_(
                CreatorCandidate.id == candidate_id,
                CreatorCandidate.user_id == user_id,
            )
        )
    )
    candidate = res.scalar_one_or_none()
    if not candidate:
        raise HTTPException(404, "Candidate not found")
    candidate.status = CandidateStatus.REJECTED
    candidate.reviewed_at = datetime.utcnow()
    await db.commit()
    await db.refresh(candidate)
    return candidate


@router.get("/settings", response_model=SettingsResponse)
async def get_settings(
    user_id: int, db: AsyncSession = Depends(get_db)
):
    await _require_user(db, user_id)
    return await get_or_create_settings(db, user_id)


@router.put("/settings", response_model=SettingsResponse)
async def update_settings(
    user_id: int,
    payload: SettingsUpdate,
    db: AsyncSession = Depends(get_db),
):
    await _require_user(db, user_id)
    s = await get_or_create_settings(db, user_id)
    data = payload.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(s, k, v)
    await db.commit()
    await db.refresh(s)
    return s
