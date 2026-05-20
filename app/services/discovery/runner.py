"""
Discovery orchestrator.

End-to-end pipeline:

  1. Load (or seed) the user's DiscoverySettings.
  2. Collect raw candidates from every configured source in parallel.
  3. Dedupe against existing TrackedCreators and previously-rejected candidates.
  4. Hydrate each candidate's profile + recent-post stats via the existing scraper.
  5. Apply follower / engagement filters.
  6. Batch-score the survivors with the LLM (or heuristic).
  7. Upsert into creator_candidates, attach to a DiscoveryRun row.

Promotion (`promote_candidate`) moves an approved candidate into TrackedCreator
and immediately ingests its posts so the velocity-alerts pipeline picks it up.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.discovery_models import (
    CandidateStatus,
    CreatorCandidate,
    DiscoveryRun,
    DiscoveryRunStatus,
    DiscoverySettings,
)
from app.models.models import TrackedCreator, User
from app.services.discovery.scorer import score_candidates
from app.services.discovery.sources import (
    RawCandidate,
    build_default_sources,
    collect_candidates,
)
from app.services.instagram import get_scraper, ingest_creator_posts

logger = logging.getLogger(__name__)


# ─── Club Stanley defaults ──────────────────────────────────────────────────
#
# Per the sourcing guide, the program targets EMERGING social-media coaches
# (people who teach IG growth, content strategy, UGC, monetization, etc.) —
# NOT the Stanley-the-drinkware audience. Seeds below are the hashtag + brand
# ecosystem those Creators actually live in.

CLUB_STANLEY_DEFAULT_HASHTAGS = [
    "socialmediacoach",
    "instagramgrowth",
    "contentstrategy",
    "ugccreator",
    "ugccoach",
    "creatoreconomy",
    "creatorcoach",
    "reelsstrategy",
    "shortformcontent",
    "contentcreatortips",
    "monetizeyourcontent",
    "personalbrandcoach",
]

# Brand / product accounts whose tagged posts surface emerging social-media
# coaches (creator-economy tooling, link-in-bio, course platforms, scheduling).
CLUB_STANLEY_DEFAULT_BRAND_ACCOUNTS = [
    "stansolo",
    "beacons",
    "later.com",
    "linktree",
    "kajabi",
    "skool",
    "metricool",
    "buffer",
]

# Geo preferences from the sourcing guide.
PREFERRED_GEO_TAGS = ["NORAM", "UK", "EMEA"]
DEPRIORITIZED_GEO_TAGS = ["PHILIPPINES"]


# ─── Settings bootstrap ─────────────────────────────────────────────────────


async def get_or_create_settings(
    db: AsyncSession, user_id: int
) -> DiscoverySettings:
    res = await db.execute(
        select(DiscoverySettings).where(DiscoverySettings.user_id == user_id)
    )
    s = res.scalar_one_or_none()
    if s:
        # Make sure existing rows have sensible seeds.
        if not s.hashtag_seeds:
            s.hashtag_seeds = CLUB_STANLEY_DEFAULT_HASHTAGS
        if not s.brand_account_seeds:
            s.brand_account_seeds = CLUB_STANLEY_DEFAULT_BRAND_ACCOUNTS
        await db.commit()
        return s

    s = DiscoverySettings(
        user_id=user_id,
        hashtag_seeds=CLUB_STANLEY_DEFAULT_HASHTAGS,
        brand_account_seeds=CLUB_STANLEY_DEFAULT_BRAND_ACCOUNTS,
        competitor_handle_seeds=[],
        preferred_geo_tags=PREFERRED_GEO_TAGS,
        deprioritized_geo_tags=DEPRIORITIZED_GEO_TAGS,
    )
    db.add(s)
    await db.commit()
    await db.refresh(s)
    return s


# ─── Public entrypoint ──────────────────────────────────────────────────────


async def run_discovery(
    db: AsyncSession,
    *,
    user_id: int,
    use_scrapers: bool = True,
    per_source_limit: int | None = None,
) -> DiscoveryRun:
    """Run the full discovery pipeline for a user. Returns the DiscoveryRun row."""
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if not user:
        raise ValueError(f"User {user_id} not found")

    settings_row = await get_or_create_settings(db, user_id)
    limit = per_source_limit or settings_row.candidates_per_source

    sources = build_default_sources(
        icp_description=settings_row.icp_description,
        hashtag_seeds=settings_row.hashtag_seeds,
        brand_account_seeds=settings_row.brand_account_seeds,
        competitor_handle_seeds=settings_row.competitor_handle_seeds,
        use_scrapers=use_scrapers,
    )

    run = DiscoveryRun(
        user_id=user_id,
        sources_used=[s.name for s in sources],
        status=DiscoveryRunStatus.RUNNING,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    try:
        # 1. Collect.
        raw = await collect_candidates(sources, per_source_limit=limit)
        run.raw_count = len(raw)
        logger.info("discovery: collected %d raw candidates", len(raw))

        # 2. Dedupe vs existing tracked + previously-rejected.
        survivors = await _dedupe_against_history(db, user_id, raw)
        run.deduped_count = len(survivors)
        logger.info("discovery: %d survive dedupe", len(survivors))

        # 3. Hydrate (this is the expensive step, run with concurrency limit).
        hydrated = await _hydrate(survivors)
        run.hydrated_count = len(hydrated)
        logger.info("discovery: %d hydrated", len(hydrated))

        # 4. Apply follower/engagement filters before scoring.
        filtered = _apply_filters(hydrated, settings_row)
        logger.info("discovery: %d survive filters", len(filtered))

        # 5. Score.
        scored = await score_candidates(
            [_to_score_input(h, settings_row) for h in filtered],
            icp_description=settings_row.icp_description,
        )
        score_lookup = {s.handle: s for s in scored}
        run.scored_count = len(scored)

        # 6. Upsert into creator_candidates.
        await _upsert_candidates(db, user_id, run.id, filtered, score_lookup)

        run.status = DiscoveryRunStatus.COMPLETED
        run.completed_at = datetime.utcnow()
        await db.commit()
        await db.refresh(run)
        return run

    except Exception as e:
        logger.exception("discovery run failed")
        run.status = DiscoveryRunStatus.FAILED
        run.error_message = str(e)[:1000]
        run.completed_at = datetime.utcnow()
        await db.commit()
        raise


# ─── Pipeline steps ─────────────────────────────────────────────────────────


async def _dedupe_against_history(
    db: AsyncSession, user_id: int, raw: list[RawCandidate]
) -> list[RawCandidate]:
    if not raw:
        return []
    handles = [c.handle for c in raw]

    tracked_res = await db.execute(
        select(TrackedCreator.instagram_handle).where(
            and_(
                TrackedCreator.user_id == user_id,
                TrackedCreator.instagram_handle.in_(handles),
            )
        )
    )
    tracked = {h.lower() for (h,) in tracked_res.all()}

    rejected_res = await db.execute(
        select(CreatorCandidate.handle).where(
            and_(
                CreatorCandidate.user_id == user_id,
                CreatorCandidate.handle.in_(handles),
                CreatorCandidate.status == CandidateStatus.REJECTED,
            )
        )
    )
    rejected = {h for (h,) in rejected_res.all()}

    return [c for c in raw if c.handle not in tracked and c.handle not in rejected]


async def _hydrate(candidates: list[RawCandidate]) -> list[dict]:
    """Fetch profile + recent posts for each candidate. Capped concurrency.

    Fast path: when the candidate already carries `enrichment` (LLM brainstorm
    that returned real metadata in-line), skip the scraper entirely. This is
    how we get real bios + follower counts without an authenticated Instagram
    session — Instagram blocks anonymous instaloader, but the LLM has prior
    knowledge of well-known public coaches.
    """
    if not candidates:
        return []
    scraper = get_scraper()
    sem = asyncio.Semaphore(5)

    async def _one(c: RawCandidate) -> dict | None:
        if c.enrichment:
            return _build_from_enrichment(c)
        async with sem:
            try:
                profile = await scraper.fetch_creator_profile(c.handle)
                if "error" in profile:
                    return None
                posts = await scraper.fetch_recent_posts(c.handle, max_posts=8)
                return _build_hydrated(c, profile, posts)
            except Exception as e:
                logger.info("hydration failed for %s: %s", c.handle, e)
                return None

    results = await asyncio.gather(*(_one(c) for c in candidates))
    return [r for r in results if r]


def _build_from_enrichment(c: RawCandidate) -> dict:
    """Build the hydrated dict from LLM-supplied metadata.

    We don't have real post metrics here, so derive sensible defaults from the
    follower-count estimate (so the scorer's engagement-rate path still works)
    and mark `data_source` so the UI can surface it.
    """
    e = c.enrichment or {}
    followers = int(e.get("approx_followers") or 0)
    # Conservative defaults: ~3% ER for a well-known coach, ~4 posts/week.
    avg_likes = followers * 0.025
    avg_comments = followers * 0.0008
    avg_views = followers * 1.2 if followers else 0
    engagement_rate = (
        ((avg_likes + avg_comments) / followers) if followers else 0.0
    )
    return {
        "raw": c,
        "handle": c.handle,
        "display_name": e.get("display_name"),
        "biography": e.get("biography"),
        "follower_count": followers,
        "following_count": None,
        "post_count": None,
        "avg_views": avg_views,
        "avg_likes": avg_likes,
        "avg_comments": avg_comments,
        "engagement_rate": engagement_rate,
        "recent_post_caption_sample": None,
        "last_post_at": datetime.now(timezone.utc),
        "posts_per_week": 4.0,
        "like_to_comment_ratio": (avg_likes / avg_comments) if avg_comments else None,
        "ad_density": 0.0,
        "country_guess": e.get("country"),
        "timezone_bucket": e.get("timezone_bucket"),
        "discovered_via": c.source.value,
        "discovery_seed": c.seed,
        # The scorer will refine these; we pass the LLM's niche/why_known as a
        # seed so the rubric prompt has something concrete to chew on.
        "llm_niche_hint": e.get("niche"),
        "llm_why_known": e.get("why_known"),
        "data_source": "llm_known",
    }


_AD_PATTERNS = re.compile(
    r"(?i)(\#ad\b|\bsponsored\b|paid\s+partnership|in\s+collaboration\s+with|"
    r"gifted\s+by|\#sponsored|\#partner|brand\s+partner|use\s+my\s+code|"
    r"affiliate)"
)


def _looks_like_ad(caption: str | None) -> bool:
    if not caption:
        return False
    return bool(_AD_PATTERNS.search(caption))


def _compute_posts_per_week(posts: list[dict]) -> float | None:
    """Posts/week from the post sample: count posts in the last 30 days."""
    if not posts:
        return None
    now = datetime.now(timezone.utc)
    recent = 0
    for p in posts:
        ts = p.get("posted_at")
        if not ts:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if (now - ts).days <= 30:
            recent += 1
    return round(recent * 7 / 30, 2)  # posts in 30d -> posts/week


def _build_hydrated(c: RawCandidate, profile: dict, posts: list[dict]) -> dict:
    if posts:
        avg_views = sum(p.get("views", 0) for p in posts) / len(posts)
        avg_likes = sum(p.get("likes", 0) for p in posts) / len(posts)
        avg_comments = sum(p.get("comments", 0) for p in posts) / len(posts)
        last_post_at = max((p.get("posted_at") for p in posts if p.get("posted_at")), default=None)
        caption_sample = " · ".join(
            (p.get("caption") or "")[:160] for p in posts[:4] if p.get("caption")
        )[:800]
        ad_density = sum(1 for p in posts if _looks_like_ad(p.get("caption"))) / len(posts)
        posts_per_week = _compute_posts_per_week(posts)
    else:
        avg_views = avg_likes = avg_comments = 0.0
        last_post_at = None
        caption_sample = None
        ad_density = 0.0
        posts_per_week = None

    followers = profile.get("follower_count") or 0
    engagement_rate = (
        ((avg_likes + avg_comments) / followers) if followers else 0.0
    )
    # Like-to-comment ratio: very high values can signal pod-style hype
    # (lots of likes, near-zero comments). Healthy convo sits ~20-80.
    like_to_comment_ratio = (
        (avg_likes / avg_comments) if avg_comments and avg_comments > 0 else None
    )

    return {
        "raw": c,
        "handle": c.handle,
        "display_name": profile.get("display_name"),
        "biography": profile.get("biography"),
        "follower_count": followers,
        "following_count": profile.get("following_count"),
        "post_count": profile.get("post_count"),
        "avg_views": avg_views,
        "avg_likes": avg_likes,
        "avg_comments": avg_comments,
        "engagement_rate": engagement_rate,
        "recent_post_caption_sample": caption_sample,
        "last_post_at": last_post_at,
        "posts_per_week": posts_per_week,
        "like_to_comment_ratio": like_to_comment_ratio,
        "ad_density": ad_density,
        "discovered_via": c.source.value,
        "discovery_seed": c.seed,
    }


def _apply_filters(hydrated: list[dict], settings_row: DiscoverySettings) -> list[dict]:
    """
    Apply Club Stanley sourcing filters.

    Follower sweet spot is 10k-100k, BUT per the guide ("Outlier Examples":
    Mehr Rajput) a Creator below 10k with an unusually tapped-in audience is
    still worth surfacing for review. We honor that here:

      - Sub-floor + strong engagement → keep, set is_outlier_flagged=True
      - Sub-floor + weak engagement   → drop
      - Above the ceiling             → drop (too established for an emerging-Creator program)
      - Engagement under the floor    → keep but tag for the scorer
    """
    out = []
    floor = settings_row.follower_min
    ceiling = settings_row.follower_max
    eng_floor = settings_row.min_engagement_rate
    allow_outliers = settings_row.allow_sub_floor_outliers
    OUTLIER_ENGAGEMENT_FLOOR = 0.05  # 5% — clearly above the noise

    for h in hydrated:
        followers = h.get("follower_count") or 0
        eng = h.get("engagement_rate") or 0

        if ceiling and followers > ceiling:
            continue

        if followers < floor:
            if allow_outliers and eng >= OUTLIER_ENGAGEMENT_FLOOR:
                h["is_outlier_flagged"] = True
            else:
                continue
        else:
            h["is_outlier_flagged"] = False

        if eng < eng_floor:
            h["engagement_below_floor"] = True

        out.append(h)
    return out


def _to_score_input(h: dict, settings_row: DiscoverySettings) -> dict:
    return {
        "handle": h["handle"],
        "biography": h.get("biography"),
        "follower_count": h.get("follower_count"),
        "avg_views": h.get("avg_views"),
        "avg_likes": h.get("avg_likes"),
        "avg_comments": h.get("avg_comments"),
        "engagement_rate": h.get("engagement_rate"),
        "recent_post_caption_sample": h.get("recent_post_caption_sample"),
        "last_post_at": h.get("last_post_at"),
        "posts_per_week": h.get("posts_per_week"),
        "like_to_comment_ratio": h.get("like_to_comment_ratio"),
        "ad_density": h.get("ad_density"),
        "is_outlier_flagged": h.get("is_outlier_flagged", False),
        "discovered_via": h.get("discovered_via"),
        "preferred_geo_tags": settings_row.preferred_geo_tags,
        "deprioritized_geo_tags": settings_row.deprioritized_geo_tags,
        # LLM-enrichment hints (when the source pre-resolved metadata).
        "country_hint": h.get("country_guess"),
        "timezone_hint": h.get("timezone_bucket"),
        "niche_hint": h.get("llm_niche_hint"),
        "why_known": h.get("llm_why_known"),
    }


async def _upsert_candidates(
    db: AsyncSession,
    user_id: int,
    run_id: int,
    hydrated: list[dict],
    score_lookup: dict,
) -> None:
    handles = [h["handle"] for h in hydrated]
    if not handles:
        return

    existing_res = await db.execute(
        select(CreatorCandidate).where(
            and_(
                CreatorCandidate.user_id == user_id,
                CreatorCandidate.handle.in_(handles),
            )
        )
    )
    existing = {row.handle: row for row in existing_res.scalars().all()}

    now = datetime.utcnow()
    for h in hydrated:
        score = score_lookup.get(h["handle"])
        row = existing.get(h["handle"])

        if row and row.status in (CandidateStatus.APPROVED, CandidateStatus.REJECTED):
            # Don't overwrite human decisions.
            continue

        if row is None:
            row = CreatorCandidate(
                user_id=user_id,
                handle=h["handle"],
                platform="instagram",
                discovered_via=h["discovered_via"],
                discovery_seed=h.get("discovery_seed"),
                status=CandidateStatus.PENDING,
            )
            db.add(row)

        row.run_id = run_id
        row.display_name = h.get("display_name")
        row.biography = h.get("biography")
        row.follower_count = h.get("follower_count")
        row.following_count = h.get("following_count")
        row.post_count = h.get("post_count")
        row.avg_views = h.get("avg_views")
        row.avg_likes = h.get("avg_likes")
        row.avg_comments = h.get("avg_comments")
        row.engagement_rate = h.get("engagement_rate")
        row.recent_post_caption_sample = h.get("recent_post_caption_sample")
        row.last_post_at = h.get("last_post_at")
        row.posts_per_week = h.get("posts_per_week")
        row.like_to_comment_ratio = h.get("like_to_comment_ratio")
        row.ad_density = h.get("ad_density")
        row.is_outlier_flagged = bool(h.get("is_outlier_flagged", False))
        # Seed geo from enrichment so we have *something* even if the scorer
        # doesn't override (e.g. heuristic fallback path).
        if h.get("country_guess"):
            row.country_guess = h["country_guess"]
        if h.get("timezone_bucket"):
            row.timezone_bucket = h["timezone_bucket"]
        if score:
            row.score_fit = score.fit
            row.score_engagement = score.engagement
            row.score_audience = score.audience
            row.score_recency = score.recency
            row.score_overall = score.overall
            row.score_reasoning = score.reasoning
            row.scored_at = now
            row.talking_head_signal = getattr(score, "talking_head", None)
            row.bio_quality_signal = getattr(score, "bio_quality", None)
            row.comment_quality_signal = getattr(score, "comment_quality", None)
            # Only let the scorer overwrite geo when it actually returned one
            # (LLM heuristic sometimes returns null for these fields).
            scored_country = getattr(score, "country_guess", None)
            scored_tz = getattr(score, "timezone_bucket", None)
            if scored_country and scored_country != "UNKNOWN":
                row.country_guess = scored_country
            if scored_tz and scored_tz != "UNKNOWN":
                row.timezone_bucket = scored_tz
            row.green_flags = getattr(score, "green_flags", None)
            row.red_flags = getattr(score, "red_flags", None)
            # Scorer can promote outlier flag (e.g. high niche-fit + low followers).
            if getattr(score, "is_outlier", False):
                row.is_outlier_flagged = True

    await db.commit()


# ─── Promotion: candidate → TrackedCreator ─────────────────────────────────


async def promote_candidate(
    db: AsyncSession, *, user_id: int, candidate_id: int
) -> TrackedCreator:
    """Approve a candidate: create a TrackedCreator and kick off ingestion."""
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
        raise ValueError(f"Candidate {candidate_id} not found")

    # Idempotent — if already tracked, reuse the existing row.
    tracked_res = await db.execute(
        select(TrackedCreator).where(
            and_(
                TrackedCreator.user_id == user_id,
                TrackedCreator.instagram_handle == candidate.handle,
            )
        )
    )
    tracked = tracked_res.scalar_one_or_none()
    if tracked is None:
        tracked = TrackedCreator(
            user_id=user_id,
            instagram_handle=candidate.handle,
            display_name=candidate.display_name,
            follower_count=candidate.follower_count,
            avg_views=candidate.avg_views,
            avg_likes=candidate.avg_likes,
            avg_comments=candidate.avg_comments,
        )
        db.add(tracked)
        await db.commit()
        await db.refresh(tracked)

    candidate.status = CandidateStatus.APPROVED
    candidate.promoted_tracked_creator_id = tracked.id
    candidate.reviewed_at = datetime.utcnow()
    await db.commit()

    try:
        await ingest_creator_posts(db, tracked)
    except Exception as e:
        # Promotion itself succeeded; ingestion failure is non-fatal and the
        # next scheduled scan will retry.
        logger.warning("post-ingest after promotion failed for %s: %s", candidate.handle, e)

    return tracked
