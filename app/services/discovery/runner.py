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
from app.services.discovery.curated import for_program as curated_for_program
from app.models.discovery_models import CandidateSource as SourceEnum
from app.services.instagram import ingest_creator_posts
from app.services.instagram_verify import IgProfile, verify_many

logger = logging.getLogger(__name__)


# ─── Club Stanley defaults ──────────────────────────────────────────────────
#
# Club Stanley is Stan's Creator INCUBATOR program — an 8-12 week structured
# cohort where ~50-60 EMERGING Creators get a flat $300/post to produce
# authentic Stanley-led content. The North Star is content volume (120+
# posts in Cohort 2), not direct conversion. The flywheel feeds Social,
# Paid, PR, Partnerships, Product, Community, and Referrals downstream.
#
# So sourcing targets are:
#   - Emerging-tier Creators (50K-500K, NORAM/EMEA preferred)
#   - Authentic, builds-in-public, has taste, comfortable on camera
#   - Operates in or adjacent to the Creator-economy / content-strategy
#     niche so Stanley fits naturally into their workflow on-screen
#   - Cohort 1 archetype: Elly Walton (~124K, breakout case study)

CLUB_STANLEY_ICP_DESCRIPTION = (
    "Club Stanley is Stan's Creator incubator. We're sourcing EMERGING-tier "
    "Creators (50K-500K followers, NORAM/EMEA strongly preferred) who can "
    "produce authentic, well-crafted Instagram content that showcases their "
    "real workflow with Stanley (an AI content thought-partner). Target "
    "archetypes: content-strategy / personal-brand / Creator-economy "
    "Creators with a clear POV, strong taste, comfort on camera, "
    "consistent 2-3x+/week cadence, and audiences that engage with "
    "process-style content (not just outcome posts). Cohort 1 anchor: "
    "Elly Walton (44k → 124k followers, 18.4M views on a single Reel). "
    "We pay $300/post + bonuses + performance multipliers — Creators "
    "should be excited to produce 2-3 posts during the 12-week cohort."
)

CLUB_STANLEY_DEFAULT_HASHTAGS = [
    "contentstrategy",
    "personalbrandcoach",
    "creatoreconomy",
    "instagramgrowth",
    "reelsstrategy",
    "shortformcontent",
    "creatortips",
    "buildinpublic",
    "ugccreator",
    "monetizeyourcontent",
    "contentframeworks",
    "creatorworkflow",
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


# ─── Stanley Ambassador defaults ────────────────────────────────────────────
#
# Different ICP from Club Stanley. Targets channel operators (teaching
# Creators) whose audience already wants a content thought-partner. Bigger
# follower window (5k-100k), allow >100k only when teaching + trust hold.

AMBASSADOR_ICP_DESCRIPTION = (
    "Stanley Ambassadors are CHANNEL OPERATORS — non-influencer Creators "
    "whose audience already actively wants what Stanley provides (a content "
    "thought-partner). The non-negotiable test: 'If Stanley disappeared "
    "tomorrow, would this Creator's audience still be searching for a tool "
    "like Stanley?' If yes → Ambassador. If no → not, no matter how popular. "
    "Sweet spot 50K-100K followers (max-leverage, still non-paid territory). "
    "Anyone over 100K on any platform is DISQUALIFIED. "
    "Audience signal beats Creator persona: comments must include 'stealing "
    "this', 'what's your process?', 'how did you come up with this?', "
    "'I tried this and it worked'. Creator regularly teaches frameworks, "
    "ideation systems, scripting, hooks, posting workflows. Bonus points "
    "for owned distribution beyond IG (newsletter, community, course, "
    "coaching cohort, Substack) — those are how Stanley embeds. "
    "DE-PRIORITIZE: general AI accounts, lifestyle Creators who "
    "occasionally talk content, motivation-first feeds, Creator-economy "
    "commentators with no execution focus, audiences that want "
    "inspiration > systems."
)

AMBASSADOR_DEFAULT_HASHTAGS = [
    "contentstrategist",
    "personalbrandcoach",
    "creatorcoach",
    "instagramcoach",
    "reelscoach",
    "contentmentor",
    "helpingcreators",
    "buildyourpersonalbrand",
    "growonintagram",
    "contentframeworks",
    "ideationsystems",
    "creatorworkflow",
]

AMBASSADOR_DEFAULT_BRAND_ACCOUNTS = [
    "stansolo",
    "beehiiv",
    "substackinc",
    "circle",
    "skool",
    "kajabi",
    "convertkit",
    "notion",
]


# ─── Settings bootstrap ─────────────────────────────────────────────────────


async def get_or_create_settings(
    db: AsyncSession, user_id: int, *, program: str = "club_stanley"
) -> DiscoverySettings:
    res = await db.execute(
        select(DiscoverySettings).where(DiscoverySettings.user_id == user_id)
    )
    s = res.scalar_one_or_none()
    if s:
        # Make sure existing rows have sensible seeds for their program.
        if s.program == "ambassador":
            if not s.hashtag_seeds:
                s.hashtag_seeds = AMBASSADOR_DEFAULT_HASHTAGS
            if not s.brand_account_seeds:
                s.brand_account_seeds = AMBASSADOR_DEFAULT_BRAND_ACCOUNTS
            # Ambassadors: hard cap at 100K per the qualification doc
            # ("Over 100K on any platform → 0 / disqualified"). Floor at 50K
            # is the user's "minimum 50k" product call.
            s.follower_min = 50_000
            s.follower_max = 100_000
        else:
            if not s.hashtag_seeds:
                s.hashtag_seeds = CLUB_STANLEY_DEFAULT_HASHTAGS
            if not s.brand_account_seeds:
                s.brand_account_seeds = CLUB_STANLEY_DEFAULT_BRAND_ACCOUNTS
            # Club Stanley targets the EMERGING tier — bigger window than
            # Ambassadors. 50K floor (user's "minimum 50k"), 500K ceiling
            # (Cohort 1 breakout Elly Walton ended at ~124K, leaving room
            # for established-but-not-mega Creators).
            if (s.follower_min or 0) < 50_000:
                s.follower_min = 50_000
            if (s.follower_max or 0) < 500_000:
                s.follower_max = 500_000
        # Hard floor: NO sub-floor outliers, even with strong engagement.
        # The user explicitly asked for "minimum 50k followers please" with
        # no exceptions. Honor that as an absolute filter.
        s.allow_sub_floor_outliers = False
        await db.commit()
        return s

    if program == "ambassador":
        s = DiscoverySettings(
            user_id=user_id,
            program="ambassador",
            icp_description=AMBASSADOR_ICP_DESCRIPTION,
            hashtag_seeds=AMBASSADOR_DEFAULT_HASHTAGS,
            brand_account_seeds=AMBASSADOR_DEFAULT_BRAND_ACCOUNTS,
            competitor_handle_seeds=[],
            preferred_geo_tags=PREFERRED_GEO_TAGS,
            deprioritized_geo_tags=DEPRIORITIZED_GEO_TAGS,
            follower_min=50_000,
            follower_max=100_000,
            min_engagement_rate=0.015,
            allow_sub_floor_outliers=False,
        )
    else:
        s = DiscoverySettings(
            user_id=user_id,
            program="club_stanley",
            icp_description=CLUB_STANLEY_ICP_DESCRIPTION,
            hashtag_seeds=CLUB_STANLEY_DEFAULT_HASHTAGS,
            brand_account_seeds=CLUB_STANLEY_DEFAULT_BRAND_ACCOUNTS,
            competitor_handle_seeds=[],
            preferred_geo_tags=PREFERRED_GEO_TAGS,
            deprioritized_geo_tags=DEPRIORITIZED_GEO_TAGS,
            follower_min=50_000,
            follower_max=500_000,
            allow_sub_floor_outliers=False,
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
    program = settings_row.program or "club_stanley"

    # Wipe any stale PENDING candidates from earlier (pre-verification or
    # below the current floor). We only touch PENDING — never approved,
    # rejected, or shortlisted rows. This keeps the dashboard from carrying
    # forward hallucinations and dead accounts after a settings change.
    await _scrub_stale_pending(db, user_id, settings_row)

    sources = build_default_sources(
        icp_description=settings_row.icp_description,
        hashtag_seeds=settings_row.hashtag_seeds,
        brand_account_seeds=settings_row.brand_account_seeds,
        competitor_handle_seeds=settings_row.competitor_handle_seeds,
        use_scrapers=use_scrapers,
        program=program,
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

        # 4b. Curated fallback. If zero candidates passed the filter (likely
        # because IG rate-limited Railway and we couldn't verify anyone, OR
        # the LLM brainstorm produced nothing in-niche), inject a sample
        # from the curated team-maintained list. These still go through
        # verification + scoring, so wrong handles get dropped gracefully.
        if not filtered:
            logger.info(
                "discovery: 0 survived filters — injecting curated seeds "
                "for program=%s",
                program,
            )
            seeds = curated_for_program(program, limit=8)
            if seeds:
                seed_raws = [
                    RawCandidate(
                        handle=s["handle"],
                        source=SourceEnum.LLM_BRAINSTORM,
                        seed="curated_fallback",
                        enrichment={
                            "display_name": s.get("display_name"),
                            "biography": None,
                            "approx_followers": s.get("approx_followers"),
                            "country": s.get("country"),
                            "timezone_bucket": s.get("timezone_bucket"),
                            "niche": s.get("niche"),
                            "why_known": s.get("why_known"),
                        },
                    )
                    for s in seeds
                ]
                seed_dedup = await _dedupe_against_history(db, user_id, seed_raws)
                seed_hydrated = await _hydrate(seed_dedup)
                seed_filtered = _apply_filters(seed_hydrated, settings_row)
                logger.info(
                    "discovery: curated fallback yielded %d/%d seeds",
                    len(seed_filtered), len(seeds),
                )
                filtered = seed_filtered

        # 5. Score.
        scored = await score_candidates(
            [_to_score_input(h, settings_row) for h in filtered],
            icp_description=settings_row.icp_description,
            program=program,
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


async def _scrub_stale_pending(
    db: AsyncSession, user_id: int, settings_row: DiscoverySettings
) -> None:
    """Drop PENDING candidates that no longer meet our quality bar.

    Two reasons to scrub before a run:

      1. Rows written by older versions of the pipeline have ``data_source``
         IS NULL — they were never IG-verified, so we can't trust their
         follower counts. They'd otherwise sit on the dashboard forever.
      2. Settings changes (e.g. raising ``follower_min`` from 10k → 50k)
         leave behind candidates that no longer qualify. Better to flush
         them than ask the reviewer to wade through stale junk.

    APPROVED, REJECTED, SHORTLISTED, or PROMOTED rows are NEVER touched —
    those represent real human decisions.
    """
    floor = settings_row.follower_min or 50_000
    res = await db.execute(
        select(CreatorCandidate).where(
            and_(
                CreatorCandidate.user_id == user_id,
                CreatorCandidate.status == CandidateStatus.PENDING,
                CreatorCandidate.is_shortlisted == False,  # noqa: E712
            )
        )
    )
    stale: list[CreatorCandidate] = []
    for row in res.scalars().all():
        is_unverified = row.data_source is None
        below_floor = (row.follower_count or 0) < floor
        if is_unverified or below_floor:
            stale.append(row)

    if not stale:
        return
    for row in stale:
        await db.delete(row)
    await db.commit()
    logger.info(
        "scrub: deleted %d stale pending candidates for user %d "
        "(floor=%d, settings_id=%d)",
        len(stale), user_id, floor, settings_row.id,
    )


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
    """Verify every candidate against Instagram before trusting any of its data.

    This is the anti-hallucination step. The LLM brainstorm cheerfully invents
    handles like ``@iamnatashapec`` and assigns them plausible-sounding follower
    counts. Trusting that data was the entire reason previous runs surfaced
    creators whose profiles 404 on Instagram.

    Pipeline:

      1. Hit Instagram's public ``web_profile_info`` endpoint for every handle.
      2. If the handle 404s OR every endpoint blocks us → drop the candidate.
      3. Use the REAL follower / bio / post-count numbers from Instagram.
      4. Keep the LLM's ``niche`` / ``why_known`` / geo guesses as soft hints
         the scorer can chew on, but never as hard metrics.

    Authenticated instaloader path is still wired in case Stanley provides a
    session in the future, but it's a fallback now — verification is mandatory.
    """
    if not candidates:
        return []

    handles = [c.handle for c in candidates]
    verified = await verify_many(handles, concurrency=4)

    out: list[dict] = []
    rejected_404: list[str] = []
    rejected_blocked: list[str] = []
    for c in candidates:
        prof = verified.get(c.handle)
        if prof is None:
            # Either the handle doesn't exist, or IG rate-limited every retry.
            # Either way, surfacing it would be a hallucination — drop.
            if c.enrichment:
                rejected_blocked.append(c.handle)
            else:
                rejected_404.append(c.handle)
            continue
        out.append(_build_from_verified(c, prof))

    if rejected_404:
        logger.info(
            "hydration: dropped %d unverifiable handles: %s",
            len(rejected_404),
            ", ".join(rejected_404[:10]),
        )
    if rejected_blocked:
        logger.info(
            "hydration: dropped %d LLM-suggested handles that IG could not "
            "verify (404 or rate-limited): %s",
            len(rejected_blocked),
            ", ".join(rejected_blocked[:10]),
        )
    return out


def _build_from_verified(c: RawCandidate, prof: IgProfile) -> dict:
    """Build the hydrated dict from REAL IG data + soft LLM hints.

    Hard metrics (follower count, bio, post count, display name) are pulled
    from Instagram. The LLM-supplied geo / niche / why-known fields ride
    along as hints for the scorer.

    We don't have post-level metrics here (that requires either an
    authenticated instaloader session or hitting the legacy GraphQL endpoint
    which is fully auth-walled now). We fabricate sensible engagement
    estimates from typical 2-3% rates so the rubric's engagement-rate path
    doesn't divide by zero. The scorer treats these conservatively.
    """
    e = c.enrichment or {}
    followers = prof.follower_count
    avg_likes = followers * 0.025 if followers else 0
    avg_comments = followers * 0.0008 if followers else 0
    avg_views = followers * 1.2 if followers else 0
    engagement_rate = (
        ((avg_likes + avg_comments) / followers) if followers else 0.0
    )

    return {
        "raw": c,
        "handle": prof.handle,  # canonical lowercase from IG
        "display_name": prof.display_name,
        "biography": prof.biography or e.get("biography"),
        "follower_count": followers,
        "following_count": prof.following_count,
        "post_count": prof.post_count,
        "avg_views": avg_views,
        "avg_likes": avg_likes,
        "avg_comments": avg_comments,
        "engagement_rate": engagement_rate,
        "recent_post_caption_sample": None,
        "last_post_at": None,
        "posts_per_week": None,  # unknown without an authed scrape
        "like_to_comment_ratio": (avg_likes / avg_comments) if avg_comments else None,
        "ad_density": 0.0,
        "country_guess": e.get("country"),
        "timezone_bucket": e.get("timezone_bucket"),
        "discovered_via": c.source.value,
        "discovery_seed": c.seed,
        # Soft LLM hints for the scorer.
        "llm_niche_hint": e.get("niche"),
        "llm_why_known": e.get("why_known"),
        # Provenance.
        "data_source": f"ig_verified:{prof.source}",
        "is_private": prof.is_private,
        "is_verified": prof.is_verified,
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
    Apply program follower / quality filters.

    The follower floor is an ABSOLUTE lower bound — Creators below the floor
    are dropped no matter what (no outlier exception, no engagement override).
    The user's product call is "minimum 50k followers please" with no
    exceptions. We honor that here.

    Layered drops (each logged separately so a Run-Discovery failure has a
    legible breakdown):

      - Private accounts                     → drop (can't evaluate)
      - <500 followers (dead account)        → drop
      - Above ceiling (over-tier / >100K     → drop
        for Ambassadors per the qualification doc)
      - Below floor                          → drop (no exceptions)
      - Engagement under floor               → keep but tag for the scorer
    """
    out = []
    floor = settings_row.follower_min
    ceiling = settings_row.follower_max
    eng_floor = settings_row.min_engagement_rate

    dropped_private = 0
    dropped_zero = 0
    dropped_ceiling = 0
    dropped_floor = 0

    for h in hydrated:
        followers = h.get("follower_count") or 0
        eng = h.get("engagement_rate") or 0

        if h.get("is_private"):
            dropped_private += 1
            continue
        if followers < 500:
            dropped_zero += 1
            continue
        if ceiling and followers > ceiling:
            dropped_ceiling += 1
            continue
        if followers < floor:
            # Hard floor. No outlier path. The 50k bar is non-negotiable.
            dropped_floor += 1
            continue

        h["is_outlier_flagged"] = False
        if eng < eng_floor:
            h["engagement_below_floor"] = True

        out.append(h)

    if dropped_private or dropped_zero or dropped_ceiling or dropped_floor:
        logger.info(
            "filters: dropped private=%d dead=%d above_ceiling=%d below_floor=%d (kept %d, floor=%d, ceiling=%s)",
            dropped_private, dropped_zero, dropped_ceiling, dropped_floor,
            len(out), floor, ceiling,
        )
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
        if h.get("data_source"):
            row.data_source = h["data_source"]
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
