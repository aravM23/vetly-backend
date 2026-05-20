"""
Timing service — seeds, aggregates, and simulates niche timing data.

Demo-ready pipeline that replaces the synthetic frontend model with a real
database-backed one. The data it generates follows realistic per-niche hourly
distributions so the chart is honest in shape even before real Instagram data
flows in.

Flow:
  seed_demo_timing_data()     → populates niches + ~2k posts per niche
  recompute_niche_hourly_stats() → (re)computes percentile stats per bucket
  simulate_batch()            → appends a trend surge + recomputes
"""
import logging
import math
import random
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.timing_models import Niche, NichePost, NicheHourlyStat

logger = logging.getLogger(__name__)


# ─── Niche catalogue ───────────────────────────────────────────
# Each niche has two peak hours (in 24h clock) driving its engagement rhythm.
# `weight` scales the strength of each peak.

NICHE_CATALOGUE = [
    {
        "slug": "travel",
        "label": "Travel",
        "color_hex": "#22d3ee",
        "blurb": "Daydream hours — late morning and post-dinner browse.",
        "peaks": [(9.0, 0.85), (20.5, 1.0)],
        "handles": [
            "wanderlust.diaries", "atlas.unbound", "passport.pages",
            "offgrid.emma", "the.layover",
        ],
    },
    {
        "slug": "growth",
        "label": "Growth Coach",
        "color_hex": "#a78bfa",
        "blurb": "Commute-first audience. Morning ritual beats evening scroll.",
        "peaks": [(6.5, 1.0), (18.0, 0.9)],
        "handles": [
            "dailymindset.co", "peakperform", "discipline.daily",
            "thefoundrmethod", "6amclub.official",
        ],
    },
    {
        "slug": "fashion",
        "label": "Fashion",
        "color_hex": "#f472b6",
        "blurb": "Lunch scroll + prime-time inspiration window.",
        "peaks": [(12.5, 0.75), (19.5, 1.0)],
        "handles": [
            "closet.theory", "seamed.style", "minimal.edit",
            "fits.byzo", "quietluxe",
        ],
    },
    {
        "slug": "tech",
        "label": "Tech / Startup",
        "color_hex": "#34d399",
        "blurb": "Early checkers and late-night builders. Skips midday.",
        "peaks": [(8.0, 0.85), (22.0, 1.0)],
        "handles": [
            "shipit.daily", "yc.adjacent", "founder.notes",
            "latenight.commits", "prototyped",
        ],
    },
    {
        "slug": "fitness",
        "label": "Fitness",
        "color_hex": "#fb923c",
        "blurb": "Pre-workout morning + post-work motivation double peak.",
        "peaks": [(5.5, 1.0), (17.0, 0.9)],
        "handles": [
            "5am.lifts", "formfirst.fit", "leantracked",
            "strength.journal", "repsbyalex",
        ],
    },
]


def _bell(h: float, center: float, spread: float = 2.7) -> float:
    """Wrapped Gaussian — h=23 and h=0 are neighbors."""
    d = min(abs(h - center), 24 - abs(h - center))
    return math.exp(-(d * d) / (2 * spread * spread))


def _hour_energy(hour: int, peaks: list[tuple[float, float]], is_weekend: bool) -> float:
    """Expected engagement strength for a given hour in this niche."""
    effective_peaks = [
        ((h + 1.8) % 24, w * (0.92 if is_weekend else 1.0)) for h, w in peaks
    ] if is_weekend else peaks
    return max(_bell(hour, h, 2.7) * w for h, w in effective_peaks)


def _sample_post_for_hour(rng: random.Random, energy: float) -> float:
    """
    Draw a lift_24h sample for a post published at an hour with this energy.

    Log-normal so the distribution is right-skewed (a few viral outliers per
    bucket) and the median scales with the hour's engagement weight.
    """
    mu = math.log(0.35 + 1.9 * energy)
    sigma = 0.55
    lift = rng.lognormvariate(mu, sigma)
    return min(lift, 12.0)  # clip silly outliers


# ─── Seeding ──────────────────────────────────────────────────

async def seed_demo_timing_data(db: AsyncSession, posts_per_niche: int = 1800) -> dict:
    """
    Populate the niches + ~N posts per niche. Idempotent: if data already exists,
    returns counts and does nothing.
    """
    existing_niche_count = await db.scalar(select(func.count()).select_from(Niche))
    if existing_niche_count and existing_niche_count >= len(NICHE_CATALOGUE):
        post_count = await db.scalar(select(func.count()).select_from(NichePost))
        logger.info(f"Timing data already seeded ({post_count} posts). Skipping.")
        return {"niches": existing_niche_count, "posts": post_count, "seeded": False}

    now = datetime.utcnow()
    rng = random.Random(42)  # deterministic seed

    total_posts = 0
    for cat in NICHE_CATALOGUE:
        existing = await db.scalar(
            select(Niche).where(Niche.slug == cat["slug"])
        )
        if existing:
            niche = existing
        else:
            niche = Niche(
                slug=cat["slug"],
                label=cat["label"],
                color_hex=cat["color_hex"],
                blurb=cat["blurb"],
            )
            db.add(niche)
            await db.flush()

        posts = []
        for _ in range(posts_per_niche):
            days_ago = rng.uniform(0, 30)
            posted_at = now - timedelta(days=days_ago)
            is_weekend = posted_at.weekday() >= 5

            hour_weights = [
                _hour_energy(h, cat["peaks"], is_weekend) + 0.08  # uniform floor
                for h in range(24)
            ]
            total = sum(hour_weights)
            r = rng.uniform(0, total)
            acc = 0
            chosen_hour = 0
            for h, w in enumerate(hour_weights):
                acc += w
                if r <= acc:
                    chosen_hour = h
                    break

            energy = _hour_energy(chosen_hour, cat["peaks"], is_weekend)
            lift = _sample_post_for_hour(rng, energy)
            baseline = rng.uniform(25_000, 180_000)

            posts.append(NichePost(
                niche_id=niche.id,
                creator_handle=rng.choice(cat["handles"]),
                publish_hour_local=chosen_hour,
                publish_day_type="weekend" if is_weekend else "weekday",
                lift_24h=round(lift, 3),
                views_at_24h=int(lift * baseline),
                baseline_used=baseline,
                posted_at=posted_at,
            ))

        db.add_all(posts)
        total_posts += len(posts)
        logger.info(f"Seeded {len(posts)} posts for niche {cat['slug']}")

    await db.commit()
    logger.info(f"Timing seed complete: {total_posts} posts across {len(NICHE_CATALOGUE)} niches")
    return {"niches": len(NICHE_CATALOGUE), "posts": total_posts, "seeded": True}


# ─── Aggregation ──────────────────────────────────────────────

async def recompute_niche_hourly_stats(db: AsyncSession) -> dict:
    """
    Recompute percentile buckets for every (niche, hour, day_type).
    Wipes the stats table and rebuilds — cheap because there are only ~240 rows.
    """
    await db.execute(delete(NicheHourlyStat))

    niches_result = await db.execute(select(Niche))
    niches = niches_result.scalars().all()

    computed_at = datetime.utcnow()
    bucket_count = 0

    for niche in niches:
        posts_result = await db.execute(
            select(NichePost.publish_hour_local, NichePost.publish_day_type, NichePost.lift_24h)
            .where(NichePost.niche_id == niche.id)
        )
        rows = posts_result.all()
        if not rows:
            continue

        # Group lift values by (hour, day_type)
        buckets: dict[tuple[int, str], list[float]] = {}
        for hour, day_type, lift in rows:
            buckets.setdefault((hour, day_type), []).append(lift)

        for (hour, day_type), lifts in buckets.items():
            arr = np.array(lifts, dtype=float)
            db.add(NicheHourlyStat(
                niche_id=niche.id,
                hour=hour,
                day_type=day_type,
                p10=float(np.percentile(arr, 10)),
                p25=float(np.percentile(arr, 25)),
                p50=float(np.percentile(arr, 50)),
                p75=float(np.percentile(arr, 75)),
                p90=float(np.percentile(arr, 90)),
                sample_size=len(lifts),
                computed_at=computed_at,
            ))
            bucket_count += 1

    await db.commit()
    logger.info(f"Recomputed {bucket_count} hourly stat buckets")
    return {"buckets": bucket_count, "niches": len(niches), "computed_at": computed_at}


# ─── Simulation ──────────────────────────────────────────────

async def simulate_batch(
    db: AsyncSession,
    niche_slug: str | None = None,
    batch_size: int = 80,
) -> dict:
    """
    Inject a batch of new posts with a surge concentrated at one random hour,
    so the chart visibly moves on the next render. Then recompute stats.
    """
    if niche_slug:
        niche = await db.scalar(select(Niche).where(Niche.slug == niche_slug))
        target_niches = [niche] if niche else []
    else:
        target_niches = (await db.execute(select(Niche))).scalars().all()

    if not target_niches:
        return {"added": 0, "surge_hour": None, "niche_slug": niche_slug}

    rng = random.Random()
    surge_hour = rng.randint(0, 23)
    surge_intensity = rng.uniform(1.8, 3.2)  # how much lift to boost
    now = datetime.utcnow()

    added = 0
    for niche in target_niches:
        cat = next((n for n in NICHE_CATALOGUE if n["slug"] == niche.slug), None)
        peaks = cat["peaks"] if cat else [(12.0, 1.0)]
        handle_pool = cat["handles"] if cat else ["unknown.handle"]

        for _ in range(batch_size // max(len(target_niches), 1)):
            hours_ago = rng.uniform(0.25, 12)
            posted_at = now - timedelta(hours=hours_ago)
            is_weekend = posted_at.weekday() >= 5

            # Half of the surge batch lands in the surge hour, half spreads normally
            if rng.random() < 0.6:
                chosen_hour = surge_hour
                energy_boost = surge_intensity
            else:
                chosen_hour = rng.randint(0, 23)
                energy_boost = 1.0

            base_energy = _hour_energy(chosen_hour, peaks, is_weekend)
            effective_energy = min(base_energy * energy_boost, 2.2)
            lift = _sample_post_for_hour(rng, effective_energy)
            baseline = rng.uniform(25_000, 180_000)

            db.add(NichePost(
                niche_id=niche.id,
                creator_handle=rng.choice(handle_pool),
                publish_hour_local=chosen_hour,
                publish_day_type="weekend" if is_weekend else "weekday",
                lift_24h=round(lift, 3),
                views_at_24h=int(lift * baseline),
                baseline_used=baseline,
                posted_at=posted_at,
            ))
            added += 1

    await db.commit()
    recompute = await recompute_niche_hourly_stats(db)

    return {
        "added": added,
        "surge_hour": surge_hour,
        "surge_intensity": round(surge_intensity, 2),
        "niche_slug": niche_slug,
        "buckets_recomputed": recompute["buckets"],
    }
