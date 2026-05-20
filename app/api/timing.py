"""
Timing API — serves niche hourly distributions to the frontend chart.

GET  /api/timing/niches              → list of niches with metadata
GET  /api/timing/{slug}              → candles for a niche (weekday|weekend)
POST /api/timing/simulate-batch      → add a surge batch, recompute, return stats
POST /api/timing/recompute           → force recompute of all buckets
"""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.timing_models import Niche, NichePost, NicheHourlyStat
from app.services.timing_service import (
    simulate_batch,
    recompute_niche_hourly_stats,
)

router = APIRouter(prefix="/timing", tags=["timing"])


@router.get("/niches")
async def list_niches(db: AsyncSession = Depends(get_db)):
    rows = await db.execute(select(Niche).order_by(Niche.label))
    niches = rows.scalars().all()

    total_posts = await db.scalar(select(func.count()).select_from(NichePost)) or 0
    latest_computed = await db.scalar(
        select(func.max(NicheHourlyStat.computed_at))
    )

    return {
        "niches": [
            {
                "slug": n.slug,
                "label": n.label,
                "color_hex": n.color_hex,
                "blurb": n.blurb,
            }
            for n in niches
        ],
        "total_posts": total_posts,
        "last_computed_at": latest_computed.isoformat() if latest_computed else None,
    }


@router.get("/{niche_slug}")
async def get_timing(
    niche_slug: str,
    day_type: str = Query("weekday", pattern="^(weekday|weekend)$"),
    db: AsyncSession = Depends(get_db),
):
    niche = await db.scalar(select(Niche).where(Niche.slug == niche_slug))
    if not niche:
        raise HTTPException(404, f"Niche '{niche_slug}' not found")

    stats_result = await db.execute(
        select(NicheHourlyStat)
        .where(NicheHourlyStat.niche_id == niche.id)
        .where(NicheHourlyStat.day_type == day_type)
    )
    stats = {s.hour: s for s in stats_result.scalars().all()}

    candles = []
    for h in range(24):
        s = stats.get(h)
        if s:
            candles.append({
                "hour": h,
                "low": round(s.p10, 3),
                "p25": round(s.p25, 3),
                "median": round(s.p50, 3),
                "p75": round(s.p75, 3),
                "high": round(s.p90, 3),
                "n": s.sample_size,
                "confidence": "high" if s.sample_size >= 20 else "low",
            })
        else:
            candles.append({
                "hour": h,
                "low": 0.1, "p25": 0.2, "median": 0.3, "p75": 0.4, "high": 0.5,
                "n": 0,
                "confidence": "empty",
            })

    latest_computed = max(
        (s.computed_at for s in stats.values()), default=None
    )
    total_samples = sum(s.sample_size for s in stats.values())

    return {
        "niche": {
            "slug": niche.slug,
            "label": niche.label,
            "color_hex": niche.color_hex,
            "blurb": niche.blurb,
        },
        "day_type": day_type,
        "candles": candles,
        "total_samples": total_samples,
        "computed_at": latest_computed.isoformat() if latest_computed else None,
    }


@router.post("/simulate-batch")
async def post_simulate_batch(
    niche_slug: str | None = Query(None),
    batch_size: int = Query(80, ge=10, le=500),
    db: AsyncSession = Depends(get_db),
):
    """
    Demo action: inject a batch of new posts with a surge at a random hour,
    recompute stats, return the surge details so the UI can highlight.
    """
    result = await simulate_batch(db, niche_slug=niche_slug, batch_size=batch_size)
    return result


@router.post("/recompute")
async def post_recompute(db: AsyncSession = Depends(get_db)):
    """Force a full recomputation of all hourly stat buckets."""
    return await recompute_niche_hourly_stats(db)
