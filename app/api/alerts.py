from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, and_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.database import get_db
from app.models.models import (
    User, VelocityAlert, AlertStatus, AlertUrgency,
    CreatorPost, TrackedCreator
)
from app.schemas.schemas import (
    AlertResponse, AlertFeedResponse, VelocityFeedItem,
    VelocityFeedResponse, TriggerScanRequest, TriggerScanResponse,
)
from app.services.scanner import run_velocity_scan

router = APIRouter(prefix="/users/{user_id}", tags=["alerts"])


@router.get("/alerts", response_model=AlertFeedResponse)
async def get_alerts(
    user_id: int,
    urgency: str | None = Query(None, description="Filter by urgency level"),
    status: str | None = Query(None, description="Filter by alert status"),
    limit: int = Query(20, le=100),
    db: AsyncSession = Depends(get_db),
):
    conditions = [VelocityAlert.user_id == user_id]
    if urgency:
        conditions.append(VelocityAlert.urgency == urgency)
    if status:
        conditions.append(VelocityAlert.status == status)

    result = await db.execute(
        select(VelocityAlert)
        .where(and_(*conditions))
        .order_by(desc(VelocityAlert.created_at))
        .limit(limit)
    )
    alerts = result.scalars().all()

    total_result = await db.execute(
        select(func.count()).select_from(VelocityAlert).where(
            VelocityAlert.user_id == user_id
        )
    )
    total = total_result.scalar() or 0

    pending_result = await db.execute(
        select(func.count()).select_from(VelocityAlert).where(
            and_(
                VelocityAlert.user_id == user_id,
                VelocityAlert.status == AlertStatus.PENDING,
            )
        )
    )
    pending = pending_result.scalar() or 0

    return AlertFeedResponse(alerts=alerts, total=total, pending_count=pending)


@router.get("/alerts/{alert_id}", response_model=AlertResponse)
async def get_alert_detail(
    user_id: int,
    alert_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(VelocityAlert).where(
            and_(
                VelocityAlert.id == alert_id,
                VelocityAlert.user_id == user_id,
            )
        )
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(404, "Alert not found")
    if alert.status == AlertStatus.PENDING:
        alert.status = AlertStatus.OPENED
        alert.opened_at = datetime.utcnow()
        await db.commit()
    return alert


@router.post("/alerts/{alert_id}/act")
async def mark_alert_acted(
    user_id: int,
    alert_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(VelocityAlert).where(
            and_(
                VelocityAlert.id == alert_id,
                VelocityAlert.user_id == user_id,
            )
        )
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(404, "Alert not found")
    alert.status = AlertStatus.ACTED_ON
    await db.commit()
    return {"status": "marked_acted"}


@router.post("/alerts/{alert_id}/dismiss")
async def dismiss_alert(
    user_id: int,
    alert_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(VelocityAlert).where(
            and_(
                VelocityAlert.id == alert_id,
                VelocityAlert.user_id == user_id,
            )
        )
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(404, "Alert not found")
    alert.status = AlertStatus.DISMISSED
    await db.commit()
    return {"status": "dismissed"}


@router.get("/velocity-feed", response_model=VelocityFeedResponse)
async def get_velocity_feed(
    user_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Real-time velocity feed showing all tracked creator posts ranked by multiplier."""
    creators_result = await db.execute(
        select(TrackedCreator).where(
            and_(
                TrackedCreator.user_id == user_id,
                TrackedCreator.is_active == True,
            )
        )
    )
    creators = creators_result.scalars().all()
    creator_ids = [c.id for c in creators]
    creator_map = {c.id: c for c in creators}

    if not creator_ids:
        return VelocityFeedResponse(items=[], spike_count=0, last_scan_at=None)

    posts_result = await db.execute(
        select(CreatorPost)
        .where(CreatorPost.creator_id.in_(creator_ids))
        .order_by(desc(CreatorPost.velocity_multiplier))
        .limit(50)
    )
    posts = posts_result.scalars().all()

    alert_post_ids_result = await db.execute(
        select(VelocityAlert.post_id).where(VelocityAlert.user_id == user_id)
    )
    alert_post_ids = {row[0] for row in alert_post_ids_result.all()}

    items = []
    spike_count = 0
    for post in posts:
        creator = creator_map.get(post.creator_id)
        is_spike = post.is_spike or False
        if is_spike:
            spike_count += 1
        items.append(VelocityFeedItem(
            creator_handle=creator.instagram_handle if creator else "unknown",
            creator_name=creator.display_name if creator else None,
            post_url=post.post_url,
            caption_preview=(post.caption or "")[:120],
            views=post.views,
            velocity_multiplier=post.velocity_multiplier or 0,
            hours_since_post=post.hours_since_post or 0,
            detected_format=post.detected_format,
            is_spike=is_spike,
            alert_generated=post.id in alert_post_ids,
        ))

    last_scan = max(
        (c.last_scraped_at for c in creators if c.last_scraped_at), default=None
    )
    return VelocityFeedResponse(
        items=items, spike_count=spike_count, last_scan_at=last_scan
    )


@router.post("/scan", response_model=TriggerScanResponse)
async def trigger_scan(
    user_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Manually trigger a velocity scan for a user. Used for demos and testing."""
    user = await db.execute(select(User).where(User.id == user_id))
    if not user.scalar_one_or_none():
        raise HTTPException(404, "User not found")

    result = await run_velocity_scan(user_id=user_id)
    return TriggerScanResponse(
        posts_scanned=result["posts_scanned"],
        spikes_detected=result["spikes_detected"],
        alerts_generated=result["alerts_generated"],
        alerts=result["alerts"],
    )
