"""
Background velocity scanner.

Orchestrates the full pipeline: ingest -> detect -> rewrite -> alert.
Runs on a configurable interval via APScheduler.
"""
import logging
from datetime import datetime, timedelta

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import async_session
from app.models.models import User, TrackedCreator, VelocityAlert, AlertStatus
from app.services.instagram import ingest_creator_posts
from app.services.velocity import VelocityEngine
from app.services.notifications import generate_alert

logger = logging.getLogger(__name__)


async def run_velocity_scan(user_id: int | None = None):
    """
    Execute a full scan cycle for one or all users.

    Pipeline:
    1. For each user, fetch fresh data for all tracked creators
    2. Run velocity detection on each creator's recent posts
    3. For each spike above threshold, check cooldown and generate alert
    4. Push notifications for new alerts
    """
    async with async_session() as db:
        if user_id:
            users_result = await db.execute(
                select(User).where(User.id == user_id)
            )
            users = [users_result.scalar_one_or_none()]
            users = [u for u in users if u]
        else:
            users_result = await db.execute(select(User))
            users = users_result.scalars().all()

        total_posts = 0
        total_spikes = 0
        total_alerts = 0
        all_alerts = []

        engine = VelocityEngine()

        for user in users:
            creators_result = await db.execute(
                select(TrackedCreator).where(
                    and_(
                        TrackedCreator.user_id == user.id,
                        TrackedCreator.is_active == True,
                    )
                )
            )
            creators = creators_result.scalars().all()

            for creator in creators:
                try:
                    posts = await ingest_creator_posts(db, creator)
                    total_posts += len(posts)

                    spikes = await engine.analyze_creator(db, creator)
                    total_spikes += len(spikes)

                    for spike in spikes:
                        if await _is_cooldown_active(db, user.id, spike.post.id):
                            logger.debug(
                                f"Skipping alert for post {spike.post.id} — cooldown active"
                            )
                            continue

                        alert = await generate_alert(db, user, spike)
                        total_alerts += 1
                        all_alerts.append(alert)
                        logger.info(
                            f"Alert generated: {alert.creator_handle} "
                            f"({spike.velocity_multiplier}x) for user {user.username}"
                        )

                except Exception as e:
                    logger.error(
                        f"Scan failed for creator {creator.instagram_handle}: {e}",
                        exc_info=True,
                    )

        await db.commit()

        logger.info(
            f"Scan complete: {total_posts} posts scanned, "
            f"{total_spikes} spikes detected, {total_alerts} alerts generated"
        )
        return {
            "posts_scanned": total_posts,
            "spikes_detected": total_spikes,
            "alerts_generated": total_alerts,
            "alerts": all_alerts,
        }


async def _is_cooldown_active(
    db: AsyncSession, user_id: int, post_id: int
) -> bool:
    """Prevent duplicate alerts for the same post within the cooldown window."""
    cooldown_cutoff = datetime.utcnow() - timedelta(
        hours=settings.alert_cooldown_hours
    )
    result = await db.execute(
        select(VelocityAlert).where(
            and_(
                VelocityAlert.user_id == user_id,
                VelocityAlert.post_id == post_id,
                VelocityAlert.created_at >= cooldown_cutoff,
                VelocityAlert.status != AlertStatus.EXPIRED,
            )
        )
    )
    return result.scalar_one_or_none() is not None
