"""
Alert generation and push notification delivery.

Generates the push notification payload from a spike detection,
then delivers via Firebase Cloud Messaging. Falls back to logging
for development.
"""
import logging
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.models import User, VelocityAlert, AlertStatus, AlertUrgency, CreatorPost
from app.services.velocity import SpikeDetection
from app.services.content_rewriter import generate_draft

logger = logging.getLogger(__name__)

# Firebase is optional — only imported if configured
_fcm_app = None


def _init_firebase():
    global _fcm_app
    if _fcm_app is not None:
        return
    if not settings.firebase_credentials_path:
        return
    try:
        import firebase_admin
        from firebase_admin import credentials
        cred = credentials.Certificate(settings.firebase_credentials_path)
        _fcm_app = firebase_admin.initialize_app(cred)
    except Exception as e:
        logger.warning(f"Firebase init failed (push notifications disabled): {e}")


HEADLINE_TEMPLATES = {
    AlertUrgency.CRITICAL: (
        "{creator} just hit {views:,} views in {hours:.0f}h. "
        "Your draft is ready — tap to ride this wave before it peaks."
    ),
    AlertUrgency.HIGH: (
        "{creator} is at {multiplier:.1f}x their average using a {format} format. "
        "I already rewrote it for your pillars. Open your draft."
    ),
    AlertUrgency.MEDIUM: (
        "{creator} is gaining traction ({multiplier:.1f}x) with a {format}. "
        "Draft available if you want to catch this wave."
    ),
    AlertUrgency.LOW: (
        "Heads up: {creator} posted a {format} that's picking up steam ({multiplier:.1f}x)."
    ),
}

BODY_TEMPLATES = {
    AlertUrgency.CRITICAL: (
        "{creator} hit {views:,} views in just {hours:.0f} hours using a {format} format. "
        "The algorithm is actively pushing this wave. I've already reverse-engineered their "
        "exact visual beat structure and rewrote it using your core content pillars. "
        "Estimated window: ~{peak_hours:.0f}h before this wave peaks. "
        "Tap to open your draft flow."
    ),
    AlertUrgency.HIGH: (
        "{creator} is running at {multiplier:.1f}x their normal engagement with a {format}. "
        "This format is triggering high early-retention signals. "
        "I've built you a draft using your narrative ({narrative}). "
        "Window: ~{peak_hours:.0f}h remaining."
    ),
    AlertUrgency.MEDIUM: (
        "{creator} is outperforming at {multiplier:.1f}x with a {format}. "
        "A draft is ready based on your content pillars if you want to move on this."
    ),
    AlertUrgency.LOW: (
        "{creator} posted a {format} performing at {multiplier:.1f}x. Worth watching."
    ),
}


async def generate_alert(
    db: AsyncSession,
    user: User,
    spike: SpikeDetection,
) -> VelocityAlert:
    """Generate a complete alert with rewritten draft content."""
    draft = await generate_draft(user, spike.post, spike.velocity_multiplier)

    pillars = user.content_pillars or {}
    narrative = pillars.get("primary_narrative", "your content")
    creator_handle = spike.post.creator.instagram_handle if spike.post.creator else "A creator"
    fmt = spike.post.detected_format or "content"

    template_vars = {
        "creator": creator_handle,
        "views": spike.post.views or 0,
        "hours": spike.hours_since_post,
        "multiplier": spike.velocity_multiplier,
        "format": fmt,
        "peak_hours": spike.estimated_peak_hours or 0,
        "narrative": narrative,
    }

    headline = HEADLINE_TEMPLATES[spike.urgency].format(**template_vars)
    body = BODY_TEMPLATES[spike.urgency].format(**template_vars)

    alert = VelocityAlert(
        user_id=user.id,
        post_id=spike.post.id,
        creator_handle=creator_handle,
        velocity_multiplier=spike.velocity_multiplier,
        views_at_detection=spike.post.views,
        hours_since_post=spike.hours_since_post,
        detected_format=fmt,
        alert_headline=headline,
        alert_body=body,
        draft_hook=draft.hook,
        draft_structure=draft.structure,
        rewrite_rationale=draft.rationale,
        urgency=spike.urgency,
        status=AlertStatus.PENDING,
        estimated_peak_hours=spike.estimated_peak_hours,
    )

    db.add(alert)
    await db.commit()
    await db.refresh(alert)

    if user.push_token and user.notification_enabled:
        await _send_push(user.push_token, alert)

    return alert


async def _send_push(token: str, alert: VelocityAlert) -> bool:
    """Send push notification via FCM. Falls back to log output."""
    _init_firebase()

    payload = {
        "title": _push_title(alert),
        "body": alert.alert_headline,
        "data": {
            "alert_id": str(alert.id),
            "urgency": alert.urgency,
            "action": "open_draft",
        },
    }

    if _fcm_app is not None:
        try:
            from firebase_admin import messaging
            message = messaging.Message(
                notification=messaging.Notification(
                    title=payload["title"],
                    body=payload["body"],
                ),
                data=payload["data"],
                token=token,
                android=messaging.AndroidConfig(
                    priority="high",
                    notification=messaging.AndroidNotification(
                        channel_id="velocity_alerts",
                        priority="max",
                    ),
                ),
                apns=messaging.APNSConfig(
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(
                            alert=messaging.ApsAlert(
                                title=payload["title"],
                                body=payload["body"],
                            ),
                            sound="default",
                            badge=1,
                        ),
                    ),
                ),
            )
            messaging.send(message)
            alert.status = AlertStatus.SENT
            alert.sent_at = datetime.utcnow()
            logger.info(f"Push sent for alert {alert.id} (urgency={alert.urgency})")
            return True
        except Exception as e:
            logger.error(f"FCM send failed for alert {alert.id}: {e}")
            return False
    else:
        logger.info(
            f"[MOCK PUSH] Alert {alert.id} | {payload['title']}\n"
            f"  → {payload['body']}\n"
            f"  → Urgency: {alert.urgency} | Data: {payload['data']}"
        )
        alert.status = AlertStatus.SENT
        alert.sent_at = datetime.utcnow()
        return True


def _push_title(alert: VelocityAlert) -> str:
    urgency_prefix = {
        AlertUrgency.CRITICAL: "WAVE ALERT",
        AlertUrgency.HIGH: "Trend Spike",
        AlertUrgency.MEDIUM: "Velocity Alert",
        AlertUrgency.LOW: "Trend Watch",
    }
    prefix = urgency_prefix.get(AlertUrgency(alert.urgency), "Alert")
    return f"{prefix} — {alert.creator_handle}"
