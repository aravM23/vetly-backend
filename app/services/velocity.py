"""
Velocity detection engine.

Calculates view velocity, detects multiplier spikes, estimates peak timing,
and scores urgency. This is the algorithmic core that decides when to fire alerts.
"""
import logging
import math
from datetime import datetime, timedelta
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.models import (
    TrackedCreator, CreatorPost, PostSnapshot, AlertUrgency
)

logger = logging.getLogger(__name__)


@dataclass
class SpikeDetection:
    post: CreatorPost
    velocity_multiplier: float
    view_velocity: float          # views per hour right now
    hours_since_post: float
    acceleration: float           # is velocity increasing or decreasing?
    estimated_peak_hours: float   # hours until wave crests
    urgency: AlertUrgency
    confidence: float             # 0-1 confidence this is a real spike


class VelocityEngine:

    def __init__(self, spike_threshold: float | None = None):
        self.spike_threshold = spike_threshold or settings.velocity_spike_threshold

    async def analyze_creator(
        self, db: AsyncSession, creator: TrackedCreator
    ) -> list[SpikeDetection]:
        """Analyze all recent posts from a creator for velocity spikes."""
        cutoff = datetime.utcnow() - timedelta(hours=72)
        result = await db.execute(
            select(CreatorPost).where(
                and_(
                    CreatorPost.creator_id == creator.id,
                    CreatorPost.posted_at >= cutoff,
                )
            ).order_by(CreatorPost.posted_at.desc())
        )
        recent_posts = result.scalars().all()
        if not recent_posts:
            return []

        baseline_views = creator.avg_views or 1
        spikes = []

        for post in recent_posts:
            detection = await self._evaluate_post(db, post, baseline_views)
            if detection and detection.velocity_multiplier >= self.spike_threshold:
                spikes.append(detection)

        spikes.sort(key=lambda s: s.velocity_multiplier, reverse=True)
        return spikes

    async def _evaluate_post(
        self, db: AsyncSession, post: CreatorPost, baseline_views: float
    ) -> SpikeDetection | None:
        if not post.posted_at:
            return None

        hours_since = (datetime.utcnow() - post.posted_at).total_seconds() / 3600
        if hours_since < 0.5:
            return None  # too early, not enough signal

        current_views = post.views or 0
        if current_views < settings.min_views_threshold:
            return None

        velocity_multiplier = current_views / max(baseline_views, 1)

        snapshots_result = await db.execute(
            select(PostSnapshot)
            .where(PostSnapshot.post_id == post.id)
            .order_by(PostSnapshot.captured_at.asc())
        )
        snapshots = snapshots_result.scalars().all()

        view_velocity = current_views / max(hours_since, 0.5)
        acceleration = self._calculate_acceleration(snapshots)
        estimated_peak = self._estimate_peak_hours(
            hours_since, velocity_multiplier, acceleration
        )
        urgency = self._score_urgency(velocity_multiplier, hours_since, acceleration)
        confidence = self._calculate_confidence(
            len(snapshots), velocity_multiplier, hours_since
        )

        post.view_velocity = view_velocity
        post.velocity_multiplier = velocity_multiplier
        post.hours_since_post = hours_since
        post.is_spike = velocity_multiplier >= self.spike_threshold

        return SpikeDetection(
            post=post,
            velocity_multiplier=round(velocity_multiplier, 2),
            view_velocity=round(view_velocity, 1),
            hours_since_post=round(hours_since, 1),
            acceleration=round(acceleration, 3),
            estimated_peak_hours=round(estimated_peak, 1),
            urgency=urgency,
            confidence=round(confidence, 2),
        )

    def _calculate_acceleration(self, snapshots: list[PostSnapshot]) -> float:
        """
        Positive = views accelerating (wave still building).
        Negative = views decelerating (wave cresting/dying).
        """
        if len(snapshots) < 2:
            return 0.0

        recent = snapshots[-min(5, len(snapshots)):]
        if len(recent) < 2:
            return 0.0

        velocities = []
        for i in range(1, len(recent)):
            dt = (recent[i].captured_at - recent[i - 1].captured_at).total_seconds() / 3600
            if dt > 0:
                dv = (recent[i].views - recent[i - 1].views) / dt
                velocities.append(dv)

        if len(velocities) < 2:
            return 0.0

        # Linear regression slope on velocity values = acceleration
        x = np.arange(len(velocities), dtype=float)
        y = np.array(velocities, dtype=float)
        if np.std(x) == 0:
            return 0.0
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def _estimate_peak_hours(
        self, hours_since: float, multiplier: float, acceleration: float
    ) -> float:
        """
        Estimate how many hours until the algorithmic wave peaks.

        Uses a logistic decay model: viral posts on Instagram typically peak
        between 4-24 hours. Higher acceleration means the peak is further out.
        """
        if acceleration <= 0:
            # Already decelerating — peak is now or passed
            return max(0, 2 - hours_since * 0.5)

        # Base peak time depends on the multiplier magnitude
        # Higher multiplier = algorithm is pushing harder = longer wave
        base_peak = 6 + math.log(max(multiplier, 1)) * 4

        # Acceleration shifts peak forward
        accel_factor = min(acceleration / 1000, 2.0)
        estimated = base_peak * (1 + accel_factor * 0.3) - hours_since

        return max(estimated, 0.5)

    def _score_urgency(
        self, multiplier: float, hours_since: float, acceleration: float
    ) -> AlertUrgency:
        if multiplier >= 5.0 and hours_since <= 3:
            return AlertUrgency.CRITICAL
        if multiplier >= 3.0 and hours_since <= 6:
            return AlertUrgency.HIGH
        if multiplier >= self.spike_threshold:
            if acceleration > 0:
                return AlertUrgency.HIGH
            return AlertUrgency.MEDIUM
        return AlertUrgency.LOW

    def _calculate_confidence(
        self, snapshot_count: int, multiplier: float, hours_since: float
    ) -> float:
        """
        Higher confidence when we have more data points and the signal is strong.
        Low confidence for brand-new posts with few snapshots.
        """
        data_confidence = min(snapshot_count / 5, 1.0)
        signal_strength = min((multiplier - 1) / 5, 1.0)
        time_confidence = min(hours_since / 3, 1.0)
        return data_confidence * 0.4 + signal_strength * 0.4 + time_confidence * 0.2
