import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select

from app.core.config import settings
from app.core.database import init_db, async_session
from app.api.users import router as users_router
from app.api.creators import router as creators_router
from app.api.alerts import router as alerts_router
from app.api.timing import router as timing_router
from app.api.discover import router as discover_router
from app.services.scanner import run_velocity_scan
from app.models.models import User
# Ensure all model modules are registered on Base.metadata before init_db runs.
from app.models import timing_models  # noqa: F401
from app.models import discovery_models  # noqa: F401
from app.services.timing_service import (
    seed_demo_timing_data,
    recompute_niche_hourly_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


async def _ensure_default_user():
    """
    Render's free tier uses an ephemeral filesystem — the SQLite DB is wiped
    on every restart/redeploy. The frontend hard-codes user_id=1 (DISCOVER_USER_ID
    in discoverApi.ts), so we idempotently re-seed that user on boot. Without
    this, the first /discover/run after every restart fails with "User not found".
    """
    async with async_session() as db:
        res = await db.execute(select(User).where(User.id == 1))
        if res.scalar_one_or_none():
            return
        db.add(
            User(
                id=1,
                username="stanley",
                instagram_handle="clubstanley",
                niche_tags=["social-media-coach"],
                notification_enabled=True,
            )
        )
        await db.commit()
        logger.info("Seeded default user id=1 (stanley)")


async def _seed_and_recompute_timing():
    async with async_session() as db:
        await seed_demo_timing_data(db)
        await recompute_niche_hourly_stats(db)


async def _scheduled_recompute_timing():
    async with async_session() as db:
        await recompute_niche_hourly_stats(db)


async def _background_seed_timing():
    """Fire-and-forget timing seed so the app starts accepting traffic
    immediately — important for Railway/Render healthchecks that bail
    after ~60–120s if /health isn't responding."""
    try:
        await _seed_and_recompute_timing()
        logger.info("Timing data seeded & stats computed (background)")
    except Exception:
        logger.exception("Background timing seed failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Critical, must-finish-before-traffic work only.
    await init_db()
    logger.info("Database initialized")

    await _ensure_default_user()

    # Heavy seed runs in the background so /health responds instantly.
    asyncio.create_task(_background_seed_timing())

    scheduler.add_job(
        run_velocity_scan,
        "interval",
        minutes=settings.polling_interval_minutes,
        id="velocity_scan",
        name="Velocity Scan",
        replace_existing=True,
    )
    scheduler.add_job(
        _scheduled_recompute_timing,
        "interval",
        hours=1,
        id="timing_stats_recompute",
        name="Timing Stats Recompute",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        f"Velocity scanner started (every {settings.polling_interval_minutes} min)"
    )

    yield

    scheduler.shutdown()
    logger.info("Scheduler stopped")


app = FastAPI(
    title="Stanley Velocity Alerts",
    description=(
        "Detects algorithmic trend spikes from competitor creators and pushes "
        "high-urgency, pre-drafted notifications before the wave peaks."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users_router, prefix="/api")
app.include_router(creators_router, prefix="/api")
app.include_router(alerts_router, prefix="/api")
app.include_router(timing_router, prefix="/api")
app.include_router(discover_router, prefix="/api")


@app.get("/health")
async def health():
    return {
        "status": "operational",
        "scanner_running": scheduler.running,
        "polling_interval_min": settings.polling_interval_minutes,
        "spike_threshold": settings.velocity_spike_threshold,
    }
