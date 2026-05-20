"""
Instagram data ingestion layer.

Fetches competitor posts and engagement metrics using instaloader.
Falls back to a mock data provider for development/demo without credentials.
"""
import logging
from datetime import datetime, timedelta
from typing import Any

import instaloader
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.models import TrackedCreator, CreatorPost, PostSnapshot

logger = logging.getLogger(__name__)


class InstagramScraper:
    def __init__(self):
        self._loader: instaloader.Instaloader | None = None

    def _get_loader(self) -> instaloader.Instaloader:
        if self._loader is None:
            self._loader = instaloader.Instaloader(
                download_pictures=False,
                download_videos=False,
                download_video_thumbnails=False,
                download_geotags=False,
                download_comments=False,
                save_metadata=False,
                compress_json=False,
            )
            if settings.instagram_session_id:
                try:
                    self._loader.load_session_from_file(
                        "stan_bot", settings.instagram_session_id
                    )
                except Exception:
                    logger.warning("Could not load Instagram session, running without auth")
        return self._loader

    async def fetch_creator_profile(self, handle: str) -> dict[str, Any]:
        """Fetch basic profile info for a creator."""
        try:
            loader = self._get_loader()
            profile = instaloader.Profile.from_username(loader.context, handle)
            return {
                "handle": handle,
                "display_name": profile.full_name,
                "follower_count": profile.followers,
                "following_count": profile.followees,
                "post_count": profile.mediacount,
                "biography": profile.biography,
            }
        except Exception as e:
            logger.error(f"Failed to fetch profile for {handle}: {e}")
            return {"handle": handle, "error": str(e)}

    async def fetch_recent_posts(
        self, handle: str, max_posts: int = 20
    ) -> list[dict[str, Any]]:
        """Fetch recent posts with engagement metrics."""
        posts = []
        try:
            loader = self._get_loader()
            profile = instaloader.Profile.from_username(loader.context, handle)
            for i, post in enumerate(profile.get_posts()):
                if i >= max_posts:
                    break
                posts.append({
                    "post_id": post.shortcode,
                    "post_url": f"https://instagram.com/p/{post.shortcode}/",
                    "caption": post.caption or "",
                    "post_type": self._detect_post_type(post),
                    "posted_at": post.date_utc,
                    "views": post.video_view_count or 0,
                    "likes": post.likes,
                    "comments": post.comments,
                })
        except Exception as e:
            logger.error(f"Failed to fetch posts for {handle}: {e}")
        return posts

    def _detect_post_type(self, post) -> str:
        if post.is_video:
            return "reel"
        if post.typename == "GraphSidecar":
            return "carousel"
        return "image"


class MockInstagramScraper:
    """Development/demo scraper with realistic fake data for testing the pipeline."""

    _mock_creators = {
        "nabeel.ae": {
            "display_name": "Nabeel Ahmed",
            "follower_count": 285000,
            "avg_views": 45000,
        },
        "fashion.startup.daily": {
            "display_name": "Fashion Startup Daily",
            "follower_count": 132000,
            "avg_views": 28000,
        },
        "cs.to.ceo": {
            "display_name": "CS to CEO",
            "follower_count": 98000,
            "avg_views": 18000,
        },
    }

    async def fetch_creator_profile(self, handle: str) -> dict[str, Any]:
        mock = self._mock_creators.get(handle, {
            "display_name": handle.replace(".", " ").title(),
            "follower_count": 50000,
            "avg_views": 12000,
        })
        return {
            "handle": handle,
            "display_name": mock["display_name"],
            "follower_count": mock["follower_count"],
            "following_count": 800,
            "post_count": 340,
            "biography": f"Content creator | {handle}",
        }

    async def fetch_recent_posts(
        self, handle: str, max_posts: int = 20
    ) -> list[dict[str, Any]]:
        import random
        mock = self._mock_creators.get(handle, {"avg_views": 12000})
        avg = mock["avg_views"]

        posts = []
        now = datetime.utcnow()
        formats = [
            ("FOMO listicle", "hook_question"),
            ("storytime", "hook_cliffhanger"),
            ("hot take", "hook_controversial"),
            ("tutorial", "hook_promise"),
            ("day in the life", "hook_relatable"),
            ("comparison", "hook_versus"),
        ]
        for i in range(max_posts):
            hours_ago = random.uniform(1, 720)
            is_viral = i == 0 and random.random() > 0.3
            view_mult = random.uniform(3.0, 8.0) if is_viral else random.uniform(0.4, 1.8)
            views = int(avg * view_mult)
            fmt = random.choice(formats)
            posts.append({
                "post_id": f"{handle}_{i}_{int(now.timestamp())}",
                "post_url": f"https://instagram.com/p/mock_{handle}_{i}/",
                "caption": self._generate_mock_caption(handle, fmt[0]),
                "post_type": "reel" if random.random() > 0.2 else "carousel",
                "posted_at": now - timedelta(hours=hours_ago),
                "views": views,
                "likes": int(views * random.uniform(0.03, 0.08)),
                "comments": int(views * random.uniform(0.002, 0.01)),
                "detected_format": fmt[0],
                "detected_hook_type": fmt[1],
            })
        return sorted(posts, key=lambda p: p["posted_at"], reverse=True)

    def _generate_mock_caption(self, handle: str, fmt: str) -> str:
        captions = {
            "FOMO listicle": f"5 things nobody tells you about building a brand in 2026. Number 3 changed everything for me.",
            "storytime": f"I almost quit last month. Here's what happened next...",
            "hot take": f"Unpopular opinion: 90% of content advice is keeping you mediocre.",
            "tutorial": f"Step by step: How I grew from 0 to 100k in 6 months (save this).",
            "day in the life": f"A day in my life running a startup and creating content.",
            "comparison": f"Creator A vs Creator B: Why one hit 1M and the other didn't.",
        }
        return captions.get(fmt, "New post")


class FallbackScraper:
    """Tries live Instagram first, falls back to mock on any failure."""

    def __init__(self):
        self._live = InstagramScraper()
        self._mock = MockInstagramScraper()

    async def fetch_creator_profile(self, handle: str) -> dict[str, Any]:
        try:
            result = await self._live.fetch_creator_profile(handle)
            if "error" not in result:
                return result
        except Exception:
            pass
        logger.info(f"Live profile fetch failed for {handle}, using mock data")
        return await self._mock.fetch_creator_profile(handle)

    async def fetch_recent_posts(
        self, handle: str, max_posts: int = 20
    ) -> list[dict[str, Any]]:
        try:
            posts = await self._live.fetch_recent_posts(handle, max_posts)
            if posts:
                return posts
        except Exception:
            pass
        logger.info(f"Live post fetch failed for {handle}, using mock data")
        return await self._mock.fetch_recent_posts(handle, max_posts)


def get_scraper() -> FallbackScraper | MockInstagramScraper:
    if settings.instagram_session_id:
        return FallbackScraper()
    logger.info("No Instagram session configured, using mock scraper for demo")
    return MockInstagramScraper()


async def ingest_creator_posts(
    db: AsyncSession, creator: TrackedCreator
) -> list[CreatorPost]:
    """Fetch and store/update posts for a tracked creator."""
    scraper = get_scraper()
    raw_posts = await scraper.fetch_recent_posts(
        creator.instagram_handle, max_posts=settings.baseline_post_count
    )

    ingested = []
    for raw in raw_posts:
        existing = await db.execute(
            select(CreatorPost).where(
                CreatorPost.instagram_post_id == raw["post_id"]
            )
        )
        post = existing.scalar_one_or_none()

        if post is None:
            post = CreatorPost(
                creator_id=creator.id,
                instagram_post_id=raw["post_id"],
                post_url=raw.get("post_url"),
                caption=raw.get("caption"),
                post_type=raw.get("post_type"),
                posted_at=raw.get("posted_at"),
                views=raw.get("views", 0),
                likes=raw.get("likes", 0),
                comments=raw.get("comments", 0),
                detected_format=raw.get("detected_format"),
                detected_hook_type=raw.get("detected_hook_type"),
                content_analysis=raw.get("content_analysis"),
            )
            db.add(post)
        else:
            post.views = raw.get("views", post.views)
            post.likes = raw.get("likes", post.likes)
            post.comments = raw.get("comments", post.comments)
            post.last_updated_at = datetime.utcnow()

        snapshot = PostSnapshot(
            post_id=post.id if post.id else None,
            views=raw.get("views", 0),
            likes=raw.get("likes", 0),
            comments=raw.get("comments", 0),
        )

        await db.flush()
        if snapshot.post_id is None:
            snapshot.post_id = post.id
        db.add(snapshot)
        ingested.append(post)

    # Recalculate creator averages
    avg_result = await db.execute(
        select(
            func.avg(CreatorPost.views),
            func.avg(CreatorPost.likes),
            func.avg(CreatorPost.comments),
        ).where(CreatorPost.creator_id == creator.id)
    )
    avgs = avg_result.one()
    creator.avg_views = avgs[0] or 0
    creator.avg_likes = avgs[1] or 0
    creator.avg_comments = avgs[2] or 0
    creator.last_scraped_at = datetime.utcnow()

    await db.commit()
    return ingested
