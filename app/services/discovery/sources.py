"""
Candidate sources — pluggable producers of (handle, source, seed) tuples.

Each source is independent and degrades gracefully:
  • If Instagram blocks us, we keep the LLM brainstorm.
  • If OpenAI is missing, we keep the scrapers.
  • If everything fails, MockSource keeps the demo alive.

Add a new source by subclassing CandidateSource and listing it in
`build_default_sources()`. The runner will fan-out across all enabled sources
in parallel and dedupe by handle.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from typing import Iterable

import instaloader

from app.core.config import settings
from app.models.discovery_models import CandidateSource as SourceEnum
from app.services.discovery.llm import get_llm_client

logger = logging.getLogger(__name__)


# ─── Data plumbing ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RawCandidate:
    handle: str
    source: SourceEnum
    seed: str | None  # e.g. the hashtag, brand handle, or LLM prompt id
    # Optional pre-enriched payload from an LLM that knows the creator. When
    # present, the runner skips the (mock) scraper and uses these fields
    # directly, so we get real bios / display names / follower estimates for
    # public creators the model has training-data knowledge of.
    enrichment: dict | None = None

    def normalized(self) -> "RawCandidate":
        h = self.handle.strip().lstrip("@").lower()
        return RawCandidate(
            handle=h,
            source=self.source,
            seed=self.seed,
            enrichment=self.enrichment,
        )


class CandidateSourceBase:
    name: str = "base"

    async def fetch(self, *, limit: int) -> list[RawCandidate]:  # pragma: no cover
        raise NotImplementedError


# ─── 1. Hashtag scraping (instaloader) ──────────────────────────────────────


class HashtagSource(CandidateSourceBase):
    """Pulls the author handle from the most recent posts under each tag."""

    name = "hashtag"

    def __init__(self, hashtags: list[str]):
        self.hashtags = [h.strip().lstrip("#").lower() for h in hashtags if h]

    async def fetch(self, *, limit: int) -> list[RawCandidate]:
        if not self.hashtags:
            return []
        # Instaloader is sync; offload each tag to a thread and gather.
        per_tag = max(1, limit // len(self.hashtags))
        results = await asyncio.gather(
            *(self._fetch_tag(tag, per_tag) for tag in self.hashtags),
            return_exceptions=True,
        )
        out: list[RawCandidate] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("hashtag fetch failed: %s", r)
                continue
            out.extend(r)
        return out

    async def _fetch_tag(self, tag: str, limit: int) -> list[RawCandidate]:
        def _sync():
            loader = _get_loader()
            try:
                ht = instaloader.Hashtag.from_name(loader.context, tag)
            except Exception as e:
                logger.info("hashtag %s unavailable: %s", tag, e)
                return []
            handles: list[str] = []
            try:
                for i, post in enumerate(ht.get_posts()):
                    if i >= limit:
                        break
                    if post.owner_username:
                        handles.append(post.owner_username)
            except Exception as e:
                logger.info("hashtag iteration stopped on %s: %s", tag, e)
            return handles

        try:
            handles = await asyncio.to_thread(_sync)
        except Exception as e:
            logger.warning("hashtag thread failed for %s: %s", tag, e)
            return []
        return [RawCandidate(handle=h, source=SourceEnum.HASHTAG, seed=tag) for h in handles]


# ─── 2. Brand-mention scraping ──────────────────────────────────────────────


class BrandMentionSource(CandidateSourceBase):
    """
    Pulls the owners of posts tagged into competitor / partner brand accounts.
    These are creators the brand already has a relationship-shaped signal with.
    """

    name = "brand_mention"

    def __init__(self, brand_handles: list[str]):
        self.brand_handles = [h.strip().lstrip("@").lower() for h in brand_handles if h]

    async def fetch(self, *, limit: int) -> list[RawCandidate]:
        if not self.brand_handles:
            return []
        per_brand = max(1, limit // len(self.brand_handles))
        results = await asyncio.gather(
            *(self._fetch_brand(b, per_brand) for b in self.brand_handles),
            return_exceptions=True,
        )
        out: list[RawCandidate] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("brand-mention fetch failed: %s", r)
                continue
            out.extend(r)
        return out

    async def _fetch_brand(self, brand: str, limit: int) -> list[RawCandidate]:
        def _sync():
            loader = _get_loader()
            try:
                profile = instaloader.Profile.from_username(loader.context, brand)
            except Exception as e:
                logger.info("brand %s unavailable: %s", brand, e)
                return []
            handles: list[str] = []
            try:
                for i, post in enumerate(profile.get_tagged_posts()):
                    if i >= limit:
                        break
                    if post.owner_username and post.owner_username.lower() != brand:
                        handles.append(post.owner_username)
            except Exception as e:
                logger.info("brand iteration stopped on %s: %s", brand, e)
            return handles

        try:
            handles = await asyncio.to_thread(_sync)
        except Exception as e:
            logger.warning("brand thread failed for %s: %s", brand, e)
            return []
        return [
            RawCandidate(handle=h, source=SourceEnum.BRAND_MENTION, seed=brand)
            for h in handles
        ]


# ─── 3. LLM brainstorm ──────────────────────────────────────────────────────


_BRAINSTORM_SYSTEM = """You are a sourcing scout for STAN's CLUB STANLEY
program — an incubator for EMERGING SOCIAL-MEDIA COACHES on Instagram.

YOUR JOB: return REAL, well-known public Instagram accounts you have
training-data knowledge of, with their actual public metadata filled in.

Target Creator archetypes:
  • Instagram growth coaches (hooks, reels strategy, captions)
  • UGC creators / UGC coaches
  • Content-strategy coaches for entrepreneurs
  • Personal-brand coaches
  • Creator-economy / monetization coaches (link-in-bio, courses, digital products)
  • Soft-mentorship voices in the social-media-coaching niche

Sweet spot: 10k–500k followers, NORAM / UK / EMEA, posting 3x+/week,
talking-head or voiceover content with a clear POV.

HARD RULES (read carefully):
- ONLY suggest accounts you actually have training-data knowledge of.
- If you're not confident the handle exists, OMIT it. Do NOT invent handles.
- Fill in `display_name`, `biography`, `approx_followers`, `country`,
  `timezone_bucket`, `niche`, and `why_known` for every creator you return.
- `timezone_bucket` MUST be one of: "NORAM", "UK", "EMEA", "APAC", "LATAM".
- Avoid: Stanley/drinkware accounts, mega-influencers >2M followers, generic
  "growth-hack reel" farms, and Philippines-timezone accounts (low cohort fit).

Output ONLY valid JSON:
{
  "creators": [
    {
      "handle": "lowercase_handle",
      "display_name": "Their actual name",
      "biography": "Short 1–2 line summary of their public IG bio",
      "approx_followers": 85000,
      "country": "United States",
      "timezone_bucket": "NORAM",
      "niche": "UGC coach for service businesses",
      "why_known": "Featured in Later's UGC playbook; runs UGC bootcamp"
    }
  ]
}

Handles must be lowercase, no '@' prefix, no spaces.
"""


class LLMBrainstormSource(CandidateSourceBase):
    """Cold-start creator handles from GPT given the ICP + seeds.

    Cheap, no rate limits, and survives when scraping is blocked. The runner
    verifies each suggestion against IG (or the mock scraper) before scoring,
    so hallucinated handles just get dropped at hydration time.
    """

    name = "llm_brainstorm"

    def __init__(
        self,
        *,
        icp_description: str,
        hashtag_seeds: list[str] | None = None,
        brand_seeds: list[str] | None = None,
        competitor_seeds: list[str] | None = None,
    ):
        self.icp_description = icp_description
        self.hashtag_seeds = hashtag_seeds or []
        self.brand_seeds = brand_seeds or []
        self.competitor_seeds = competitor_seeds or []

    async def fetch(self, *, limit: int) -> list[RawCandidate]:
        client, model = get_llm_client()
        if client is None or model is None:
            return self._fallback(limit)

        user_msg = json.dumps(
            {
                "icp": self.icp_description,
                "hashtag_seeds": self.hashtag_seeds,
                "brand_seeds": self.brand_seeds,
                "competitor_seeds": self.competitor_seeds,
                "count_requested": limit,
                "instructions": (
                    f"Return up to {limit} real, well-known public Instagram "
                    "coaches that you have training-data knowledge of. Include "
                    "the metadata fields described in the system prompt for "
                    "each one. Skip any creator you're not confident exists."
                ),
            },
            indent=2,
        )
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _BRAINSTORM_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=2200,
            )
            payload = json.loads(resp.choices[0].message.content or "{}")
        except Exception as e:
            logger.warning("LLM brainstorm failed, falling back: %s", e)
            return self._fallback(limit)

        # Newer prompt format returns {"creators": [{handle, display_name, ...}]}.
        # Tolerate the legacy {"handles": [...]} shape too.
        creators = payload.get("creators")
        if not creators:
            handles = payload.get("handles") or []
            return [
                RawCandidate(handle=str(h), source=SourceEnum.LLM_BRAINSTORM, seed=model)
                for h in handles[:limit]
            ]

        out: list[RawCandidate] = []
        for c in creators[:limit]:
            handle = (c.get("handle") or "").strip().lstrip("@").lower()
            if not handle:
                continue
            out.append(
                RawCandidate(
                    handle=handle,
                    source=SourceEnum.LLM_BRAINSTORM,
                    seed=model,
                    enrichment={
                        "display_name": c.get("display_name"),
                        "biography": c.get("biography"),
                        "approx_followers": c.get("approx_followers"),
                        "country": c.get("country"),
                        "timezone_bucket": c.get("timezone_bucket"),
                        "niche": c.get("niche"),
                        "why_known": c.get("why_known"),
                    },
                )
            )
        return out

    def _fallback(self, limit: int) -> list[RawCandidate]:
        # Plausible social-media-coach handle shapes for offline / no-key runs.
        # Hydrator verifies each one, so anything fake just gets dropped.
        bench = [
            "growwith.maya", "thereelsstrategist", "ugc.with.lola",
            "contentcoach.kira", "ig.growth.guy", "hookwriter.daily",
            "creator.economy.coach", "soheila.scales", "ugc.school.zara",
            "captions.that.convert", "buildyourbrand.with.sam",
            "personalbrand.playbook", "reels.lab.uk", "monetize.ur.content",
            "lina.teaches.ig", "thecontentstudio.co", "ugc.tips.daily",
            "smallbiz.contentcoach", "growthhacks.alex", "tia.makes.reels",
            "the.algorithm.translator", "story.driven.creator",
        ]
        random.shuffle(bench)
        picked = bench[:limit]
        return [
            RawCandidate(handle=h, source=SourceEnum.LLM_BRAINSTORM, seed="fallback")
            for h in picked
        ]


# ─── 4. Mock source (always works) ──────────────────────────────────────────


class MockSource(CandidateSourceBase):
    """Deterministic-ish offline source so demos never go empty."""

    name = "mock"

    # Plausible social-media-coach handles + the seed they would have come from.
    _bench = [
        ("thereelsstrategist", "hashtag", "reelsstrategy"),
        ("ugc.with.lola", "hashtag", "ugccoach"),
        ("hookwriter.daily", "hashtag", "contentstrategy"),
        ("growwith.maya", "brand_mention", "stansolo"),
        ("captions.that.convert", "brand_mention", "later.com"),
        ("ig.growth.guy", "llm_brainstorm", "gpt-fallback"),
        ("monetize.ur.content", "hashtag", "creatoreconomy"),
        ("personalbrand.playbook", "brand_mention", "kajabi"),
    ]

    async def fetch(self, *, limit: int) -> list[RawCandidate]:
        out: list[RawCandidate] = []
        for handle, src, seed in self._bench[:limit]:
            out.append(
                RawCandidate(
                    handle=handle,
                    source=SourceEnum(src),
                    seed=seed,
                )
            )
        return out


# ─── Loader cache + factory ─────────────────────────────────────────────────


_LOADER: instaloader.Instaloader | None = None


def _get_loader() -> instaloader.Instaloader:
    global _LOADER
    if _LOADER is None:
        _LOADER = instaloader.Instaloader(
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
                _LOADER.load_session_from_file(
                    "stan_bot", settings.instagram_session_id
                )
            except Exception:
                logger.warning("Could not load IG session for discovery loader")
    return _LOADER


def build_default_sources(
    *,
    icp_description: str,
    hashtag_seeds: list[str] | None,
    brand_account_seeds: list[str] | None,
    competitor_handle_seeds: list[str] | None,
    use_scrapers: bool = True,
) -> list[CandidateSourceBase]:
    """Default Club Stanley source fan-out. Always includes a usable backup."""
    sources: list[CandidateSourceBase] = []

    if use_scrapers and settings.instagram_session_id:
        if hashtag_seeds:
            sources.append(HashtagSource(hashtag_seeds))
        if brand_account_seeds:
            sources.append(BrandMentionSource(brand_account_seeds))

    llm_source = LLMBrainstormSource(
        icp_description=icp_description,
        hashtag_seeds=hashtag_seeds,
        brand_seeds=brand_account_seeds,
        competitor_seeds=competitor_handle_seeds,
    )
    sources.append(llm_source)

    # MockSource is a last-resort safety net so empty runs never happen. We
    # only attach it when there's no LLM client AND no Instagram session — i.e.
    # the LLM brainstorm can't actually generate anything real either.
    # Otherwise it just pollutes the candidate list with template "Content
    # creator | <handle>" bios.
    llm_client, _ = get_llm_client()
    if not settings.instagram_session_id and llm_client is None:
        sources.append(MockSource())

    return sources


async def collect_candidates(
    sources: Iterable[CandidateSourceBase], *, per_source_limit: int
) -> list[RawCandidate]:
    """Fan-out across all sources, gather, normalize, and dedupe by handle."""
    results = await asyncio.gather(
        *(s.fetch(limit=per_source_limit) for s in sources),
        return_exceptions=True,
    )
    raw: list[RawCandidate] = []
    for s, r in zip(sources, results):
        if isinstance(r, Exception):
            logger.warning("source %s errored: %s", s.name, r)
            continue
        raw.extend(r)

    # Dedupe by normalized handle, keep first-seen source for sourcing trail.
    seen: dict[str, RawCandidate] = {}
    for c in raw:
        n = c.normalized()
        if n.handle and n.handle not in seen:
            seen[n.handle] = n
    return list(seen.values())
