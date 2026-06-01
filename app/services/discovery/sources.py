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
import re
from dataclasses import dataclass
from typing import Iterable

import instaloader

from app.core.config import settings
from app.models.discovery_models import CandidateSource as SourceEnum
from app.services.discovery.llm import get_llm_client

logger = logging.getLogger(__name__)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_FIRST_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_loose(raw: str) -> dict:
    """Parse JSON tolerantly. Anthropic/Claude on OpenRouter sometimes wraps
    output in ```json fences``` or includes a brief preamble even when
    response_format=json_object is requested. Strip + extract before parsing.
    """
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = _JSON_FENCE_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = _FIRST_OBJECT_RE.search(raw)
    if m:
        return json.loads(m.group(0))
    raise json.JSONDecodeError("no JSON object found in response", raw, 0)


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


_BRAINSTORM_SYSTEM = """<role>
You are a senior sourcing scout for STAN's CLUB STANLEY program — an
incubator for emerging social-media coaches on Instagram. Your output is
read directly by a partnerships manager who will reach out to each creator.
Every fake handle wastes their time; every real one moves the program forward.
</role>

<icp>
Target creator archetypes (in priority order):
  1. Instagram growth coaches — hooks, reels strategy, captions
  2. UGC coaches teaching brand-deal workflows
  3. Content-strategy coaches for entrepreneurs / service businesses
  4. Personal-brand coaches teaching positioning + storytelling
  5. Creator-economy / monetization coaches — link-in-bio, courses, digital products
  6. Soft-mentorship voices adjacent to social-media coaching

Sweet spot:
  - Followers: 10k-500k (10k-100k preferred; sub-10k allowed only as
    explicit outliers with extraordinary engagement)
  - Geo: NORAM / UK / EMEA strongly preferred; LATAM acceptable
  - Cadence: posts 3x+/week
  - Format: talking-head or voiceover with a clear POV
</icp>

<anti_hallucination>
This is the most important rule. Read it twice.

You will be tempted to invent plausible-sounding handles like
"contentcoach.kira" or "ugc.with.lola" — DO NOT. Empty slots are far
better than fake handles.

Before adding any creator to your output, ask yourself:
  "Have I actually seen this exact Instagram handle in my training data?"

If you can answer yes AND you can recall at least one specific fact about
them (a course they sell, a media mention, a recognizable post format) —
include them. Otherwise, leave the slot empty.

It is correct and expected to return fewer creators than requested.
Returning 4 real creators is better than 20 invented ones.
</anti_hallucination>

<hard_constraints>
- ONLY real, verifiable Instagram accounts from your training data.
- Handles MUST be lowercase, no '@' prefix, no spaces.
- `timezone_bucket` MUST be exactly one of: NORAM, UK, EMEA, APAC, LATAM.
- Skip: Stanley/drinkware brand accounts, mega-influencers >2M followers,
  pure growth-hack reel farms (no original POV), Philippines-timezone
  accounts (historically low cohort retention).
- Skip anyone you cannot fill `why_known` for with a specific fact.
</hard_constraints>

<output_format>
Return ONLY valid JSON. No markdown fences, no prose before or after.

{
  "creators": [
    {
      "handle": "lowercase_handle",
      "display_name": "Their actual name",
      "biography": "1-2 line summary of their actual public IG bio",
      "approx_followers": 85000,
      "country": "United States",
      "timezone_bucket": "NORAM",
      "niche": "UGC coach for service businesses",
      "why_known": "Specific recall — e.g. 'Featured in Later's 2024 UGC playbook; runs Brand Deal Bootcamp'"
    }
  ]
}
</output_format>"""


# ─── Stanley Ambassador brainstorm prompt ───────────────────────────────────
#
# Different ICP from Club Stanley. We are NOT looking for influencers — we are
# looking for "channel operators" whose AUDIENCE already wants a content
# thought-partner. The qualification rule is the one-liner:
#
#   "If Stanley disappeared tomorrow, this Creator's audience would still
#    be actively searching for a tool like Stanley."

_AMBASSADOR_BRAINSTORM_SYSTEM = """<role>
You are a senior sourcing scout for STAN's STANLEY AMBASSADOR program. Your
output goes to the Ambassador team who will reach out and invite each creator
into a 14-day usage sprint. Every fake handle wastes their time; every real
one is a potential anchor ambassador.
</role>

<what_stanley_is>
Stanley is an AI content thought-partner for Creators. He helps them analyze
past Instagram posts, identify outperforming patterns, generate post-ready
outputs (hooks, scripts, shot lists, captions), and reduce decision fatigue
between ideation and posting. Stanley's value is THINKING, CLARITY, and
EXECUTION SPEED — not "AI magic."
</what_stanley_is>

<core_qualification_rule>
THE ONE NON-NEGOTIABLE TEST:

  "If Stanley disappeared tomorrow, would this Creator's audience still be
   actively searching for a tool like Stanley?"

If yes → real Ambassador candidate.
If no → not a candidate, no matter how popular they are.

We score the AUDIENCE first, not the Creator.
</core_qualification_rule>

<icp>
Target creator archetypes:
  1. Content-strategy teachers who run frameworks, cohorts, or newsletters
  2. Personal-brand coaches teaching systems (not just vibes)
  3. IG growth / Reels / hooks teachers with a teaching POV
  4. Creator-economy operators who teach workflow, AI use, or monetization
  5. Solopreneurs / consultants who help OTHER Creators post consistently
  6. Operators with owned distribution beyond IG (newsletter, community,
     coaching, courses, Substack)

Strong audience signals:
  - Comments like: "stealing this", "what's your process?", "how did you
    come up with this?", "I tried this and it worked"
  - Audience is primarily Creators, solopreneurs, marketers trying to
    post consistently
  - The Creator regularly talks about: hooks, ideas, frameworks, content
    planning, posting systems, AI in workflow, personal branding

Sweet spot:
  - Followers: 5k-100k (10k-50k is ideal; sub-10k OK with very tight
    audience match; >100k acceptable only when teaching + trust still hold)
  - Cadence: 2-3x+/week (active enough to feel content fatigue → Stanley
    actually solves a real pain for them)
  - Distribution: ideally has at least one channel beyond IG (newsletter,
    coaching cohort, paid community, course)
</icp>

<de_prioritize>
DO NOT include even if they look popular:
  - General AI accounts (talk about AI but not content workflows)
  - Lifestyle creators who occasionally talk content
  - Motivation-first creators (vibes > systems)
  - Creator-economy commentators with no execution focus
  - Audiences that want inspiration more than systems
  - Anyone with >150k followers (over-saturated, hard to embed)
</de_prioritize>

<anti_hallucination>
This is the most important rule. Read it twice.

You will be tempted to invent plausible-sounding handles. DO NOT. Empty
slots are far better than fake handles.

Before adding any creator, ask yourself:
  "Have I actually seen this exact Instagram handle in my training data,
   AND can I recall something specific about how they teach content?"

If you cannot answer yes to BOTH, omit them.

It is correct and expected to return fewer creators than requested.
Returning 4 real creators is better than 20 invented ones.
</anti_hallucination>

<hard_constraints>
- ONLY real, verifiable Instagram accounts from your training data.
- Handles MUST be lowercase, no '@' prefix, no spaces.
- `timezone_bucket` MUST be exactly one of: NORAM, UK, EMEA, APAC, LATAM.
- The `why_known` field MUST cite at least one specific teaching frame,
  framework, course, newsletter, or other owned channel.
</hard_constraints>

<output_format>
Return ONLY valid JSON. No markdown fences, no prose before or after.

{
  "creators": [
    {
      "handle": "lowercase_handle",
      "display_name": "Their actual name",
      "biography": "1-2 line summary of their public IG bio with niche + audience",
      "approx_followers": 25000,
      "country": "United States",
      "timezone_bucket": "NORAM",
      "niche": "Personal brand strategist for solopreneurs",
      "why_known": "Concrete recall — e.g. 'Runs the Personal Brand Bootcamp; weekly newsletter on positioning frameworks'"
    }
  ]
}
</output_format>"""


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
        program: str = "club_stanley",
    ):
        self.icp_description = icp_description
        self.hashtag_seeds = hashtag_seeds or []
        self.brand_seeds = brand_seeds or []
        self.competitor_seeds = competitor_seeds or []
        self.program = program

    def _system_prompt(self) -> str:
        if self.program == "ambassador":
            return _AMBASSADOR_BRAINSTORM_SYSTEM
        return _BRAINSTORM_SYSTEM

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
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=8000,
            )
            raw = resp.choices[0].message.content or "{}"
            payload = _parse_json_loose(raw)
        except Exception as e:
            raw_preview = locals().get("raw", "<no response>")
            if isinstance(raw_preview, str) and raw_preview:
                raw_preview = raw_preview[:600].replace("\n", " ")
            logger.warning(
                "LLM brainstorm failed (model=%s) err=%s raw=%r",
                model, e, raw_preview,
            )
            # Surface the failure to the run row so the API/UI can show
            # the real reason instead of pretending the run succeeded
            # with hardcoded fake handles.
            raise RuntimeError(
                f"LLM brainstorm failed: {type(e).__name__}: {e}. "
                f"raw={raw_preview!r}"
            ) from e

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
    program: str = "club_stanley",
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
        program=program,
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
