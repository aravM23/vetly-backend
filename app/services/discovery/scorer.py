"""
Candidate scorer — Club Stanley sourcing rubric.

The Club Stanley Sourcing Guide defines the green / red flags an internal
sourcer would use when evaluating a Creator for the cohort. This module turns
that rubric into a structured LLM prompt + deterministic anchors so each
Creator is scored against the same checklist every time.

Rubric → score axes (each 0-100):

  fit (40%)        Niche fit: are they a SOCIAL-MEDIA COACH (or close)?
  engagement (25%) Engagement quality + comment quality (not just view count)
  audience (20%)   Audience size (10k-100k sweet spot) + geo fit (UK/NORAM ≥ APAC)
  recency (15%)    Posting consistency: 3x+/week green, ≤1x/week red

Extra LLM signals (0-100), surfaced independently for sourcer sanity-checking:
  talking_head     Talking-head/voiceover vs growth-reel dominance
  bio_quality      Niche clarity + proof points + CTA
  comment_quality  Real conversation vs comment-pod hype
  country_guess    Best guess (e.g. "United Kingdom") for the geo column
  timezone_bucket  One of: NORAM | UK | EMEA | APAC | PHILIPPINES | UNKNOWN
  green_flags / red_flags  Explicit short strings ("posts daily", "growth reels dominant", ...)
  is_outlier       True for Mehr-Rajput-style cases (small but tapped-in)

Falls back to a deterministic heuristic when no OpenAI key is configured.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from app.services.discovery.llm import get_llm_client, has_llm

logger = logging.getLogger(__name__)


SCORER_BATCH_SIZE = 5

WEIGHTS = {"fit": 0.40, "engagement": 0.25, "audience": 0.20, "recency": 0.15}


@dataclass
class ScoredCandidate:
    handle: str
    fit: int
    engagement: int
    audience: int
    recency: int
    overall: int
    reasoning: str
    talking_head: int | None = None
    bio_quality: int | None = None
    comment_quality: int | None = None
    country_guess: str | None = None
    timezone_bucket: str | None = None
    green_flags: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    is_outlier: bool = False


SYSTEM_PROMPT = """<role>
You are the sourcing brain for STAN's CLUB STANLEY program — an incubator
for emerging social-media coaches on Instagram. A partnerships manager
reads your scores and reasoning before reaching out, so accuracy and
honest signal matter more than confident-sounding numbers.
</role>

<icp>
Niche: SOCIAL-MEDIA COACHES — people who teach IG growth, content strategy,
UGC, hooks/storytelling, monetization, creator-economy tactics. Adjacent
niches (UGC how-to, personal-brand coaching, content-business coaching) are
acceptable case-by-case when other signals are strong.

Followers: 10k-100k is the sweet spot. Sub-10k is ALLOWED as an OUTLIER
when the audience is unusually tapped-in (a Mehr-Rajput case: tiny
following, high conversion, comments full of "I tried this and it worked").
</icp>

<rubric>
Score each axis 0-100. Use the full range — don't cluster everything at 70.

fit (40% weight)
  90+  Clearly a social-media coach with a sharp POV and visible teaching frame
  70   Coaches social-media-adjacent things (e.g. personal brand, UGC how-to)
  50   Lifestyle creator who occasionally talks about IG tactics
  <30  Off-niche entirely

engagement (25% weight)
  Quality, not volume. Reward:
    - Real conversation in comments (replies, questions, "I tried this")
    - Creator replies to commenters
    - Healthy like:comment ratio (<150 typical for coaching content)
  Penalize:
    - 7k-10k views with only 4-10 low-effort comments
    - Very high like:comment ratio (>200 = likely comment pod)
    - Generic hype comments ("fire 🔥🔥🔥")

audience (20% weight)
  10k-100k followers in NORAM / UK / EMEA = 80-95
  Same range in APAC / LATAM = 60-75
  Same range in Philippines = 30-50 (historically poor cohort retention)
  Sub-10k with strong fit = 50-65 AND set is_outlier=true
  >100k = 40-55 (likely too established for an emerging-creator program)

recency (15% weight)
  3x+/week    = 85-100
  2x/week     = 60-75
  ~1x/week    = 20-40
  No post in 30+ days = ≤10
</rubric>

<extra_signals>
talking_head    Higher when feed is talking-head/voiceover with clear POV.
                Lower when dominated by short growth reels (7-10s b-roll +
                text overlay) or recycled meme content.
bio_quality     Clear niche + who they help + proof points + clean CTA = high.
                Vague "creator | lifestyle | DM for collabs" = low.
comment_quality Inferred from caption style, like:comment ratio, and any
                visible conversation cues. Honest 50 when unclear.
</extra_signals>

<geo>
country_guess    Best guess from bio + caption + handle conventions. Use the
                 full country name ("United Kingdom", "United States",
                 "Philippines"). Empty string "" if genuinely unclear.
timezone_bucket  Exactly one of: NORAM, UK, EMEA, APAC, PHILIPPINES, UNKNOWN.
</geo>

<flags_guidance>
green_flags and red_flags must cite EVIDENCE that's actually present in
the data you were given — never invent details to support a flag.

Good green_flags: "posts 4x/week", "talking-head with clear POV",
"real conversation in comments", "UK-based per bio".
Good red_flags: "growth-reel dominant", "comment pod suspected
(like:comment 280)", "ad density 60%", "Philippines TZ", "vague bio",
"posts ~1x/week".
</flags_guidance>

<outlier_rule>
Set is_outlier=true ONLY when ALL three are true:
  1. Followers < 10,000
  2. fit ≥ 75
  3. Engagement quality looks strong (real conversation, not pods)
This mirrors the Mehr-Rajput exception. Setting it incorrectly pollutes
the cohort.
</outlier_rule>

<reasoning_style>
1-2 sentences max. Write like a sourcer briefing a partnerships manager —
specific, concrete, no hedging. Cite the strongest green or red flag.
Capitalize "Creator" and "Creators".
</reasoning_style>

<output_format>
Return ONLY valid JSON. No markdown fences. One entry per handle in the input.

{
  "scores": {
    "<handle>": {
      "fit": 0-100,
      "engagement": 0-100,
      "audience": 0-100,
      "recency": 0-100,
      "talking_head": 0-100,
      "bio_quality": 0-100,
      "comment_quality": 0-100,
      "country_guess": "United States",
      "timezone_bucket": "NORAM",
      "green_flags": ["..."],
      "red_flags": ["..."],
      "is_outlier": false,
      "reasoning": "..."
    }
  }
}
</output_format>"""


async def score_candidates(
    candidates: list[dict],
    *,
    icp_description: str,
) -> list[ScoredCandidate]:
    if not candidates:
        return []

    anchors = {c["handle"]: _compute_anchors(c) for c in candidates}

    client, model = get_llm_client()
    if client is None or model is None:
        logger.info("No LLM key — using heuristic Club Stanley scorer for %d Creators", len(candidates))
        return [_heuristic_score(c, anchors[c["handle"]]) for c in candidates]

    batches = [
        candidates[i : i + SCORER_BATCH_SIZE]
        for i in range(0, len(candidates), SCORER_BATCH_SIZE)
    ]
    batch_results = await asyncio.gather(
        *(_score_batch(client, model, b, icp_description) for b in batches),
        return_exceptions=True,
    )

    out: list[ScoredCandidate] = []
    for batch, result in zip(batches, batch_results):
        if isinstance(result, Exception):
            logger.warning("scoring batch failed, falling back to heuristic: %s", result)
            out.extend(_heuristic_score(c, anchors[c["handle"]]) for c in batch)
            continue
        for c in batch:
            llm_scores = result.get(c["handle"]) or result.get(c["handle"].lower())
            if not llm_scores:
                out.append(_heuristic_score(c, anchors[c["handle"]]))
                continue
            out.append(_merge_with_anchors(c, llm_scores, anchors[c["handle"]]))
    return out


async def _score_batch(
    client: AsyncOpenAI, model: str, batch: list[dict], icp_description: str
) -> dict:
    payload = {
        "icp": icp_description,
        "creators": [_serialize_for_prompt(c) for c in batch],
    }
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, default=str)},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=3000,
    )
    parsed = json.loads(resp.choices[0].message.content or "{}")
    return parsed.get("scores", {})


def _serialize_for_prompt(c: dict) -> dict:
    return {
        "handle": c["handle"],
        "bio": (c.get("biography") or "")[:320],
        "followers": c.get("follower_count"),
        "avg_views": c.get("avg_views"),
        "avg_likes": c.get("avg_likes"),
        "avg_comments": c.get("avg_comments"),
        "engagement_rate": round(c.get("engagement_rate") or 0, 4),
        "posts_per_week": c.get("posts_per_week"),
        "like_to_comment_ratio": c.get("like_to_comment_ratio"),
        "ad_density": round(c.get("ad_density") or 0, 3),
        "last_post_at": c.get("last_post_at"),
        "caption_sample": (c.get("recent_post_caption_sample") or "")[:700],
        "discovered_via": c.get("discovered_via"),
        "pre_flagged_outlier": c.get("is_outlier_flagged", False),
        "preferred_geo_tags": c.get("preferred_geo_tags"),
        "deprioritized_geo_tags": c.get("deprioritized_geo_tags"),
        # Hints supplied by the LLM-enrichment source (real, named creators
        # the model has prior knowledge of). Treat these as priors, not facts —
        # the scorer can still adjust them based on the rubric.
        "country_hint": c.get("country_hint"),
        "timezone_hint": c.get("timezone_hint"),
        "niche_hint": c.get("niche_hint"),
        "why_known": c.get("why_known"),
    }


# ─── Deterministic anchors ──────────────────────────────────────────────────


def _compute_anchors(c: dict) -> dict:
    """Engagement + recency scored from raw numbers, no LLM needed."""
    eng_rate = c.get("engagement_rate") or 0
    eng_score = max(0, min(100, int(eng_rate * 2000)))  # 5%+ = 100

    # Penalize unhealthy like:comment ratios (pod signal).
    ltc = c.get("like_to_comment_ratio")
    if ltc is not None:
        if ltc > 300:                # extreme: very few comments per like
            eng_score = max(0, eng_score - 25)
        elif ltc > 150:
            eng_score = max(0, eng_score - 12)

    # Consistency anchor from posts_per_week, with a secondary recency penalty
    # so a Creator who posted 5x last month but nothing in 3 weeks still drops.
    ppw = c.get("posts_per_week")
    if ppw is None:
        cadence_score = 40
    elif ppw >= 5:
        cadence_score = 100
    elif ppw >= 3:
        cadence_score = 85
    elif ppw >= 2:
        cadence_score = 65
    elif ppw >= 1:
        cadence_score = 35
    else:
        cadence_score = 10

    last_post = c.get("last_post_at")
    if isinstance(last_post, str):
        try:
            last_post = datetime.fromisoformat(last_post.replace("Z", "+00:00"))
        except Exception:
            last_post = None
    if last_post is not None:
        if last_post.tzinfo is None:
            last_post = last_post.replace(tzinfo=timezone.utc)
        days = max(0, (datetime.now(timezone.utc) - last_post).days)
        if days > 30:
            cadence_score = min(cadence_score, 15)
        elif days > 14:
            cadence_score = min(cadence_score, 50)

    return {"engagement": eng_score, "recency": cadence_score}


def _heuristic_score(c: dict, anchors: dict) -> ScoredCandidate:
    """No-LLM fallback. Honest neutral fit, real cadence + engagement signals."""
    followers = c.get("follower_count") or 0
    if 10_000 <= followers <= 100_000:
        audience = 80
    elif 5_000 <= followers < 10_000:
        audience = 55  # outlier zone
    elif followers > 100_000:
        audience = 40  # too established for emerging-Creator program
    elif followers > 0:
        audience = 30
    else:
        audience = 0

    fit = 50  # Cannot judge niche fit without LLM — neutral default.
    reasoning = (
        "Heuristic score — no LLM available. Fit and audience are approximate; "
        "engagement and consistency are computed from observed post stats."
    )

    overall = int(
        WEIGHTS["fit"] * fit
        + WEIGHTS["engagement"] * anchors["engagement"]
        + WEIGHTS["audience"] * audience
        + WEIGHTS["recency"] * anchors["recency"]
    )

    green: list[str] = []
    red: list[str] = []
    ppw = c.get("posts_per_week")
    if ppw is not None:
        if ppw >= 3:
            green.append(f"posts {ppw}x/week")
        elif ppw <= 1:
            red.append(f"low cadence ({ppw}x/week)")
    if (c.get("ad_density") or 0) >= 0.4:
        red.append("ad-heavy caption sample")
    ltc = c.get("like_to_comment_ratio")
    if ltc is not None and ltc > 200:
        red.append("very high like:comment ratio")

    return ScoredCandidate(
        handle=c["handle"],
        fit=fit,
        engagement=anchors["engagement"],
        audience=audience,
        recency=anchors["recency"],
        overall=overall,
        reasoning=reasoning,
        green_flags=green,
        red_flags=red,
        is_outlier=bool(c.get("is_outlier_flagged", False)),
    )


def _merge_with_anchors(c: dict, llm: dict, anchors: dict) -> ScoredCandidate:
    """Trust LLM for niche / audience / signals, deterministic for engagement+cadence."""
    fit = _clamp_int(llm.get("fit"), default=50)
    audience = _clamp_int(llm.get("audience"), default=50)
    # Blend LLM and anchor so we capture both qualitative + quantitative signal.
    engagement = (_clamp_int(llm.get("engagement"), default=anchors["engagement"]) + anchors["engagement"]) // 2
    recency = (_clamp_int(llm.get("recency"), default=anchors["recency"]) + anchors["recency"]) // 2

    reasoning = (llm.get("reasoning") or "").strip()[:500] or "No reasoning provided."

    overall = int(
        WEIGHTS["fit"] * fit
        + WEIGHTS["engagement"] * engagement
        + WEIGHTS["audience"] * audience
        + WEIGHTS["recency"] * recency
    )

    green = [str(g)[:120] for g in (llm.get("green_flags") or [])][:6]
    red = [str(r)[:120] for r in (llm.get("red_flags") or [])][:6]

    return ScoredCandidate(
        handle=c["handle"],
        fit=fit,
        engagement=engagement,
        audience=audience,
        recency=recency,
        overall=overall,
        reasoning=reasoning,
        talking_head=_clamp_int(llm.get("talking_head"), default=None),
        bio_quality=_clamp_int(llm.get("bio_quality"), default=None),
        comment_quality=_clamp_int(llm.get("comment_quality"), default=None),
        country_guess=(llm.get("country_guess") or None) or None,
        timezone_bucket=(llm.get("timezone_bucket") or None) or None,
        green_flags=green,
        red_flags=red,
        is_outlier=bool(llm.get("is_outlier") or c.get("is_outlier_flagged")),
    )


def _clamp_int(v, *, default):
    try:
        return max(0, min(100, int(v)))
    except (TypeError, ValueError):
        return default
