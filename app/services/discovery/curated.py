"""
Curated Creator seed list — emergency fallback when discovery returns 0.

When the LLM brainstorm + Instagram verification pipeline produces zero
survivors (because IG rate-limited Railway, Apify is misconfigured, the
prompt is too strict, or all of the above), we still want the dashboard
to show real, on-niche Creators rather than an empty state. This module
holds a hand-curated set of Creators verified to exist on Instagram
with follower counts in the program windows.

VERIFICATION CONTRACT:

  Every handle in CURATED_CREATORS has been live-verified against
  Instagram's public web_profile_info endpoint and confirmed to:
    1. Exist (HTTP 200, not 404)
    2. Hold real follower counts within the documented windows
    3. Be a real Creator (not a tiny same-name impostor)

  When you add a new entry, run::

      python -c "import asyncio; from app.services.instagram_verify \
        import verify_handle; print(asyncio.run(verify_handle('YOUR_HANDLE')))"

  before committing. If the handle fails verification, do NOT add it —
  the curated list exists specifically to be the trusted floor.

EXTENSION GUIDE:

  • Goal: 100+ entries over time, owned by Jonathan / Mina / Nathan / Leo.
  • Each entry must include `programs` — which dashboards it appears on.
  • Club Stanley window: 50K-500K. Ambassadors: 50K-100K HARD cap.
  • Creators over 100K → ["club_stanley"] only.
  • Creators with 50K-100K who fit the Ambassador qualification rule
    ("audience would be searching for a Stanley-like tool if Stanley
    disappeared tomorrow") → both programs.
"""
from __future__ import annotations

from typing import Literal

Program = Literal["club_stanley", "ambassador"]


# ─── The list ───────────────────────────────────────────────────────────────
#
# Every entry below has been verified against Instagram on 2026-06-03.
# Follower counts are best-known approximations; the verifier always
# overwrites them with truth at runtime.

CURATED_CREATORS: list[dict] = [
    # ── Anchor list (Jonathan's Stanley Ambassador anchors) ──────────────
    {
        "handle": "lizzypalios",
        "display_name": "Lizzy Palios",
        "niche": "Personal brand / content strategy",
        "why_known": "On Jonathan's Ambassador anchor list. Active personal-brand educator on Instagram with consistent teaching cadence.",
        "approx_followers": 112_000,
        "country": "United States",
        "timezone_bucket": "NORAM",
        "programs": ["club_stanley"],
    },
    {
        "handle": "kimcoles",
        "display_name": "Kim Coles",
        "niche": "Personal brand / lifestyle leadership",
        "why_known": "On the user's general anchor list. Long-running personal-brand voice with strong audience engagement.",
        "approx_followers": 362_000,
        "country": "United States",
        "timezone_bucket": "NORAM",
        "programs": ["club_stanley"],
    },
    {
        "handle": "tina.gmorad",
        "display_name": "Tina Ghazimorad",
        "niche": "Personal brand / IG growth",
        "why_known": "On Jonathan's Stanley Ambassador anchor list. Real handle confirmed by user.",
        "approx_followers": 265_000,
        "country": "United Kingdom",
        "timezone_bucket": "UK",
        "programs": ["club_stanley"],
    },

    # ── Vetted by user (2026-06-03) — Creator-economy / writing-online ──
    {
        "handle": "thejustinwelsh",
        "display_name": "Justin Welsh",
        "niche": "Solopreneur / LinkedIn-led personal brand educator",
        "why_known": "The Diversified Solopreneur newsletter; flagship solopreneur-systems educator.",
        "approx_followers": 90_000,
        "country": "United States",
        "timezone_bucket": "NORAM",
        "programs": ["ambassador", "club_stanley"],
    },
    {
        "handle": "nicolascole77",
        "display_name": "Nicolas Cole",
        "niche": "Writing online / Ship 30 for 30",
        "why_known": "Co-founder of Ship 30 for 30; writing-online educator with curriculum.",
        "approx_followers": 94_000,
        "country": "United States",
        "timezone_bucket": "NORAM",
        "programs": ["ambassador", "club_stanley"],
    },
    {
        "handle": "gregisenberg",
        "display_name": "Greg Isenberg",
        "niche": "Startup ideas / community-driven products",
        "why_known": "Late Checkout founder; weekly newsletter + podcast on community-driven startups.",
        "approx_followers": 123_000,
        "country": "United States",
        "timezone_bucket": "NORAM",
        "programs": ["club_stanley"],
    },
    {
        "handle": "alexlieb",
        "display_name": "Alex Lieberman",
        "niche": "Founder content / Morning Brew co-founder",
        "why_known": "Morning Brew co-founder; long-form founder-content operator.",
        "approx_followers": 151_000,
        "country": "United States",
        "timezone_bucket": "NORAM",
        "programs": ["club_stanley"],
    },
    {
        "handle": "kallaway",
        "display_name": "Kallaway",
        "niche": "AI + creator workflows",
        "why_known": "AI-creator-economy commentator with strong execution focus.",
        "approx_followers": 437_000,
        "country": "United States",
        "timezone_bucket": "NORAM",
        "programs": ["club_stanley"],
    },
    {
        "handle": "noahkagan",
        "display_name": "Noah Kagan",
        "niche": "Founder / AppSumo / business-of-creating",
        "why_known": "AppSumo founder; long-running creator-business educator.",
        "approx_followers": 253_000,
        "country": "United States",
        "timezone_bucket": "NORAM",
        "programs": ["club_stanley"],
    },
    {
        "handle": "joepompliano",
        "display_name": "Joe Pompliano",
        "niche": "Sports business newsletter Creator",
        "why_known": "Huddle Up newsletter; sports business Creator-economy operator.",
        "approx_followers": 120_000,
        "country": "United States",
        "timezone_bucket": "NORAM",
        "programs": ["club_stanley"],
    },
    {
        "handle": "taylorlorenz",
        "display_name": "Taylor Lorenz",
        "niche": "Creator-economy journalist",
        "why_known": "Creator-economy reporter; deep audience overlap with Stanley ICP.",
        "approx_followers": 183_000,
        "country": "United States",
        "timezone_bucket": "NORAM",
        "programs": ["club_stanley"],
    },
    {
        "handle": "colinandsamir",
        "display_name": "Colin and Samir",
        "niche": "Creator-economy podcast",
        "why_known": "Colin & Samir — flagship Creator-economy podcast and YouTube channel.",
        "approx_followers": 299_000,
        "country": "United States",
        "timezone_bucket": "NORAM",
        "programs": ["club_stanley"],
    },
    {
        "handle": "christopherclaflin",
        "display_name": "Christopher Claflin",
        "niche": "Personal brand / content strategy",
        "why_known": "Vetted on user's curated list (2026-06-03).",
        "approx_followers": 195_000,
        "country": "United States",
        "timezone_bucket": "NORAM",
        "programs": ["club_stanley"],
    },

    # ── High-confidence content-business operators ───────────────────────
    {
        "handle": "elisedarma",
        "display_name": "Elise Darma",
        "niche": "Instagram & content-business coach",
        "why_known": "Runs InstaGrowth Boss; flagship content-business educator with multi-year track record.",
        "approx_followers": 174_000,
        "country": "Canada",
        "timezone_bucket": "NORAM",
        "programs": ["club_stanley"],
    },
    {
        "handle": "vanessalau.co",
        "display_name": "Vanessa Lau",
        "niche": "Career & personal-brand coach for Creators",
        "why_known": "Bossgram Academy founder; YouTube + IG personal-brand educator with curriculum products.",
        "approx_followers": 318_000,
        "country": "Canada",
        "timezone_bucket": "NORAM",
        "programs": ["club_stanley"],
    },
    {
        "handle": "amyporterfield",
        "display_name": "Amy Porterfield",
        "niche": "Online business / digital course educator",
        "why_known": "Online Marketing Made Easy podcast; Digital Course Academy.",
        "approx_followers": 469_000,
        "country": "United States",
        "timezone_bucket": "NORAM",
        "programs": ["club_stanley"],
    },
    {
        "handle": "seancannell",
        "display_name": "Sean Cannell",
        "niche": "YouTube + video Creator-economy educator",
        "why_known": "Think Media founder; flagship video-Creator education channel.",
        "approx_followers": 209_000,
        "country": "United States",
        "timezone_bucket": "NORAM",
        "programs": ["club_stanley"],
    },
    {
        "handle": "katybellotte",
        "display_name": "Katy Bellotte",
        "niche": "Personal brand / content strategy podcaster",
        "why_known": "Thread podcast — multi-channel personal-brand operator with builds-in-public energy.",
        "approx_followers": 191_000,
        "country": "United States",
        "timezone_bucket": "NORAM",
        "programs": ["club_stanley"],
    },
    {
        "handle": "angiebellemare",
        "display_name": "Angie Bellemare",
        "niche": "Personal brand / content Creator",
        "why_known": "Multi-platform personal-brand Creator with strong teaching POV on content workflow.",
        "approx_followers": 111_000,
        "country": "Canada",
        "timezone_bucket": "NORAM",
        "programs": ["club_stanley"],
    },

    # ── Ambassador-window (50K-100K) verified seeds ──────────────────────
    {
        "handle": "creativelysquared",
        "display_name": "Creatively Squared",
        "niche": "Instagram visual-content educator for businesses",
        "why_known": "Visual content strategy for solopreneurs and small businesses. Audience explicitly wants execution help.",
        "approx_followers": 75_000,
        "country": "Australia",
        "timezone_bucket": "APAC",
        "programs": ["ambassador", "club_stanley"],
    },
]


def for_program(program: Program, limit: int = 8) -> list[dict]:
    """Return up to ``limit`` curated Creators relevant to ``program``.

    Picks the program-relevant subset (intersection of entry's
    ``programs`` field with the requested program), shuffles for
    variety, and slices. The runner uses this as its emergency fallback
    when the LLM-brainstorm + verification pipeline produces zero
    survivors.
    """
    import random

    pool = [c for c in CURATED_CREATORS if program in c.get("programs", [])]
    random.shuffle(pool)
    return pool[:limit]
