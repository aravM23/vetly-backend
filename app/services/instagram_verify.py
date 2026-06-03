"""
Instagram handle verifier — anti-hallucination ground truth.

The LLM brainstorm hallucinates handles + follower counts. Before any
candidate makes it into the dashboard, we verify the handle exists on
Instagram and pull REAL follower / bio / post-count numbers.

Two paths, in order of preference:

  1. ``i.instagram.com/api/v1/users/web_profile_info`` with the public
     ``x-ig-app-id`` header. Returns clean JSON for any public profile.
     This is what the IG website itself calls. No auth required, but
     rate-limited per IP.

  2. HTML fallback: ``instagram.com/<handle>/`` and parse the
     ``<meta property="og:description">`` line, which always contains
     "X Followers, Y Following, Z Posts - …" for public profiles. Used
     when the API path 429s us.

If both fail with a definitive 404, the handle is hallucinated → drop.
If both fail transiently (rate-limit, network), we return ``None`` and
let the caller decide whether to trust the LLM's claim or drop.
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# Public web-client app id Instagram itself ships in the browser.
_IG_APP_ID = "936619743392459"

_USER_AGENTS = [
    # Recent Chrome on macOS / Windows / Linux. We rotate to spread load.
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

_HANDLE_OK = re.compile(r"^[A-Za-z0-9._]{1,30}$")

# Parses "1,234,567 Followers, 200 Following, 1,200 Posts - …" from og:description.
_META_FOLLOWERS = re.compile(
    r"([\d,\.]+)\s*(?:K|M|B)?\s*Followers,\s*([\d,\.]+)\s*(?:K|M|B)?\s*Following,\s*([\d,\.]+)\s*(?:K|M|B)?\s*Posts",
    re.I,
)

# OG-description sometimes uses K/M suffixes; capture optional letter too.
_META_FOLLOWERS_SUFFIX = re.compile(
    r"([\d,\.]+)\s*(K|M|B)?\s*Followers,\s*([\d,\.]+)\s*(K|M|B)?\s*Following,\s*([\d,\.]+)\s*(K|M|B)?\s*Posts",
    re.I,
)


@dataclass
class IgProfile:
    """A real, verified-on-Instagram profile snapshot."""

    handle: str
    display_name: str | None
    biography: str | None
    follower_count: int
    following_count: int | None
    post_count: int | None
    is_private: bool
    is_verified: bool
    profile_pic_url: str | None
    external_url: str | None
    source: str  # "web_profile_info" | "og_meta"


async def verify_handle(handle: str, *, client: httpx.AsyncClient | None = None) -> IgProfile | None:
    """Verify a handle exists on Instagram and pull real public profile data.

    Returns ``IgProfile`` on success, ``None`` if the handle is definitively
    bogus (404) or if every endpoint blocked us (transient — caller should
    treat as a soft failure and drop the candidate to be safe).
    """
    handle = (handle or "").strip().lstrip("@").lower()
    if not handle or not _HANDLE_OK.match(handle):
        return None

    own_client = client is None
    if own_client:
        # Tight timeouts: a single request shouldn't hold up the whole run.
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(6.0, connect=4.0, read=6.0),
            follow_redirects=True,
        )

    try:
        # 1. JSON path
        prof = await _try_web_profile_info(client, handle)
        if prof is not None:
            return prof

        # 2. HTML fallback
        prof = await _try_og_meta(client, handle)
        if prof is not None:
            return prof

        return None
    finally:
        if own_client:
            await client.aclose()


async def _try_web_profile_info(client: httpx.AsyncClient, handle: str) -> IgProfile | None:
    url = f"https://i.instagram.com/api/v1/users/web_profile_info/?username={handle}"
    headers = {
        "User-Agent": random.choice(_USER_AGENTS),
        "x-ig-app-id": _IG_APP_ID,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    # Two attempts max — a third doesn't materially help if IG is rate-limiting,
    # and stretches the per-handle worst case past the discovery run budget.
    for attempt in range(2):
        try:
            r = await client.get(url, headers=headers)
        except (httpx.RequestError, asyncio.TimeoutError) as e:
            logger.debug("verify[%s]: web_profile_info network error %s", handle, e)
            return None

        if r.status_code == 200:
            try:
                data = r.json()
            except ValueError:
                return None
            user = (data.get("data") or {}).get("user")
            if not user:
                return None
            return IgProfile(
                handle=(user.get("username") or handle).lower(),
                display_name=(user.get("full_name") or None) or None,
                biography=user.get("biography") or None,
                follower_count=int(((user.get("edge_followed_by") or {}).get("count")) or 0),
                following_count=_safe_int((user.get("edge_follow") or {}).get("count")),
                post_count=_safe_int(
                    (user.get("edge_owner_to_timeline_media") or {}).get("count")
                ),
                is_private=bool(user.get("is_private")),
                is_verified=bool(user.get("is_verified")),
                profile_pic_url=user.get("profile_pic_url_hd") or user.get("profile_pic_url"),
                external_url=user.get("external_url") or None,
                source="web_profile_info",
            )

        if r.status_code == 404:
            return None

        # 429 / 401 / 403 — back off briefly then try once more.
        await asyncio.sleep(0.4 + random.random() * 0.3)

    return None


async def _try_og_meta(client: httpx.AsyncClient, handle: str) -> IgProfile | None:
    url = f"https://www.instagram.com/{handle}/"
    headers = {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        r = await client.get(url, headers=headers, follow_redirects=True)
    except (httpx.RequestError, asyncio.TimeoutError):
        return None

    if r.status_code == 404:
        return None
    if r.status_code != 200:
        return None

    html = r.text or ""
    # We only get follower count + post count from the og:description meta.
    # That's enough for verification — bio comes from the JSON path on retry.
    m = _META_FOLLOWERS_SUFFIX.search(html)
    if not m:
        # Some private / age-walled / login-only profiles don't expose meta.
        return None

    followers_raw, f_suf, following_raw, g_suf, posts_raw, p_suf = m.groups()
    followers = _parse_count(followers_raw, f_suf)
    following = _parse_count(following_raw, g_suf)
    posts = _parse_count(posts_raw, p_suf)

    # Try to lift the display name from <title>"<name> (@handle) • Instagram photos…"</title>
    display_name = None
    title_match = re.search(r"<title>([^<]*)</title>", html, re.I)
    if title_match:
        title = title_match.group(1)
        # Title format: "Name (@handle) • Instagram photos and videos"
        name_match = re.match(r"^(.*?)\s*\(\s*@", title)
        if name_match:
            display_name = name_match.group(1).strip() or None

    biography = None
    desc_match = re.search(
        r'<meta\s+name="description"\s+content="([^"]*)"', html, re.I
    )
    if desc_match:
        # Description is "X Followers, Y Following, Z Posts - See Instagram photos
        # and videos from Name (@handle)" — useful as last-resort bio? Skip.
        pass

    return IgProfile(
        handle=handle,
        display_name=display_name,
        biography=biography,
        follower_count=followers,
        following_count=following,
        post_count=posts,
        is_private=False,  # if we got the meta, it's at least listable
        is_verified=False,
        profile_pic_url=None,
        external_url=None,
        source="og_meta",
    )


# ─── helpers ────────────────────────────────────────────────────────────────


def _safe_int(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _parse_count(num: str, suffix: str | None) -> int:
    """Parse "1,234" / "1.2K" / "3M" into an int."""
    n = float(num.replace(",", ""))
    s = (suffix or "").upper()
    if s == "K":
        n *= 1_000
    elif s == "M":
        n *= 1_000_000
    elif s == "B":
        n *= 1_000_000_000
    return int(n)


# ─── batched API for the discovery runner ──────────────────────────────────


async def verify_many(
    handles: list[str], *, concurrency: int = 6
) -> dict[str, IgProfile | None]:
    """Verify a batch of handles. Returns ``{handle: IgProfile|None}``.

    Capped concurrency to be polite to Instagram. Each ``None`` value
    means the handle either doesn't exist or every endpoint blocked us.
    Bounded per-call timeout (~6s + 1 retry) keeps worst-case batch time
    in the tens of seconds even when IG is throttling us.
    """
    if not handles:
        return {}
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, IgProfile | None] = {}

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(6.0, connect=4.0, read=6.0),
        follow_redirects=True,
    ) as client:
        async def _one(h: str) -> None:
            async with sem:
                results[h] = await verify_handle(h, client=client)
                # Tiny jitter so we don't burst.
                await asyncio.sleep(0.1 + random.random() * 0.15)

        await asyncio.gather(*(_one(h) for h in handles))

    return results
