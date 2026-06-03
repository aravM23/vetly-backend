"""
Instagram handle verifier — anti-hallucination ground truth.

The LLM brainstorm hallucinates handles + follower counts. Before any
candidate makes it into the dashboard, we verify the handle exists on
Instagram and pull REAL follower / bio / post-count numbers.

Three paths, in order of preference:

  1. ``i.instagram.com/api/v1/users/web_profile_info`` with the public
     ``x-ig-app-id`` header. Returns clean JSON for any public profile.
     This is what the IG website itself calls. No auth required, free,
     but rate-limited per IP — Railway's cloud IPs get 429'd hard.

  2. HTML fallback: ``instagram.com/<handle>/`` and parse the
     ``<meta property="og:description">`` line, which always contains
     "X Followers, Y Following, Z Posts - …" for public profiles. Used
     when the API path 429s us. Same IP gets rate-limited though.

  3. Apify fallback: when ``APIFY_TOKEN`` is set, batch every still-
     unresolved handle into a single ``apify/instagram-profile-scraper``
     run. Apify uses residential IPs so they don't hit IG's cloud-IP
     rate limits. ~$0.45 per 1K profile lookups. This is what saves
     the Railway deploy.

If everything fails with a definitive 404, the handle is hallucinated
→ drop. If everything fails transiently (rate-limit, network, Apify
not configured), we return ``None`` and the runner drops the candidate
to be safe.
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
from dataclasses import dataclass

import httpx

from app.core.config import settings

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

    Two-pass strategy:

      Pass 1: try every handle through the free public IG endpoints
              (concurrency-capped). Fast for handles IG actually serves
              us; gets 429'd from cloud IPs.

      Pass 2: every handle that came back ``None`` is batched into a
              SINGLE Apify actor run (residential IPs, won't 429). This
              gives us ~30s end-to-end for the worst case where IG
              blocks 100% of requests, instead of N×retries.

    If ``APIFY_TOKEN`` isn't configured, we skip pass 2 and return what
    we got from pass 1. ``None`` values mean either a real 404 or a
    transient block we couldn't recover from.
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

    # Apify fallback for the handles IG blocked us on.
    unresolved = [h for h, p in results.items() if p is None]
    if unresolved and settings.apify_token:
        logger.info(
            "verify_many: %d handles unresolved by public IG endpoints, "
            "falling back to Apify (token configured)",
            len(unresolved),
        )
        apify_results = await _verify_via_apify(unresolved)
        for h, prof in apify_results.items():
            if prof is not None:
                results[h] = prof
        logger.info(
            "verify_many: Apify returned %d/%d profiles",
            sum(1 for p in apify_results.values() if p),
            len(unresolved),
        )
    elif unresolved and not settings.apify_token:
        logger.warning(
            "verify_many: %d handles unresolved and APIFY_TOKEN not set — "
            "they will be dropped. Configure APIFY_TOKEN to recover from "
            "Instagram rate-limiting on cloud IPs.",
            len(unresolved),
        )

    return results


# ─── Apify fallback ────────────────────────────────────────────────────────

# Apify's official profile-scraper actor. Returns one record per handle
# with followers, bio, post count, privacy, verified status, etc. Uses
# residential IPs so it doesn't hit Instagram's cloud-IP rate limits.
_APIFY_ACTOR = "apify~instagram-profile-scraper"
_APIFY_RUN_SYNC_URL = (
    f"https://api.apify.com/v2/acts/{_APIFY_ACTOR}/run-sync-get-dataset-items"
)


async def _verify_via_apify(handles: list[str]) -> dict[str, IgProfile | None]:
    """Batch-verify handles via Apify. Returns ``{handle: IgProfile|None}``.

    One synchronous actor run for the entire batch; Apify charges per
    profile, not per call, so batching minimizes latency without
    affecting cost. ~$0.45 per 1K profiles.

    On any failure (auth, timeout, malformed response) we log loudly and
    return all-None so the runner drops the affected candidates rather
    than surfacing fabricated data.
    """
    token = settings.apify_token
    if not token:
        return {h: None for h in handles}

    out: dict[str, IgProfile | None] = {h: None for h in handles}
    if not handles:
        return out

    payload = {"usernames": handles}

    # The Apify actor takes 5-30s for a batch of ~30 handles. Give it
    # plenty of read budget but cap connect quickly.
    timeout = httpx.Timeout(60.0, connect=6.0, read=60.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                _APIFY_RUN_SYNC_URL,
                params={"token": token},
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        if r.status_code != 200:
            logger.warning(
                "Apify verify failed: status=%d body=%r",
                r.status_code, (r.text or "")[:400],
            )
            return out
        data = r.json()
    except (httpx.RequestError, asyncio.TimeoutError, ValueError) as e:
        logger.warning("Apify verify network error: %s", e)
        return out

    if not isinstance(data, list):
        logger.warning("Apify verify: unexpected response shape: %r", type(data))
        return out

    for entry in data:
        if not isinstance(entry, dict):
            continue
        # Apify actor returns an `error` key when a username doesn't exist
        # (the equivalent of a 404 on the public endpoint).
        if entry.get("error"):
            uname = (entry.get("username") or "").lower()
            if uname in out:
                # Definitive 404 — leave as None so runner drops it.
                out[uname] = None
            continue
        username = (entry.get("username") or "").lower()
        if not username or username not in out:
            continue
        followers = _safe_int(entry.get("followersCount")) or 0
        out[username] = IgProfile(
            handle=username,
            display_name=entry.get("fullName") or None,
            biography=entry.get("biography") or None,
            follower_count=followers,
            following_count=_safe_int(entry.get("followsCount")),
            post_count=_safe_int(entry.get("postsCount")),
            is_private=bool(entry.get("private")),
            is_verified=bool(entry.get("verified")),
            profile_pic_url=(
                entry.get("profilePicUrlHD")
                or entry.get("profilePicUrl")
                or None
            ),
            external_url=entry.get("externalUrl") or None,
            source="apify",
        )

    return out
