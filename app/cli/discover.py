"""
Standalone discovery runner.

Usage:
  cd backend
  python -m app.cli.discover                       # uses first user, default settings
  python -m app.cli.discover --user 1              # specific user
  python -m app.cli.discover --user 1 --limit 30   # bigger run
  python -m app.cli.discover --user 1 --no-scrape  # LLM brainstorm + mock only
  python -m app.cli.discover --top 25              # show top N after the run

Prints a ranked table of pending candidates from the latest run. Approve
candidates via the API (or call promote_candidate from the Python REPL).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime

from sqlalchemy import desc, select

from app.core.database import async_session, init_db
from app.models.discovery_models import CandidateStatus, CreatorCandidate
from app.models.models import User
# Make sure all model modules are loaded so init_db sees every table.
from app.models import timing_models  # noqa: F401
from app.models import discovery_models  # noqa: F401
from app.services.discovery.runner import run_discovery


# ─── Pretty printing ────────────────────────────────────────────────────────


def _truncate(s: str | None, width: int) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ").strip()
    return s if len(s) <= width else s[: width - 1] + "…"


def _fmt_followers(n: int | None) -> str:
    if not n:
        return "—"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def _print_table(candidates: list[CreatorCandidate]) -> None:
    if not candidates:
        print("\n(no Creators surfaced)\n")
        return

    headers = [
        "#", "Score", "Handle", "Followers", "Posts/wk",
        "ER", "Fit", "Geo", "Flags", "Source",
    ]
    widths = [3, 5, 26, 9, 8, 6, 4, 10, 4, 12]
    sep = "  "

    def row(values):
        return sep.join(str(v).ljust(w) for v, w in zip(values, widths))

    print()
    print(row(headers))
    print(row(["-" * w for w in widths]))
    for i, c in enumerate(candidates, start=1):
        er = f"{(c.engagement_rate or 0) * 100:.1f}%" if c.engagement_rate else "—"
        ppw = f"{c.posts_per_week:.1f}" if c.posts_per_week is not None else "—"
        fit = c.score_fit if c.score_fit is not None else "—"
        geo = (c.timezone_bucket or c.country_guess or "—")[: widths[7]]
        flag_bits = []
        if c.is_outlier_flagged:
            flag_bits.append("OUT")
        if c.green_flags:
            flag_bits.append(f"+{len(c.green_flags)}")
        if c.red_flags:
            flag_bits.append(f"-{len(c.red_flags)}")
        flags = ",".join(flag_bits) or "—"
        print(
            row(
                [
                    i,
                    c.score_overall if c.score_overall is not None else "—",
                    _truncate("@" + c.handle, widths[2]),
                    _fmt_followers(c.follower_count),
                    ppw,
                    er,
                    fit,
                    geo,
                    flags,
                    _truncate(c.discovered_via, widths[9]),
                ]
            )
        )
    print()

    # Per-Creator reasoning + flag detail below the table — easier to scan
    # than cramming it into a column.
    for i, c in enumerate(candidates, start=1):
        line = f"  {i}. @{c.handle}"
        if c.is_outlier_flagged:
            line += "  [OUTLIER]"
        print(line)
        if c.green_flags:
            print(f"     + {'; '.join(c.green_flags)}")
        if c.red_flags:
            print(f"     - {'; '.join(c.red_flags)}")
        if c.score_reasoning:
            print(f"     · {c.score_reasoning}")
    print()


# ─── Main ───────────────────────────────────────────────────────────────────


async def _resolve_user(db, user_arg: int | None) -> User:
    if user_arg is not None:
        res = await db.execute(select(User).where(User.id == user_arg))
        u = res.scalar_one_or_none()
        if not u:
            print(f"User id={user_arg} not found.", file=sys.stderr)
            sys.exit(2)
        return u

    res = await db.execute(select(User).order_by(User.id).limit(1))
    u = res.scalar_one_or_none()
    if u:
        return u

    # First-run bootstrap so the CLI works on a totally fresh DB.
    print("No users found — creating a demo user 'club_stanley'.")
    u = User(username="club_stanley", niche_tags=["lifestyle", "hydration", "wellness"])
    db.add(u)
    await db.commit()
    await db.refresh(u)
    return u


async def main_async(args: argparse.Namespace) -> None:
    await init_db()
    async with async_session() as db:
        user = await _resolve_user(db, args.user)
        print(
            f"\nDiscovery run for user #{user.id} ({user.username}) "
            f"— scrapers={'on' if args.scrape else 'off'}, per_source_limit={args.limit}\n"
        )

        started = datetime.utcnow()
        run = await run_discovery(
            db,
            user_id=user.id,
            use_scrapers=args.scrape,
            per_source_limit=args.limit,
        )
        elapsed = (datetime.utcnow() - started).total_seconds()

        print(
            f"Run #{run.id} status={run.status} in {elapsed:.1f}s   "
            f"raw={run.raw_count}  deduped={run.deduped_count}  "
            f"hydrated={run.hydrated_count}  scored={run.scored_count}"
        )
        if run.sources_used:
            print(f"Sources: {', '.join(run.sources_used)}")
        if run.error_message:
            print(f"Error: {run.error_message}")

        cand_res = await db.execute(
            select(CreatorCandidate)
            .where(
                CreatorCandidate.user_id == user.id,
                CreatorCandidate.status == CandidateStatus.PENDING,
            )
            .order_by(desc(CreatorCandidate.score_overall))
            .limit(args.top)
        )
        _print_table(list(cand_res.scalars().all()))

        print(
            "Approve a candidate via the API:\n"
            f"  curl -X POST http://localhost:8000/api/users/{user.id}/discover/candidates/<id>/approve\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Club Stanley creator discovery.")
    parser.add_argument("--user", type=int, default=None, help="User id (defaults to first user, creates one if none).")
    parser.add_argument("--limit", type=int, default=None, help="Per-source candidate limit (overrides settings).")
    parser.add_argument("--top", type=int, default=20, help="How many top candidates to print.")
    parser.add_argument(
        "--no-scrape",
        dest="scrape",
        action="store_false",
        help="Skip IG scrapers; use only the LLM brainstorm + mock source.",
    )
    parser.set_defaults(scrape=True)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
