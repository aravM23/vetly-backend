import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

logger = logging.getLogger(__name__)

engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session() as session:
        yield session


# ─── Lightweight in-place migrations ───────────────────────────────────────
# SQLAlchemy's create_all only creates missing tables — it never adds new
# columns to tables that already exist. For the small set of additive columns
# we ship between releases, run idempotent ALTER TABLEs here so prod boots
# without anyone having to hand-run a migration.

_ADDITIVE_COLUMNS: list[tuple[str, str, str]] = [
    # (table, column, DDL fragment)
    ("creator_candidates", "is_shortlisted", "BOOLEAN DEFAULT 0 NOT NULL"),
    ("creator_candidates", "shortlisted_at", "DATETIME"),
    ("discovery_settings", "program", "VARCHAR(32) DEFAULT 'club_stanley' NOT NULL"),
]


async def _apply_additive_columns():
    is_sqlite = settings.database_url.startswith("sqlite")
    async with engine.begin() as conn:
        for table, column, ddl in _ADDITIVE_COLUMNS:
            try:
                if is_sqlite:
                    cols_res = await conn.execute(
                        text(f"PRAGMA table_info({table})")
                    )
                    existing = {row[1] for row in cols_res.fetchall()}
                else:
                    cols_res = await conn.execute(
                        text(
                            "SELECT column_name FROM information_schema.columns "
                            "WHERE table_name = :t"
                        ),
                        {"t": table},
                    )
                    existing = {row[0] for row in cols_res.fetchall()}
                if column in existing:
                    continue
                await conn.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
                )
                logger.info("migration: added %s.%s", table, column)
            except Exception as e:
                logger.warning(
                    "migration skipped for %s.%s: %s", table, column, e
                )


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await _apply_additive_columns()
