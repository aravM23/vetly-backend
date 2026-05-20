from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.models import User, TrackedCreator
from app.schemas.schemas import TrackedCreatorCreate, TrackedCreatorResponse
from app.services.instagram import get_scraper, ingest_creator_posts

router = APIRouter(prefix="/users/{user_id}/creators", tags=["tracked creators"])


@router.post("/", response_model=TrackedCreatorResponse, status_code=201)
async def track_creator(
    user_id: int,
    payload: TrackedCreatorCreate,
    db: AsyncSession = Depends(get_db),
):
    user = await db.execute(select(User).where(User.id == user_id))
    if not user.scalar_one_or_none():
        raise HTTPException(404, "User not found")

    existing = await db.execute(
        select(TrackedCreator).where(
            and_(
                TrackedCreator.user_id == user_id,
                TrackedCreator.instagram_handle == payload.instagram_handle,
            )
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(400, "Already tracking this creator")

    scraper = get_scraper()
    profile = await scraper.fetch_creator_profile(payload.instagram_handle)

    creator = TrackedCreator(
        user_id=user_id,
        instagram_handle=payload.instagram_handle,
        display_name=profile.get("display_name"),
        follower_count=profile.get("follower_count"),
    )
    db.add(creator)
    await db.commit()
    await db.refresh(creator)

    await ingest_creator_posts(db, creator)

    return creator


@router.get("/", response_model=list[TrackedCreatorResponse])
async def list_tracked_creators(
    user_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(TrackedCreator).where(
            and_(
                TrackedCreator.user_id == user_id,
                TrackedCreator.is_active == True,
            )
        )
    )
    return result.scalars().all()


@router.delete("/{creator_id}")
async def untrack_creator(
    user_id: int,
    creator_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(TrackedCreator).where(
            and_(
                TrackedCreator.id == creator_id,
                TrackedCreator.user_id == user_id,
            )
        )
    )
    creator = result.scalar_one_or_none()
    if not creator:
        raise HTTPException(404, "Creator not found")
    creator.is_active = False
    await db.commit()
    return {"status": "untracked"}
