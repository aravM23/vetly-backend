"""
Content rewriter service.

When a velocity spike is detected, this service takes the trending post's
format/structure and rewrites it through the lens of the user's content pillars.
The output is a ready-to-film draft, not a suggestion.
"""
import logging
import json
from dataclasses import dataclass

from openai import AsyncOpenAI

from app.core.config import settings
from app.models.models import CreatorPost, User

logger = logging.getLogger(__name__)


@dataclass
class DraftContent:
    hook: str
    structure: dict
    rationale: str
    estimated_production_time: str


REWRITE_SYSTEM_PROMPT = """You are Stanley's Velocity Alert engine. A competitor just hit a viral spike. 
Your job is to reverse-engineer WHY the post is performing and generate an immediately actionable draft 
for the user that rides the same algorithmic wave but using THEIR unique content pillars.

You must return valid JSON with exactly these keys:
{
  "hook": "The exact opening line/hook the user should use (written in their voice)",
  "visual_beats": ["Beat 1: description", "Beat 2: description", ...],
  "caption_draft": "Full caption draft with hashtag strategy",
  "format_breakdown": "Why this format is working right now (1-2 sentences)", 
  "adaptation_notes": "How this was adapted to the user's pillars (1-2 sentences)",
  "cta": "Call to action that fits the user's brand",
  "estimated_production_time": "Quick estimate (e.g. '30 min', '1 hour')"
}

Rules:
- The hook must be SPECIFIC to the user's narrative, not generic
- Visual beats should be concrete enough to film right now
- Do not be generic. Reference the user's actual content pillars.
- The tone must match what the user's audience expects
- Be urgent but not desperate"""


async def generate_draft(
    user: User, spike_post: CreatorPost, velocity_multiplier: float
) -> DraftContent:
    """Generate a rewritten draft based on a trending post and user's pillars."""
    pillars = user.content_pillars or {}
    primary_narrative = pillars.get("primary_narrative", "their content journey")
    topics = pillars.get("topics", [])
    tone = pillars.get("tone", "authentic and direct")
    audience = pillars.get("audience", "young professionals")

    user_prompt = f"""TRENDING POST DETECTED:
- Creator: {spike_post.creator.instagram_handle if spike_post.creator else 'unknown'}
- Views: {spike_post.views:,} ({velocity_multiplier:.1f}x their average)
- Format: {spike_post.detected_format or 'unknown'}
- Hook type: {spike_post.detected_hook_type or 'unknown'}
- Caption: {(spike_post.caption or '')[:500]}
- Post type: {spike_post.post_type or 'reel'}

USER'S CONTENT PILLARS:
- Primary narrative: {primary_narrative}
- Core topics: {', '.join(topics) if topics else 'general content creation'}
- Brand tone: {tone}
- Target audience: {audience}
- Instagram handle: {user.instagram_handle or 'unknown'}

Generate a complete draft that rides this exact algorithmic wave using the user's unique positioning."""

    if not settings.openai_api_key or settings.openai_api_key.startswith("sk-your"):
        return _generate_fallback_draft(user, spike_post, velocity_multiplier)

    try:
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.8,
            max_tokens=1000,
        )
        result = json.loads(response.choices[0].message.content)
        return DraftContent(
            hook=result.get("hook", ""),
            structure={
                "visual_beats": result.get("visual_beats", []),
                "caption_draft": result.get("caption_draft", ""),
                "format_breakdown": result.get("format_breakdown", ""),
                "adaptation_notes": result.get("adaptation_notes", ""),
                "cta": result.get("cta", ""),
            },
            rationale=result.get("format_breakdown", ""),
            estimated_production_time=result.get("estimated_production_time", "unknown"),
        )
    except Exception as e:
        logger.error(f"OpenAI rewrite failed: {e}")
        return _generate_fallback_draft(user, spike_post, velocity_multiplier)


def _generate_fallback_draft(
    user: User, post: CreatorPost, multiplier: float
) -> DraftContent:
    """Template-based draft when OpenAI is unavailable."""
    pillars = user.content_pillars or {}
    narrative = pillars.get("primary_narrative", "your journey")
    topics = pillars.get("topics", ["content creation"])
    fmt = post.detected_format or "reel"
    hook_type = post.detected_hook_type or "hook_question"

    hook_templates = {
        "hook_question": f"What nobody tells you about {topics[0] if topics else 'this'}...",
        "hook_cliffhanger": f"I almost gave up on {narrative}. Then this happened.",
        "hook_controversial": f"Unpopular opinion about {topics[0] if topics else 'this industry'}:",
        "hook_promise": f"Here's exactly how I handle {topics[0] if topics else 'this'} (step by step)",
        "hook_relatable": f"POV: You're trying to balance {narrative} and nobody gets it",
        "hook_versus": f"The difference between people who succeed at {topics[0] if topics else 'this'} and those who don't",
    }

    hook = hook_templates.get(hook_type, hook_templates["hook_question"])

    return DraftContent(
        hook=hook,
        structure={
            "visual_beats": [
                f"Beat 1: Open with {hook_type.replace('hook_', '')} — straight to camera",
                f"Beat 2: The problem/tension — connect to {narrative}",
                f"Beat 3: The insight/turn — your unique angle on {topics[0] if topics else 'this'}",
                f"Beat 4: Proof/example from your experience",
                f"Beat 5: CTA — drive to comments or saves",
            ],
            "format_breakdown": (
                f"This {fmt} format hit {multiplier:.1f}x because the "
                f"'{hook_type.replace('hook_', '')}' opening pattern triggers "
                f"the algorithm's early retention signal."
            ),
            "adaptation_notes": f"Adapted from {fmt} format to fit your narrative: {narrative}",
            "cta": "Save this for later and drop a comment with your experience",
        },
        rationale=(
            f"The original post used a '{fmt}' format with a "
            f"'{hook_type.replace('hook_', '')}' hook pattern, which hit "
            f"{multiplier:.1f}x the creator's average. This format works because "
            f"it front-loads curiosity, which boosts early retention — the #1 signal "
            f"Instagram's algorithm uses to push reels."
        ),
        estimated_production_time="30-45 min",
    )
