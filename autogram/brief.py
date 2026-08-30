"""AI-tools creative brief generation.

This module is intentionally dedicated to the AI-tools content workflow.

It generates fresh, non-repeating briefs covering:
  * AI tools and platforms
  * AI features and capabilities
  * AI automation
  * AI productivity workflows
  * AI coding/development tools
  * AI image/video/audio tools
  * AI content-creation workflows
  * Practical AI use cases and tutorials

This module intentionally contains NO:
  * romantic/couple logic
  * male/female character descriptors
  * interaction_styles
  * relationship content
  * couple photography
  * romance-specific locations

The active AI-tools profile remains responsible for the editorial theme
and image prompt anchor. This module only generates the structured brief
and deterministic visual variation.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from rapidfuzz import fuzz

from .caption import OllamaClient, OllamaError
from .config import Config
from .logging_utils import get_logger

log = get_logger("brief")


# ---------------------------------------------------------------------------
# Brief schema
# ---------------------------------------------------------------------------


class Brief(BaseModel):
    """Structured AI-tools creative brief."""

    subject: str
    setting: str
    lighting: str
    mood: str
    composition: str
    color_palette: str
    time_of_day: str

    style_modifiers: list[str] = Field(
        default_factory=list
    )

    # Deterministic scene metadata.
    location_name: str = ""
    interaction: str = ""
    framing: str = ""


# ---------------------------------------------------------------------------
# Deterministic seed / variation
# ---------------------------------------------------------------------------


def compute_seed(
    run_date: str,
    salt: str,
) -> int:
    """Create a deterministic 31-bit seed."""

    digest = hashlib.sha256(
        f"{run_date}|{salt}".encode()
    ).hexdigest()

    return int(
        digest[:8],
        16,
    )


def select_axis_hints(
    rng: random.Random,
    axes: dict[str, list[str]],
) -> dict[str, str]:
    """Select one value from every configured creative axis."""

    return {
        axis: rng.choice(values)
        for axis, values in axes.items()
        if values
    }


# ---------------------------------------------------------------------------
# Subject deduplication
# ---------------------------------------------------------------------------


def _normalize_subject(
    subject: str,
) -> str:
    """Normalize subject text for duplicate comparison."""

    return re.sub(
        r"\s+",
        " ",
        subject.lower().strip(),
    )


def is_near_duplicate(
    subject: str,
    history_subjects: list[str],
    threshold: float,
) -> bool:
    """Return True when a subject is too similar to recent subjects."""

    normalized = _normalize_subject(subject)

    for previous in history_subjects:
        previous_normalized = _normalize_subject(
            previous
        )

        if (
            fuzz.token_set_ratio(
                normalized,
                previous_normalized,
            )
            > threshold
        ):
            return True

    return False


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


def flatten_locations(
    locations_dict: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    """Flatten configured AI-tools location categories."""

    locations: list[dict[str, str]] = []

    for scenarios in locations_dict.values():
        if isinstance(scenarios, list):
            locations.extend(scenarios)

    return locations


def select_unique_location(
    rng: random.Random,
    locations: list[dict[str, str]],
    history_locations: list[str],
) -> dict[str, str] | None:
    """Select an unused AI-tools environment.

    Locations rotate until the configured pool is exhausted.
    """

    available = [
        location
        for location in locations
        if location.get("name", "")
        not in history_locations
    ]

    if not available:
        log.warning(
            "all AI-tools locations used; cycling location pool"
        )
        available = locations

    if not available:
        return None

    return rng.choice(available)


def extract_location_from_history(
    history_briefs: list[dict[str, Any]],
) -> list[str]:
    """Extract previously used locations."""

    locations: list[str] = []

    for brief in history_briefs:
        name = brief.get(
            "location_name"
        )

        if name:
            locations.append(
                str(name)
            )

    return locations


# ---------------------------------------------------------------------------
# AI-tools visual variation
# ---------------------------------------------------------------------------


_AI_SHOT_DISTANCES = [
    "extreme close-up showing the AI interface or key technology detail clearly",
    "close-up of the AI tool being actively used",
    "medium shot showing the workstation and AI workflow",
    "three-quarter environmental shot showing the technology and its context",
    "over-the-shoulder view focused on the AI software interface",
    "wide professional technology workspace",
    "full environmental technology scene with strong visual storytelling",
    "top-down view of a complete AI productivity workspace",
]


_AI_CAMERA_ANGLES = [
    "eye-level angle",
    "slight low angle",
    "slight high angle",
    "three-quarter perspective",
    "overhead top-down view",
    "over-the-shoulder perspective",
    "side perspective",
]


_AI_ALIGNMENTS = [
    "clean rule-of-thirds composition",
    "main subject positioned on the left third",
    "main subject positioned on the right third",
    "strong central product composition",
    "generous negative space for educational text",
    "off-center composition with leading lines",
    "asymmetric composition with layered depth",
    "clean editorial technology composition",
]


_AI_VISUAL_CUES = [
    "premium technology editorial photography",
    "realistic professional workstation",
    "clean educational technology visual",
    "modern SaaS product aesthetic",
    "credible AI workflow visualization",
    "cinematic but realistic technology photography",
    "professional creator workflow",
    "modern productivity environment",
]


def select_framing(
    rng: random.Random,
) -> str:
    """Generate deterministic AI-tools framing."""

    return (
        f"{rng.choice(_AI_SHOT_DISTANCES)}, "
        f"{rng.choice(_AI_CAMERA_ANGLES)}, "
        f"{rng.choice(_AI_ALIGNMENTS)}, "
        f"{rng.choice(_AI_VISUAL_CUES)}"
    )


def _pick(
    rng: random.Random,
    data: dict[str, Any],
    key: str,
) -> str:
    """Randomly select a configured value."""

    values = data.get(key)

    if isinstance(values, list) and values:
        return str(
            rng.choice(values)
        )

    return ""


def _pick_trend(
    rng: random.Random,
    trends: dict[str, Any],
    key: str,
) -> str:
    """Randomly select a configured photography trend."""

    values = trends.get(key)

    if isinstance(values, list) and values:
        return str(
            rng.choice(values)
        )

    return ""


def select_style(
    rng: random.Random,
    visual_data: dict[str, Any],
) -> dict[str, str]:
    """Select AI-tools visual direction.

    No character or romantic interaction data is read here.
    """

    trends = visual_data.get(
        "photography_trends",
        {},
    )

    if not isinstance(trends, dict):
        trends = {}

    ai_direction = visual_data.get(
        "ai_tools_visual_direction",
        {},
    )

    if not isinstance(ai_direction, dict):
        ai_direction = {}

    return {
        "framing": select_framing(rng),

        "visual_format": _pick(
            rng,
            ai_direction,
            "formats",
        ),

        "context": _pick(
            rng,
            ai_direction,
            "contexts",
        ),

        "action": _pick(
            rng,
            ai_direction,
            "actions",
        ),

        "ui_style": _pick(
            rng,
            ai_direction,
            "ui_styles",
        ),

        "composition": _pick_trend(
            rng,
            trends,
            "compositions",
        ),

        "lighting_style": _pick_trend(
            rng,
            trends,
            "lighting_styles",
        ),

        "color_grading": _pick_trend(
            rng,
            trends,
            "color_grading",
        ),

        "depth_of_field": _pick_trend(
            rng,
            trends,
            "depth_of_field",
        ),
    }


# ---------------------------------------------------------------------------
# Image prompt anchor
# ---------------------------------------------------------------------------


def build_character_block(
    cfg: Config,
) -> str:
    """Return the configured AI-tools image prompt anchor.

    The function name is retained for compatibility with existing callers.

    It does NOT return character information.
    """

    return cfg.active_content.prompt_anchor.strip()


def render_prompts(
    brief: Brief,
    cfg: Config,
    characters_block: str = "",
) -> tuple[str, str]:
    """Render positive and negative Stable Diffusion prompts."""

    fields: dict[str, Any] = (
        brief.model_dump()
    )

    fields["style_modifiers"] = (
        ", ".join(
            brief.style_modifiers
        )
    )

    # Kept under the existing field name so the rest of the pipeline does
    # not need to change.
    fields["characters"] = (
        characters_block
    )

    positive = (
        cfg.image.positive_template.format(
            **fields
        )
    )

    negative = (
        cfg.image.negative_template
    )

    positive = re.sub(
        r"(,\s*){2,}",
        ", ",
        positive,
    ).strip(" ,")

    return positive, negative


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------


def _build_messages(
    cfg: Config,
    axis_hints: dict[str, str],
    recent_briefs: list[dict[str, Any]],
    error_feedback: str | None,
    visual_data: dict[str, Any],
    selected_location: dict[str, str] | None,
    style: dict[str, str],
) -> list[dict[str, str]]:
    """Build the AI-tools-only LLM request."""

    schema = (
        '{"subject": "string", '
        '"setting": "string", '
        '"lighting": "string", '
        '"mood": "string", '
        '"composition": "string", '
        '"color_palette": "string", '
        '"time_of_day": "string", '
        '"style_modifiers": ["string", ...]}'
    )

    system = (
        "You are generating content for an AI-tools Instagram channel. "
        f"{cfg.active_content.system_prompt.strip()} "
        "Respond ONLY with a JSON object matching this schema exactly: "
        f"{schema}. "
        "No prose. No markdown. No code fences."
    )

    # -----------------------------------------------------------------------
    # HARD AI-TOOLS CONTENT CONSTRAINT
    # -----------------------------------------------------------------------

    ai_rules = """
AI-TOOLS CONTENT ONLY.

The subject MUST be directly related to artificial intelligence.

Valid subject categories include:
- AI tools
- AI applications
- AI platforms
- AI features
- AI assistants
- AI automation
- AI productivity
- AI coding tools
- AI image tools
- AI video tools
- AI audio tools
- AI writing tools
- AI research tools
- AI design tools
- AI business workflows
- AI content creation
- AI agents
- AI workflows
- practical AI use cases
- AI tips and techniques
- AI tutorials
- AI tool comparisons

The brief should communicate a useful, interesting or visually
demonstrable AI concept.

ABSOLUTELY DO NOT GENERATE:
- romantic content
- couples
- lovers
- dates
- honeymoon scenes
- kissing
- hugging
- wedding scenes
- relationship content
- couple portraits
- male/female romantic interactions
- generic attractive-person lifestyle photography

Do not use a couple as a visual metaphor for AI.

If a person is necessary, use at most a single realistic:
developer, creator, professional, analyst, student, entrepreneur,
designer, researcher or other appropriate technology user.

The technology / AI workflow must remain the visual focus.

Do not generate a generic human portrait when the AI concept can
be represented through a tool interface, workstation, workflow,
dashboard, automation or technology environment.
"""

    # -----------------------------------------------------------------------
    # Location
    # -----------------------------------------------------------------------

    location_section = ""

    if selected_location:
        location_section = (
            "\nAI-TOOLS VISUAL ENVIRONMENT:\n"
            f"Name: {selected_location.get('name', '')}\n"
            f"Description: {selected_location.get('description', '')}\n"
            f"Lighting: {selected_location.get('lighting', '')}\n"
            f"Mood: {selected_location.get('mood', '')}\n"
        )

    # -----------------------------------------------------------------------
    # Visual direction
    # -----------------------------------------------------------------------

    style_section = (
        "\nAI-TOOLS VISUAL DIRECTION:\n"
        f"Framing: {style.get('framing', '')}\n"
        f"Visual format: {style.get('visual_format', '')}\n"
        f"Context: {style.get('context', '')}\n"
        f"Action: {style.get('action', '')}\n"
        f"UI style: {style.get('ui_style', '')}\n"
        f"Composition: {style.get('composition', '')}\n"
        f"Lighting: {style.get('lighting_style', '')}\n"
        f"Color grading: {style.get('color_grading', '')}\n"
        f"Depth of field: {style.get('depth_of_field', '')}\n"
        "\nThe generated scene must visibly follow this direction.\n"
    )

    # -----------------------------------------------------------------------
    # Axis hints
    # -----------------------------------------------------------------------

    hints = "\n".join(
        f"- {key.replace('_', ' ')}: {value}"
        for key, value in axis_hints.items()
    )

    # -----------------------------------------------------------------------
    # History
    # -----------------------------------------------------------------------

    previous = (
        "\n".join(
            f"- {brief.get('subject', '?')} | "
            f"{brief.get('composition', '?')} | "
            f"{brief.get('setting', '?')}"
            for brief in recent_briefs
        )
        or "(none yet)"
    )

    user = (
        f"ACTIVE WORKFLOW: AI TOOLS\n"
        f"STANDING THEME: "
        f"{cfg.active_content.theme}\n"
        f"{ai_rules}"
        f"{location_section}"
        f"{style_section}"
        f"\nPRE-SELECTED CREATIVE CONSTRAINTS:\n"
        f"{hints}\n\n"
        f"RECENTLY USED BRIEFS:\n"
        f"{previous}\n\n"
        f"EDITORIAL INSTRUCTION:\n"
        f"{cfg.active_content.subject_instruction}\n\n"
        "Generate ONE completely fresh AI-tools brief.\n"
        "The subject must be substantially different from every "
        "recent subject.\n"
        "Do not rephrase an existing subject.\n"
        "The visual composition must also differ from recent briefs.\n"
    )

    if error_feedback:
        user += (
            "\n\nPREVIOUS RESPONSE ERROR:\n"
            f"{error_feedback}\n\n"
            "Return valid JSON only."
        )

    return [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": user,
        },
    ]


# ---------------------------------------------------------------------------
# Deterministic scene application
# ---------------------------------------------------------------------------


def _apply_scene(
    brief: Brief,
    selected_location: dict[str, str] | None,
    style: dict[str, str],
) -> Brief:
    """Apply deterministic AI-tools scene variation."""

    if selected_location:
        brief.location_name = (
            selected_location.get(
                "name",
                "",
            )
        )

        if brief.location_name:
            brief.setting = (
                brief.location_name
            )

        location_lighting = (
            selected_location.get(
                "lighting",
                "",
            )
        )

        if location_lighting:
            brief.lighting = (
                location_lighting
            )

    brief.framing = style.get(
        "framing",
        "",
    )

    # Only AI-tools actions are allowed here.
    if style.get("action"):
        brief.interaction = style[
            "action"
        ]

    return brief


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def _fallback_brief(
    cfg: Config,
    axis_hints: dict[str, str],
    seed: int,
    selected_location: dict[str, str] | None,
    style: dict[str, str],
) -> Brief:
    """Create a deterministic AI-tools-only fallback."""

    log.warning(
        "using deterministic AI-tools fallback brief"
    )

    location_name = ""

    if selected_location:
        location_name = (
            selected_location.get(
                "name",
                "",
            )
        )

    brief = Brief(
        subject=(
            f"AI productivity workflow "
            f"variation {seed % 1000}"
        ),
        setting=(
            location_name
            or "modern AI workstation"
        ),
        lighting=(
            axis_hints.get(
                "season",
                "soft",
            )
            + " professional technology light"
        ),
        mood=(
            style.get(
                "context",
                ""
            )
            or "focused and innovative"
        ),
        composition=(
            style.get(
                "composition",
                "",
            )
            or "clean editorial technology composition"
        ),
        color_palette=(
            style.get(
                "color_grading",
                "",
            )
            or "clean modern technology tones"
        ),
        time_of_day=(
            axis_hints.get(
                "time_of_day",
                "morning",
            )
        ),
        style_modifiers=[
            axis_hints.get(
                "focal_length",
                "35mm",
            ),
            "professional technology photography",
            "AI tools editorial visual",
        ],
    )

    return _apply_scene(
        brief,
        selected_location,
        style,
    )


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------


def generate_brief(
    client: OllamaClient,
    cfg: Config,
    seed: int,
    run_date: str,
    history_subjects: list[str],
    recent_briefs: list[dict[str, Any]],
    model: str,
) -> Brief:
    """Generate a fresh AI-tools-only creative brief."""

    rng = random.Random(
        compute_seed(
            run_date,
            cfg.seed_salt,
        )
        ^ seed
    )

    # -----------------------------------------------------------------------
    # Creative axes
    # -----------------------------------------------------------------------

    axis_hints = select_axis_hints(
        rng,
        cfg.brief.axes,
    )

    log.info(
        "AI-tools axis hints: %s",
        axis_hints,
    )

    # -----------------------------------------------------------------------
    # Visual data
    # -----------------------------------------------------------------------

    visual_data = (
        cfg.active_content.visual
    )

    if not isinstance(
        visual_data,
        dict,
    ):
        visual_data = {}

    # -----------------------------------------------------------------------
    # Location rotation
    # -----------------------------------------------------------------------

    raw_locations = visual_data.get(
        "locations",
        {},
    )

    if isinstance(
        raw_locations,
        dict,
    ):
        all_locations = flatten_locations(
            raw_locations
        )
    else:
        all_locations = []

    history_locations = (
        extract_location_from_history(
            recent_briefs
        )
    )

    selected_location = (
        select_unique_location(
            rng,
            all_locations,
            history_locations,
        )
    )

    # -----------------------------------------------------------------------
    # AI visual style
    # -----------------------------------------------------------------------

    style = select_style(
        rng,
        visual_data,
    )

    if selected_location:
        log.info(
            "selected AI-tools environment: %s",
            selected_location.get(
                "name",
                "unknown",
            ),
        )
    else:
        log.warning(
            "no AI-tools locations available"
        )

    log.info(
        "AI-tools visual style: %s",
        {
            key: value
            for key, value in style.items()
            if value
        },
    )

    # -----------------------------------------------------------------------
    # Generate with LLM
    # -----------------------------------------------------------------------

    error_feedback: str | None = None

    for attempt in range(
        1,
        cfg.brief.max_retries + 1,
    ):
        try:
            raw = client.chat_json(
                model=model,
                messages=_build_messages(
                    cfg,
                    axis_hints,
                    recent_briefs,
                    error_feedback,
                    visual_data,
                    selected_location,
                    style,
                ),
                seed=seed + attempt,
                temperature=cfg.llm.temperature,
            )

            brief = Brief.model_validate(
                raw
            )

        except (
            OllamaError,
            ValidationError,
        ) as exc:
            error_feedback = str(
                exc
            )[:300]

            log.warning(
                "AI-tools brief attempt %d/%d invalid: %s",
                attempt,
                cfg.brief.max_retries,
                error_feedback,
            )

            continue

        # ---------------------------------------------------------------
        # Reject repeated subjects
        # ---------------------------------------------------------------

        if is_near_duplicate(
            brief.subject,
            history_subjects,
            cfg.brief.dedupe_threshold,
        ):
            error_feedback = (
                f"subject '{brief.subject}' "
                "is too similar to a recent post; "
                "choose a substantially different AI-tools subject"
            )

            log.warning(
                "AI-tools brief attempt %d rejected as duplicate",
                attempt,
            )

            continue

        # ---------------------------------------------------------------
        # Apply deterministic AI scene
        # ---------------------------------------------------------------

        brief = _apply_scene(
            brief,
            selected_location,
            style,
        )

        log.info(
            "AI-tools brief accepted: %s @ %s",
            brief.subject,
            brief.location_name
            or "(no location)",
        )

        return brief

    # -----------------------------------------------------------------------
    # Deterministic fallback
    # -----------------------------------------------------------------------

    return _fallback_brief(
        cfg,
        axis_hints,
        seed,
        selected_location,
        style,
    )