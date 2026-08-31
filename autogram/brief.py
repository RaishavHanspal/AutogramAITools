"""Educational AI content brief generation.

This module is intentionally dedicated to the AI-tools / AI-education
workflow.

The goal is NOT to generate generic AI photography.

Every brief should teach one useful thing about AI:

    definition
    explanation
    steps
    example
    practical use
    common mistake
    key takeaway

The image-generation pipeline then turns that information into an
educational infographic, diagram, comparison card or technical visual.

Legacy visual fields are retained because the rest of Autogram may consume
them. They are populated with educational visual instructions rather than
photography instructions.
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


# ============================================================================
# BRIEF MODEL
# ============================================================================


class Brief(BaseModel):
    """Structured educational AI content brief."""

    # ------------------------------------------------------------------------
    # Core topic
    # ------------------------------------------------------------------------

    subject: str

    # ------------------------------------------------------------------------
    # Educational content
    # ------------------------------------------------------------------------

    content_type: str = "concept_explainer"

    hook: str = ""

    definition: str = ""

    explanation: str = ""

    steps: list[str] = Field(
        default_factory=list
    )

    example: str = ""

    key_takeaway: str = ""

    common_mistake: str = ""

    practical_use: str = ""

    # ------------------------------------------------------------------------
    # Educational visual
    # ------------------------------------------------------------------------

    visual_type: str = "educational_infographic"

    visual_title: str = ""

    visual_elements: list[str] = Field(
        default_factory=list
    )

    diagram_flow: list[str] = Field(
        default_factory=list
    )

    image_text: list[str] = Field(
        default_factory=list
    )

    # ------------------------------------------------------------------------
    # Legacy visual fields
    #
    # These remain intentionally because other pipeline components may expect
    # them. They no longer mean photography in AI-tools mode.
    # ------------------------------------------------------------------------

    setting: str = "clean educational AI infographic"

    lighting: str = "clean presentation lighting"

    mood: str = "clear and educational"

    composition: str = "information-first educational composition"

    color_palette: str = "professional technology palette"

    time_of_day: str = "not applicable"

    style_modifiers: list[str] = Field(
        default_factory=list
    )

    # ------------------------------------------------------------------------
    # Legacy deterministic metadata
    # ------------------------------------------------------------------------

    location_name: str = ""

    interaction: str = ""

    framing: str = "vertical 4:5 educational graphic"


# ============================================================================
# EDUCATIONAL TOPIC LIBRARY
# ============================================================================


_AI_TOPIC_POOL: list[tuple[str, str]] = [

    # ------------------------------------------------------------------------
    # LLM fundamentals
    # ------------------------------------------------------------------------

    ("AI hallucinations", "concept_explainer"),

    ("LLM tokens", "concept_explainer"),

    ("context windows", "concept_explainer"),

    ("embeddings", "concept_explainer"),

    ("attention in transformers", "concept_explainer"),

    ("transformers", "concept_explainer"),

    ("LLM inference", "concept_explainer"),

    ("temperature in LLMs", "concept_explainer"),

    ("top-p sampling", "concept_explainer"),

    ("model context", "concept_explainer"),

    ("next-token prediction", "how_it_works"),

    ("tokenization", "how_it_works"),

    # ------------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------------

    ("prompt engineering", "how_to"),

    ("zero-shot prompting", "concept_explainer"),

    ("few-shot prompting", "concept_explainer"),

    ("in-context learning", "concept_explainer"),

    ("system prompts", "concept_explainer"),

    ("structured outputs", "how_to"),

    ("prompt templates", "how_to"),

    ("prompt chaining", "how_it_works"),

    ("AI instruction hierarchy", "concept_explainer"),

    # ------------------------------------------------------------------------
    # RAG
    # ------------------------------------------------------------------------

    ("RAG", "how_it_works"),

    ("retrieval augmented generation", "concept_explainer"),

    ("chunking for RAG", "how_to"),

    ("vector databases", "concept_explainer"),

    ("semantic search", "concept_explainer"),

    ("hybrid search", "concept_explainer"),

    ("reranking in RAG", "concept_explainer"),

    ("retrieval quality in RAG", "how_to"),

    ("RAG vs fine-tuning", "comparison"),

    ("grounded generation", "concept_explainer"),

    ("knowledge bases for AI", "how_it_works"),

    # ------------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------------

    ("AI agents", "concept_explainer"),

    ("agent loops", "how_it_works"),

    ("tool calling", "concept_explainer"),

    ("function calling", "concept_explainer"),

    ("AI agent memory", "concept_explainer"),

    ("agent planning", "concept_explainer"),

    ("planning vs execution in agents", "comparison"),

    ("single-agent vs multi-agent systems", "comparison"),

    ("AI agent tools", "how_it_works"),

    # ------------------------------------------------------------------------
    # MCP / Claude / coding AI
    # ------------------------------------------------------------------------

    ("MCP (Model Context Protocol)", "concept_explainer"),

    ("MCP servers and tools", "how_it_works"),

    ("MCP resources and prompts", "concept_explainer"),

    ("Claude plugins", "how_it_works"),

    ("Claude Code", "tool_explainer"),

    ("coding agents", "concept_explainer"),

    ("AI coding assistants", "comparison"),

    ("AI-assisted code review", "how_to"),

    ("AI software development workflow", "how_to"),

    # ------------------------------------------------------------------------
    # Models and training
    # ------------------------------------------------------------------------

    ("fine-tuning", "concept_explainer"),

    ("LoRA", "concept_explainer"),

    ("parameter-efficient fine-tuning", "concept_explainer"),

    ("RLHF", "concept_explainer"),

    ("knowledge distillation", "concept_explainer"),

    ("model quantization", "concept_explainer"),

    ("open-source LLMs", "concept_explainer"),

    ("local LLMs", "how_to"),

    ("GPU inference", "concept_explainer"),

    ("latency vs throughput", "comparison"),

    ("model routing", "concept_explainer"),

    # ------------------------------------------------------------------------
    # Multimodal AI
    # ------------------------------------------------------------------------

    ("multimodal AI", "concept_explainer"),

    ("vision-language models", "concept_explainer"),

    ("AI image generation", "how_it_works"),

    ("diffusion models", "concept_explainer"),

    ("AI video generation", "how_it_works"),

    ("speech-to-text", "how_it_works"),

    ("text-to-speech", "how_it_works"),

    # ------------------------------------------------------------------------
    # Tools / platforms
    # ------------------------------------------------------------------------

    ("Ollama", "tool_explainer"),

    ("Hugging Face inference", "tool_explainer"),

    ("AI APIs vs chatbots", "comparison"),

    ("AI model APIs", "concept_explainer"),

    ("AI tool integrations", "how_it_works"),

    # ------------------------------------------------------------------------
    # Automation
    # ------------------------------------------------------------------------

    ("AI automation", "how_to"),

    ("AI workflow automation", "how_to"),

    ("AI document processing", "how_it_works"),

    ("AI research workflow", "how_to"),

    ("AI knowledge extraction", "how_it_works"),

    ("AI workflow orchestration", "concept_explainer"),

    ("LLM response caching", "concept_explainer"),

    ("AI cost optimization", "how_to"),

    # ------------------------------------------------------------------------
    # Evaluation / reliability
    # ------------------------------------------------------------------------

    ("AI evaluation", "concept_explainer"),

    ("LLM benchmarks", "concept_explainer"),

    ("AI evaluation datasets", "concept_explainer"),

    ("grounded AI answers", "how_to"),

    ("AI reliability", "concept_explainer"),

    ("why LLM answers can be inconsistent", "concept_explainer"),

    # ------------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------------

    ("prompt injection", "security_explainer"),

    ("AI security basics", "security_explainer"),

    ("AI data privacy", "security_explainer"),

    ("LLM guardrails", "concept_explainer"),

    ("AI permission boundaries", "security_explainer"),

    ("tool calling security", "security_explainer"),

    # ------------------------------------------------------------------------
    # Practical AI
    # ------------------------------------------------------------------------

    ("AI for developers", "practical_workflow"),

    ("AI for research", "practical_workflow"),

    ("AI for documentation", "practical_workflow"),

    ("AI for data analysis", "practical_workflow"),

    ("AI for customer support", "practical_workflow"),

    ("AI for internal knowledge bases", "practical_workflow"),

    ("AI content workflow", "practical_workflow"),

    ("AI workflow debugging", "how_to"),
]


_VALID_CONTENT_TYPES = {
    "concept_explainer",
    "how_it_works",
    "how_to",
    "comparison",
    "tool_explainer",
    "security_explainer",
    "practical_workflow",
}


_VALID_VISUAL_TYPES = {
    "educational_infographic",
    "process_diagram",
    "architecture_diagram",
    "comparison_card",
    "annotated_ui",
    "technical_illustration",
    "step_by_step_card",
    "concept_map",
}


_DIAGRAM_TOPICS = {
    "rag",
    "retrieval augmented generation",
    "chunking for rag",
    "mcp (model context protocol)",
    "mcp servers and tools",
    "ai agents",
    "agent loops",
    "tool calling",
    "function calling",
    "in-context learning",
    "embeddings",
    "vector databases",
    "semantic search",
    "fine-tuning vs rag",
    "diffusion models",
    "speech-to-text",
    "text-to-speech",
    "prompt chaining",
    "ai workflow automation",
}


# ============================================================================
# SEEDING
# ============================================================================


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
    """Select one value from every configured axis."""

    return {
        axis: rng.choice(values)
        for axis, values in axes.items()
        if values
    }


# ============================================================================
# SUBJECT DEDUPLICATION
# ============================================================================


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

    normalized = _normalize_subject(
        subject
    )

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


# ============================================================================
# LEGACY LOCATION COMPATIBILITY
# ============================================================================


def flatten_locations(
    locations_dict: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    """Flatten configured locations.

    AI education mode treats these as educational canvases rather than
    physical locations.
    """

    locations: list[dict[str, str]] = []

    for scenarios in locations_dict.values():
        if isinstance(
            scenarios,
            list,
        ):
            locations.extend(
                scenarios
            )

    return locations


def select_unique_location(
    rng: random.Random,
    locations: list[dict[str, str]],
    history_locations: list[str],
) -> dict[str, str] | None:
    """Select an unused educational canvas."""

    available = [
        location
        for location in locations
        if location.get(
            "name",
            "",
        )
        not in history_locations
    ]

    if not available:
        available = locations

    if not available:
        return None

    return rng.choice(
        available
    )


def extract_location_from_history(
    history_briefs: list[dict[str, Any]],
) -> list[str]:
    """Extract previously used visual environments."""

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


# ============================================================================
# VISUAL DIRECTION
# ============================================================================


_EDUCATIONAL_FRAMINGS = [
    "vertical 4:5 educational infographic",
    "vertical 4:5 process diagram",
    "vertical 4:5 technical architecture diagram",
    "vertical 4:5 step-by-step explainer",
    "vertical 4:5 comparison card",
    "vertical 4:5 concept map",
    "vertical 4:5 annotated technical graphic",
    "vertical 4:5 input-to-output workflow diagram",
]


def select_framing(
    rng: random.Random,
) -> str:
    """Select an educational graphic layout."""

    return rng.choice(
        _EDUCATIONAL_FRAMINGS
    )


def _pick(
    rng: random.Random,
    data: dict[str, Any],
    key: str,
) -> str:
    """Randomly select a configured value."""

    values = data.get(
        key
    )

    if (
        isinstance(values, list)
        and values
    ):
        return str(
            rng.choice(values)
        )

    return ""


def _pick_trend(
    rng: random.Random,
    trends: dict[str, Any],
    key: str,
) -> str:
    """Select an educational visual-design value."""

    values = trends.get(
        key
    )

    if (
        isinstance(values, list)
        and values
    ):
        return str(
            rng.choice(values)
        )

    return ""


def select_style(
    rng: random.Random,
    visual_data: dict[str, Any],
    topic: str = "",
) -> dict[str, str]:
    """Select an educational AI visual direction."""

    trends = visual_data.get(
        "photography_trends",
        {},
    )

    if not isinstance(
        trends,
        dict,
    ):
        trends = {}

    direction = visual_data.get(
        "ai_tools_visual_direction",
        {},
    )

    if not isinstance(
        direction,
        dict,
    ):
        direction = {}

    topic_normalized = _normalize_subject(
        topic
    )

    if topic_normalized in _DIAGRAM_TOPICS:
        preferred_format = rng.choice(
            [
                "process flow diagram",
                "technical architecture diagram",
                "concept map",
                "step-by-step explainer",
            ]
        )
    else:
        preferred_format = _pick(
            rng,
            direction,
            "formats",
        )

        if not preferred_format:
            preferred_format = rng.choice(
                [
                    "educational infographic",
                    "comparison card",
                    "annotated technical interface",
                    "step-by-step explainer",
                ]
            )

    return {
        "framing": select_framing(
            rng
        ),

        "visual_format": preferred_format,

        "context": (
            _pick(
                rng,
                direction,
                "contexts",
            )
            or "AI concept explanation"
        ),

        "action": (
            _pick(
                rng,
                direction,
                "actions",
            )
            or "show the AI concept and information flow"
        ),

        "ui_style": (
            _pick(
                rng,
                direction,
                "ui_styles",
            )
            or "clean technical infographic"
        ),

        "composition": (
            _pick_trend(
                rng,
                trends,
                "compositions",
            )
            or "information-first infographic"
        ),

        "lighting_style": (
            _pick_trend(
                rng,
                trends,
                "lighting_styles",
            )
            or "clean presentation lighting"
        ),

        "color_grading": (
            _pick_trend(
                rng,
                trends,
                "color_grading",
            )
            or "professional technology palette"
        ),

        "depth_of_field": (
            _pick_trend(
                rng,
                trends,
                "depth_of_field",
            )
            or "sharp readable graphic elements"
        ),
    }


# ============================================================================
# IMAGE PROMPT ANCHOR
# ============================================================================


def build_character_block(
    cfg: Config,
) -> str:
    """Return the configured AI visual anchor.

    Function name is retained for compatibility with existing callers.

    In AI education mode this is NOT a character description.
    """

    return cfg.active_content.prompt_anchor.strip()


# ============================================================================
# IMAGE PROMPT RENDERING
# ============================================================================


def render_prompts(
    brief: Brief,
    cfg: Config,
    characters_block: str = "",
) -> tuple[str, str]:
    """Render the positive and negative image prompts."""

    fields: dict[str, Any] = brief.model_dump()

    fields["style_modifiers"] = ", ".join(
        brief.style_modifiers
    )

    fields["characters"] = (
        characters_block
    )

    fields["visual_elements"] = (
        ", ".join(
            brief.visual_elements
        )
    )

    fields["diagram_flow"] = (
        " -> ".join(
            brief.diagram_flow
        )
    )

    fields["image_text"] = (
        " | ".join(
            brief.image_text
        )
    )

    positive = cfg.image.positive_template.format(
        **fields
    )

    negative = (
        cfg.image.negative_template
    )

    positive = re.sub(
        r"(,\s*){2,}",
        ", ",
        positive,
    ).strip(
        " ,"
    )

    return (
        positive,
        negative,
    )


# ============================================================================
# LLM PROMPT
# ============================================================================


def _build_messages(
    cfg: Config,
    axis_hints: dict[str, str],
    recent_briefs: list[dict[str, Any]],
    error_feedback: str | None,
    visual_data: dict[str, Any],
    selected_location: dict[str, str] | None,
    style: dict[str, str],
    topic_hint: str,
    content_type_hint: str,
) -> list[dict[str, str]]:
    """Build the educational AI LLM request."""

    schema = """
{
  "subject": "specific AI topic",
  "content_type": "concept_explainer|how_it_works|how_to|comparison|tool_explainer|security_explainer|practical_workflow",
  "hook": "short educational hook",
  "definition": "accurate plain-language definition",
  "explanation": "clear explanation of how or why it works",
  "steps": [
    "step 1",
    "step 2",
    "step 3"
  ],
  "example": "specific practical example",
  "key_takeaway": "one memorable takeaway",
  "common_mistake": "one common misconception or mistake",
  "practical_use": "where this is useful",
  "visual_type": "educational_infographic|process_diagram|architecture_diagram|comparison_card|annotated_ui|technical_illustration|step_by_step_card|concept_map",
  "visual_title": "short title for the image",
  "visual_elements": [
    "important visual element 1",
    "important visual element 2"
  ],
  "diagram_flow": [
    "component A",
    "component B",
    "component C"
  ],
  "image_text": [
    "short headline",
    "short definition",
    "short step or key fact",
    "short takeaway"
  ],
  "setting": "clean educational AI infographic",
  "lighting": "clean presentation lighting",
  "mood": "clear and educational",
  "composition": "information-first composition",
  "color_palette": "professional technology palette",
  "time_of_day": "not applicable",
  "style_modifiers": [
    "educational infographic",
    "technical diagram",
    "clear typography"
  ]
}
"""

    system = (
        "You are the educational content engine for an AI-learning "
        "Instagram publication. "
        f"{cfg.active_content.system_prompt.strip()} "
        "\n\nReturn ONLY one valid JSON object matching this schema. "
        "No markdown. No code fences. No commentary outside JSON.\n\n"
        f"SCHEMA:\n{schema}"
    )

    ai_rules = """
HARD EDITORIAL RULES

1. EDUCATION FIRST.

The viewer must learn something concrete.

Do not create an image merely showing AI.

The content itself must explain an AI concept, tool, workflow, mechanism,
definition, comparison or practical technique.

2. ONE CLEAR TOPIC.

Teach one subject per post.

3. ACCURACY.

Explain established AI concepts accurately.

Do not invent product features, APIs, statistics, benchmarks or capabilities.

If a product-specific fact is uncertain, keep the explanation conceptual.

4. USEFULNESS.

The post should help someone understand or use AI.

Prefer:
- definitions
- explanations
- practical examples
- workflows
- architecture
- comparisons
- step-by-step tutorials
- common mistakes
- useful mental models

5. IMAGE MUST TEACH.

The image should normally be:
- infographic
- process diagram
- architecture diagram
- comparison card
- concept map
- annotated technical interface
- step-by-step educational graphic

6. IMAGE TEXT IS REQUIRED.

Provide concise image_text.

Each item should normally be short enough to render on an Instagram graphic.

Do NOT put a giant paragraph into image_text.

7. NO GENERIC PEOPLE.

Do not create:
- attractive people
- developers sitting at laptops
- office workers
- business people
- generic programmers
- generic creator workspaces
- generic AI robots

A human should only appear if the person is genuinely required to explain
the concept.

8. NO LIFESTYLE CONTENT.

Absolutely no:
- romance
- couples
- dating
- kissing
- hugging
- wedding
- honeymoon
- fashion
- cinematic portraits
- generic lifestyle scenes

9. MECHANISMS NEED FLOWS.

For concepts such as RAG, agents, MCP, embeddings, inference,
tokenization and diffusion, diagram_flow should describe the actual
information or processing flow.

10. COMPARISONS NEED A REAL DIFFERENCE.

For example:

RAG:
retrieves external information at runtime.

Fine-tuning:
changes model behavior through additional training.

Do not produce empty "A vs B" content.

11. DEFINITIONS MUST BE SIMPLE.

First answer:
"What is this?"

Then explain:
"How does it work?"

12. PRACTICAL EXAMPLES.

Whenever appropriate, give one realistic example.

13. TUTORIALS.

For how_to or practical_workflow content, provide 3-6 concrete steps.

14. COMMON MISTAKES.

Include one useful misconception or implementation mistake whenever
appropriate.

15. NO GENERIC AI HYPE.

Never generate:
"AI is changing everything."
"The future of AI is here."
"Unlock the power of AI."
"AI will revolutionize your life."

Unless the phrase is part of a specific educational explanation.

16. NO PHOTOGRAPHY LANGUAGE.

Do not make photography the subject.

The visual should communicate information.

17. VISUAL HIERARCHY.

One dominant concept.

Use:
- boxes
- arrows
- labels
- concise text
- simple relationships
- visual grouping

18. NO FAKE PRODUCT UI.

Do not fabricate exact product interfaces.

If an interface is shown conceptually, describe it as an illustrative
technical interface.

19. TOPIC VARIETY.

Do not repeat a recent topic or merely rename it.

20. THE POST MUST STAND ALONE.

A viewer should understand the main idea without needing the caption.
"""

    location_section = ""

    if selected_location:
        location_section = (
            "\nEDUCATIONAL VISUAL CANVAS:\n"
            f"Name: {selected_location.get('name', '')}\n"
            f"Description: {selected_location.get('description', '')}\n"
            "This is a graphic canvas, NOT a physical workplace. "
            "The educational information remains the dominant element.\n"
        )

    style_section = (
        "\nEDUCATIONAL VISUAL DIRECTION:\n"
        f"Framing: {style.get('framing', '')}\n"
        f"Format: {style.get('visual_format', '')}\n"
        f"Context: {style.get('context', '')}\n"
        f"Action: {style.get('action', '')}\n"
        f"UI style: {style.get('ui_style', '')}\n"
        f"Composition: {style.get('composition', '')}\n"
        f"Presentation: {style.get('lighting_style', '')}\n"
        f"Palette: {style.get('color_grading', '')}\n"
        f"Clarity: {style.get('depth_of_field', '')}\n"
    )

    hints = "\n".join(
        f"- {key.replace('_', ' ')}: {value}"
        for key, value in axis_hints.items()
    )

    previous = "\n".join(
        (
            f"- subject: {brief.get('subject', '?')} | "
            f"type: {brief.get('content_type', '?')} | "
            f"visual: {brief.get('visual_type', '?')}"
        )
        for brief in recent_briefs
    )

    if not previous:
        previous = "(none)"

    user = (
        "ACTIVE WORKFLOW: AI EDUCATION\n\n"
        f"STANDING THEME:\n"
        f"{cfg.active_content.theme}\n\n"
        f"{ai_rules}\n\n"
        f"TOPIC HINT:\n"
        f"{topic_hint}\n\n"
        f"CONTENT TYPE HINT:\n"
        f"{content_type_hint}\n\n"
        f"{location_section}\n"
        f"{style_section}\n\n"
        "CREATIVE CONSTRAINTS:\n"
        f"{hints or '(none)'}\n\n"
        "RECENT POSTS:\n"
        f"{previous}\n\n"
        "EDITORIAL INSTRUCTION:\n"
        f"{cfg.active_content.subject_instruction}\n\n"
        "TASK:\n"
        "Generate ONE genuinely useful educational AI post.\n"
        "The viewer should learn a specific thing.\n"
        "Prefer a concrete concept over a generic tool advertisement.\n"
        "The image must communicate the lesson visually.\n"
        "The subject must be substantially different from recent subjects.\n"
        "Return valid JSON only."
    )

    if error_feedback:
        user += (
            "\n\nPREVIOUS RESPONSE ERROR:\n"
            f"{error_feedback}\n\n"
            "Correct the problem and return valid JSON only."
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


# ============================================================================
# CONTENT VALIDATION
# ============================================================================


_FORBIDDEN_TERMS = {
    "romance",
    "romantic",
    "couple",
    "couples",
    "lover",
    "lovers",
    "dating",
    "honeymoon",
    "kissing",
    "kiss",
    "hugging",
    "wedding",
}


def _contains_forbidden_content(
    brief: Brief,
) -> bool:
    """Reject relationship/lifestyle output."""

    searchable = " ".join(
        [
            brief.subject,
            brief.hook,
            brief.definition,
            brief.explanation,
            brief.example,
            brief.visual_title,
            brief.setting,
            brief.mood,
        ]
        + brief.visual_elements
        + brief.image_text
    ).lower()

    return any(
        term in searchable
        for term in _FORBIDDEN_TERMS
    )


def _has_educational_content(
    brief: Brief,
) -> bool:
    """Ensure the brief actually contains useful information."""

    if len(
        brief.definition.strip()
    ) < 25:
        return False

    if len(
        brief.explanation.strip()
    ) < 40:
        return False

    if not brief.key_takeaway.strip():
        return False

    if not brief.image_text:
        return False

    return True


def _normalize_content_type(
    value: str,
) -> str:
    """Return a valid content type."""

    if value in _VALID_CONTENT_TYPES:
        return value

    return "concept_explainer"


def _normalize_visual_type(
    value: str,
) -> str:
    """Return a valid visual type."""

    if value in _VALID_VISUAL_TYPES:
        return value

    return "educational_infographic"


# ============================================================================
# FALLBACK CONTENT
# ============================================================================


_FALLBACKS: dict[str, dict[str, Any]] = {

    "RAG": {
        "content_type": "how_it_works",

        "definition": (
            "Retrieval-Augmented Generation, or RAG, lets an AI retrieve "
            "relevant external information before generating an answer."
        ),

        "explanation": (
            "Instead of relying only on information stored in the model's "
            "parameters, a RAG system searches a connected knowledge source, "
            "retrieves relevant content, and provides that content to the "
            "language model as context."
        ),

        "steps": [
            "User asks a question",
            "The question is converted into a searchable representation",
            "Relevant documents or chunks are retrieved",
            "Retrieved context is sent to the LLM",
            "The LLM generates an answer using that context",
        ],

        "example": (
            "A company chatbot retrieves the latest internal HR policy "
            "before answering a question about annual leave."
        ),

        "key_takeaway": (
            "RAG gives an LLM relevant external knowledge at runtime; "
            "it does not retrain the model."
        ),

        "common_mistake": (
            "Assuming RAG automatically makes answers accurate even when "
            "retrieval returns poor or irrelevant information."
        ),

        "practical_use": (
            "Use RAG when an AI system needs to answer questions using "
            "documents, databases or knowledge that can change over time."
        ),

        "visual_type": "process_diagram",

        "visual_title": "How RAG Works",

        "visual_elements": [
            "User query",
            "Retriever",
            "Relevant documents",
            "Context",
            "LLM",
            "Grounded answer",
        ],

        "diagram_flow": [
            "User question",
            "Retrieve relevant information",
            "Add retrieved context",
            "Send context to LLM",
            "Generate answer",
        ],

        "image_text": [
            "How RAG Works",
            "Retrieve knowledge before generating",
            "1. Ask",
            "2. Retrieve",
            "3. Add context",
            "4. Generate",
            "RAG ≠ retraining",
        ],
    },

    "AI hallucinations": {
        "content_type": "concept_explainer",

        "definition": (
            "An AI hallucination is an answer that sounds plausible but "
            "contains false, unsupported or fabricated information."
        ),

        "explanation": (
            "Language models generate likely sequences of tokens based on "
            "their learned patterns and the current context. They do not "
            "automatically verify every claim against reality."
        ),

        "steps": [
            "The model receives a prompt",
            "It predicts likely token sequences",
            "The output can sound confident",
            "Some claims may still be false",
        ],

        "example": (
            "An LLM invents a research paper, citation or API function "
            "that does not actually exist."
        ),

        "key_takeaway": (
            "Confident wording is not proof that an AI answer is factual."
        ),

        "common_mistake": (
            "Treating an AI response as verified simply because it sounds "
            "specific and confident."
        ),

        "practical_use": (
            "Use verification, retrieval, citations or trusted source data "
            "when factual accuracy matters."
        ),

        "visual_type": "educational_infographic",

        "visual_title": "Why AI Hallucinates",

        "visual_elements": [
            "Prompt",
            "Token prediction",
            "Plausible output",
            "Possible false claim",
            "Verification",
        ],

        "diagram_flow": [
            "Prompt",
            "Token prediction",
            "Generated answer",
            "Verify claims",
        ],

        "image_text": [
            "AI Hallucinations",
            "Plausible ≠ factual",
            "LLMs predict text",
            "Claims can still be wrong",
            "Verify important information",
        ],
    },

    "LLM tokens": {
        "content_type": "concept_explainer",

        "definition": (
            "Tokens are the pieces of text that language models process "
            "when reading and generating language."
        ),

        "explanation": (
            "A token can represent a whole word, part of a word, punctuation "
            "or another text fragment. Tokenization converts text into the "
            "units the model can process."
        ),

        "steps": [
            "Text is provided to the model",
            "A tokenizer splits the text into tokens",
            "Tokens are represented as IDs",
            "The model processes those IDs",
            "The model predicts additional tokens",
        ],

        "example": (
            "A long prompt can consume thousands of tokens even though it "
            "looks like only a few paragraphs to a person."
        ),

        "key_takeaway": (
            "LLMs read and generate language through tokens, not whole "
            "sentences as indivisible units."
        ),

        "common_mistake": (
            "Assuming one token always equals one English word."
        ),

        "practical_use": (
            "Understanding tokens helps explain context limits, API pricing "
            "and why very long prompts can become expensive."
        ),

        "visual_type": "step_by_step_card",

        "visual_title": "How LLM Tokens Work",

        "visual_elements": [
            "Text",
            "Tokenizer",
            "Token IDs",
            "LLM",
            "Predicted tokens",
        ],

        "diagram_flow": [
            "Text",
            "Tokenization",
            "Token IDs",
            "LLM",
            "Next-token prediction",
        ],

        "image_text": [
            "What Is a Token?",
            "Text → Tokens → Model",
            "Tokens can be word parts",
            "The model predicts the next token",
            "Tokens affect context & cost",
        ],
    },

    "embeddings": {
        "content_type": "concept_explainer",

        "definition": (
            "An embedding is a numerical representation of information "
            "designed to capture useful semantic relationships."
        ),

        "explanation": (
            "An embedding model converts text or other data into a vector. "
            "Items with related meaning can have vectors that are closer "
            "together, allowing systems to perform semantic similarity "
            "search."
        ),

        "steps": [
            "Convert text into an embedding",
            "Store the resulting vector",
            "Convert a user query into another vector",
            "Compare vector similarity",
            "Retrieve the closest matches",
        ],

        "example": (
            "A support system can find documents about resetting a password "
            "even when the user's wording does not exactly match the document."
        ),

        "key_takeaway": (
            "Embeddings turn semantic meaning into numbers that AI systems "
            "can compare."
        ),

        "common_mistake": (
            "Thinking embeddings are the same thing as tokens. Tokens are "
            "model input units; embeddings represent information numerically."
        ),

        "practical_use": (
            "Embeddings are commonly used for semantic search, RAG, "
            "recommendations and similarity matching."
        ),

        "visual_type": "concept_map",

        "visual_title": "What Are Embeddings?",

        "visual_elements": [
            "Text",
            "Embedding model",
            "Vector",
            "Similarity",
            "Related content",
        ],

        "diagram_flow": [
            "Text",
            "Embedding model",
            "Vector representation",
            "Similarity search",
            "Relevant results",
        ],

        "image_text": [
            "Embeddings",
            "Turn meaning into numbers",
            "Text → Vector",
            "Similar meaning → closer vectors",
            "Used in semantic search & RAG",
        ],
    },

    "In-context learning": {
        "content_type": "concept_explainer",

        "definition": (
            "In-context learning is when an LLM uses instructions or "
            "examples inside the current prompt to perform a task."
        ),

        "explanation": (
            "The model does not need its weights changed. Instead, it uses "
            "the information present in the current context to infer the "
            "pattern or behavior expected by the prompt."
        ),

        "steps": [
            "Give the model an instruction",
            "Provide one or more examples",
            "Provide a new input",
            "The model infers the demonstrated pattern",
            "The model generates the requested output",
        ],

        "example": (
            "Give an LLM three examples of classifying customer messages "
            "and then ask it to classify a fourth message."
        ),

        "key_takeaway": (
            "Examples in a prompt can temporarily teach a model the expected "
            "pattern without changing the model's weights."
        ),

        "common_mistake": (
            "Calling every prompt-based behavior fine-tuning. In-context "
            "learning does not update model parameters."
        ),

        "practical_use": (
            "Use examples in prompts when you need consistent formatting, "
            "classification or task behavior without training a model."
        ),

        "visual_type": "process_diagram",

        "visual_title": "In-Context Learning",

        "visual_elements": [
            "Instruction",
            "Examples",
            "New input",
            "LLM",
            "Output",
        ],

        "diagram_flow": [
            "Instructions",
            "Examples",
            "New input",
            "LLM infers pattern",
            "Output",
        ],

        "image_text": [
            "In-Context Learning",
            "Teach through the prompt",
            "1. Instruction",
            "2. Examples",
            "3. New input",
            "No weight update",
        ],
    },
}


def _fallback_brief(
    cfg: Config,
    seed: int,
    topic: str,
    content_type: str,
) -> Brief:
    """Create a useful deterministic educational fallback."""

    data = _FALLBACKS.get(
        topic
    )

    if data is None:
        data = {
            "content_type": content_type,

            "definition": (
                f"{topic} is an important concept in modern AI systems."
            ),

            "explanation": (
                f"This educational post explains what {topic} means, "
                "how it works and why it is useful."
            ),

            "steps": [
                f"Understand the core idea of {topic}",
                "Identify the main components",
                "Follow the information flow",
                "Apply it to a practical example",
            ],

            "example": (
                f"A practical AI system can use {topic} as one part "
                "of an AI workflow."
            ),

            "key_takeaway": (
                f"Understand how {topic} works rather than memorizing "
                "the buzzword."
            ),

            "common_mistake": (
                "Focusing on the terminology without understanding "
                "the underlying mechanism."
            ),

            "practical_use": (
                "Use the concept when designing, building or evaluating "
                "AI workflows."
            ),

            "visual_type": "educational_infographic",

            "visual_title": topic,

            "visual_elements": [
                topic,
                "Core concept",
                "How it works",
                "Practical use",
            ],

            "diagram_flow": [
                "Input",
                "AI process",
                "Output",
            ],

            "image_text": [
                topic,
                "What is it?",
                "How does it work?",
                "Why does it matter?",
            ],
        }

    return Brief(
        subject=topic,

        content_type=data[
            "content_type"
        ],

        hook=f"How does {topic} actually work?",

        definition=data[
            "definition"
        ],

        explanation=data[
            "explanation"
        ],

        steps=data[
            "steps"
        ],

        example=data[
            "example"
        ],

        key_takeaway=data[
            "key_takeaway"
        ],

        common_mistake=data[
            "common_mistake"
        ],

        practical_use=data[
            "practical_use"
        ],

        visual_type=data[
            "visual_type"
        ],

        visual_title=data[
            "visual_title"
        ],

        visual_elements=data[
            "visual_elements"
        ],

        diagram_flow=data[
            "diagram_flow"
        ],

        image_text=data[
            "image_text"
        ],

        setting="clean educational AI infographic",

        lighting="clean presentation lighting",

        mood="clear and educational",

        composition="information-first educational composition",

        color_palette="professional technology palette",

        time_of_day="not applicable",

        style_modifiers=[
            "educational infographic",
            "technical diagram",
            "clear readable typography",
            "information-first composition",
            "useful labels",
            "no generic people",
            "no lifestyle photography",
        ],

        location_name="",

        interaction="show the AI concept and information flow",

        framing="vertical 4:5 educational graphic",
    )


# ============================================================================
# MAIN GENERATOR
# ============================================================================


def generate_brief(
    client: OllamaClient,
    cfg: Config,
    seed: int,
    run_date: str,
    history_subjects: list[str],
    recent_briefs: list[dict[str, Any]],
    model: str,
) -> Brief:
    """Generate a fresh educational AI brief."""

    rng = random.Random(
        compute_seed(
            run_date,
            cfg.seed_salt,
        )
        ^ seed
    )

    # ------------------------------------------------------------------------
    # Seeded topic selection
    # ------------------------------------------------------------------------

    topic_candidates = list(
        _AI_TOPIC_POOL
    )

    rng.shuffle(
        topic_candidates
    )

    recent_subjects = {
        _normalize_subject(
            str(
                subject
            )
        )
        for subject in history_subjects
    }

    recent_subjects.update(
        _normalize_subject(
            str(
                item.get(
                    "subject",
                    "",
                )
            )
        )
        for item in recent_briefs
    )

    topic_hint = topic_candidates[0][0]

    content_type_hint = topic_candidates[0][1]

    for candidate_topic, candidate_type in topic_candidates:

        if (
            _normalize_subject(
                candidate_topic
            )
            not in recent_subjects
        ):
            topic_hint = candidate_topic

            content_type_hint = candidate_type

            break

    log.info(
        "AI educational topic hint: %s (%s)",
        topic_hint,
        content_type_hint,
    )

    # ------------------------------------------------------------------------
    # Visual configuration
    # ------------------------------------------------------------------------

    visual_data = cfg.active_content.visual

    if not isinstance(
        visual_data,
        dict,
    ):
        visual_data = {}

    # ------------------------------------------------------------------------
    # Educational canvas rotation
    # ------------------------------------------------------------------------

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

    history_locations = extract_location_from_history(
        recent_briefs
    )

    selected_location = select_unique_location(
        rng,
        all_locations,
        history_locations,
    )

    # ------------------------------------------------------------------------
    # Educational visual style
    # ------------------------------------------------------------------------

    style = select_style(
        rng,
        visual_data,
        topic_hint,
    )

    log.info(
        "AI educational visual style: %s",
        {
            key: value
            for key, value in style.items()
            if value
        },
    )

    # ------------------------------------------------------------------------
    # Compatibility axis hints
    # ------------------------------------------------------------------------

    axis_hints = select_axis_hints(
        rng,
        cfg.brief.axes,
    )

    # ------------------------------------------------------------------------
    # LLM attempts
    # ------------------------------------------------------------------------

    error_feedback: str | None = None

    for attempt in range(
        1,
        cfg.brief.max_retries + 1,
    ):

        try:

            raw = client.chat_json(
                model=model,

                messages=_build_messages(
                    cfg=cfg,
                    axis_hints=axis_hints,
                    recent_briefs=recent_briefs,
                    error_feedback=error_feedback,
                    visual_data=visual_data,
                    selected_location=selected_location,
                    style=style,
                    topic_hint=topic_hint,
                    content_type_hint=content_type_hint,
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
            TypeError,
            ValueError,
        ) as exc:

            error_feedback = str(
                exc
            )[:500]

            log.warning(
                "AI educational brief attempt %d/%d invalid: %s",
                attempt,
                cfg.brief.max_retries,
                error_feedback,
            )

            continue

        # --------------------------------------------------------------------
        # Normalize values
        # --------------------------------------------------------------------

        brief.content_type = _normalize_content_type(
            brief.content_type
        )

        brief.visual_type = _normalize_visual_type(
            brief.visual_type
        )

        # --------------------------------------------------------------------
        # Educational substance gate
        # --------------------------------------------------------------------

        if not _has_educational_content(
            brief
        ):

            error_feedback = (
                "The brief does not contain enough educational information. "
                "Provide a real definition, explanation, key takeaway and "
                "useful image text."
            )

            log.warning(
                "AI educational brief rejected: insufficient educational content"
            )

            continue

        # --------------------------------------------------------------------
        # Forbidden content gate
        # --------------------------------------------------------------------

        if _contains_forbidden_content(
            brief
        ):

            error_feedback = (
                "Forbidden lifestyle/romantic content detected. "
                "Generate an AI educational topic only."
            )

            log.warning(
                "AI educational brief rejected: forbidden content"
            )

            continue

        # --------------------------------------------------------------------
        # Duplicate gate
        # --------------------------------------------------------------------

        if is_near_duplicate(
            brief.subject,
            history_subjects,
            cfg.brief.dedupe_threshold,
        ):

            error_feedback = (
                f"Subject '{brief.subject}' is too similar to a recent post. "
                "Choose a substantially different AI topic."
            )

            log.warning(
                "AI educational brief attempt %d rejected as duplicate",
                attempt,
            )

            # Move the topic hint for the next attempt.
            topic_index = min(
                attempt,
                len(topic_candidates) - 1,
            )

            topic_hint = topic_candidates[
                topic_index
            ][0]

            content_type_hint = topic_candidates[
                topic_index
            ][1]

            continue

        # --------------------------------------------------------------------
        # Force safe educational visual fields
        # --------------------------------------------------------------------

        brief.setting = (
            "clean educational AI infographic"
        )

        brief.lighting = (
            style.get(
                "lighting_style"
            )
            or "clean presentation lighting"
        )

        brief.mood = (
            "clear and educational"
        )

        brief.composition = (
            style.get(
                "composition"
            )
            or "information-first educational composition"
        )

        brief.color_palette = (
            style.get(
                "color_grading"
            )
            or "professional technology palette"
        )

        brief.time_of_day = (
            "not applicable"
        )

        brief.location_name = ""

        brief.interaction = (
            style.get(
                "action"
            )
            or "show the AI concept and information flow"
        )

        brief.framing = (
            style.get(
                "framing"
            )
            or "vertical 4:5 educational graphic"
        )

        # --------------------------------------------------------------------
        # Force useful visual metadata
        # --------------------------------------------------------------------

        if not brief.visual_title.strip():
            brief.visual_title = (
                brief.subject
            )

        if not brief.visual_elements:
            brief.visual_elements = (
                brief.steps[:5]
            )

        if not brief.image_text:

            brief.image_text = [
                brief.visual_title,
                brief.definition,
                brief.key_takeaway,
            ]

        if (
            brief.visual_type
            in {
                "process_diagram",
                "architecture_diagram",
                "concept_map",
            }
            and not brief.diagram_flow
        ):

            brief.diagram_flow = (
                brief.steps[:5]
            )

        # --------------------------------------------------------------------
        # Force visual style modifiers
        # --------------------------------------------------------------------

        required_modifiers = [
            "educational infographic",
            "technical diagram",
            "clear readable typography",
            "information-first composition",
            "useful labels",
            "sharp graphic elements",
            "no generic people",
            "no lifestyle photography",
        ]

        brief.style_modifiers = list(
            dict.fromkeys(
                [
                    *brief.style_modifiers,
                    *required_modifiers,
                ]
            )
        )

        log.info(
            "AI educational brief accepted: %s | type=%s | visual=%s",
            brief.subject,
            brief.content_type,
            brief.visual_type,
        )

        return brief

    # ------------------------------------------------------------------------
    # Deterministic fallback
    #
    # IMPORTANT:
    #
    # Never fall back to "AI productivity workflow variation 123".
    # That was one of the reasons the system could produce generic content.
    # ------------------------------------------------------------------------

    fallback_topic = topic_hint

    known_fallbacks = list(
        _FALLBACKS.keys()
    )

    if fallback_topic not in known_fallbacks:

        fallback_topic = known_fallbacks[
            seed % len(
                known_fallbacks
            )
        ]

    log.warning(
        "Using deterministic educational fallback: %s",
        fallback_topic,
    )

    return _fallback_brief(
        cfg=cfg,
        seed=seed,
        topic=fallback_topic,
        content_type=content_type_hint,
    )
