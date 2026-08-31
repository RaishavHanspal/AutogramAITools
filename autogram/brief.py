"""Educational AI content brief generation.

This module is intentionally dedicated to the AI-tools content workflow.

The goal is NOT to create generic "person using technology" photography.
It generates useful, learnable AI content such as:
  - definitions
  - concept explainers
  - step-by-step workflows
  - architecture diagrams
  - comparisons
  - practical examples
  - AI tools and features
  - agents, RAG, tokens, embeddings, hallucinations, ICL, MCP, etc.

The existing visual fields are retained for compatibility with the image
pipeline, but they now describe educational graphics/diagrams rather than
lifestyle photography.
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
    """Structured educational AI content brief.

    The legacy visual fields remain because the rest of the pipeline may use
    them in image templates. They are deliberately populated with educational
    visual instructions instead of photography/lifestyle instructions.
    """

    subject: str

    # Educational content
    content_type: str = "concept_explainer"
    hook: str = ""
    definition: str = ""
    explanation: str = ""
    steps: list[str] = Field(default_factory=list)
    example: str = ""
    key_takeaway: str = ""
    common_mistake: str = ""
    practical_use: str = ""

    # Image/infographic content
    visual_type: str = "educational_infographic"
    visual_title: str = ""
    visual_elements: list[str] = Field(default_factory=list)
    diagram_flow: list[str] = Field(default_factory=list)
    image_text: list[str] = Field(default_factory=list)

    # Legacy fields kept for pipeline/template compatibility
    setting: str = "clean educational AI infographic"
    lighting: str = "clean high-contrast editorial lighting"
    mood: str = "clear, intelligent and educational"
    composition: str = "information-first vertical infographic composition"
    color_palette: str = "modern professional technology palette"
    time_of_day: str = "not applicable"

    style_modifiers: list[str] = Field(default_factory=list)

    # Deterministic scene metadata retained for compatibility
    location_name: str = ""
    interaction: str = ""
    framing: str = "vertical social-media educational graphic"


# ---------------------------------------------------------------------------
# Deterministic seed / variation
# ---------------------------------------------------------------------------


def compute_seed(run_date: str, salt: str) -> int:
    """Create a deterministic 31-bit seed."""

    digest = hashlib.sha256(
        f"{run_date}|{salt}".encode()
    ).hexdigest()

    return int(digest[:8], 16)


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
# Educational topic pool
# ---------------------------------------------------------------------------

# These are deliberately broad. The LLM chooses the exact angle so the feed
# does not become a repetitive glossary.
_AI_TOPIC_POOL: list[tuple[str, str]] = [
    ("RAG", "retrieval_augmented_generation"),
    ("AI hallucinations", "concept_explainer"),
    ("LLM tokens", "concept_explainer"),
    ("context windows", "concept_explainer"),
    ("embeddings", "concept_explainer"),
    ("vector databases", "concept_explainer"),
    ("semantic search", "concept_explainer"),
    ("reranking", "concept_explainer"),
    ("chunking for RAG", "how_to"),
    ("hybrid search", "concept_explainer"),
    ("fine-tuning vs RAG", "comparison"),
    ("prompt engineering", "how_to"),
    ("zero-shot prompting", "concept_explainer"),
    ("few-shot prompting", "concept_explainer"),
    ("in-context learning", "concept_explainer"),
    ("system prompts", "concept_explainer"),
    ("structured outputs", "how_to"),
    ("temperature in LLMs", "concept_explainer"),
    ("top-p sampling", "concept_explainer"),
    ("attention in transformers", "concept_explainer"),
    ("transformers", "concept_explainer"),
    ("LLM inference", "concept_explainer"),
    ("AI agents", "concept_explainer"),
    ("agent loops", "how_it_works"),
    ("tool calling", "concept_explainer"),
    ("function calling", "concept_explainer"),
    ("AI agent memory", "concept_explainer"),
    ("planning vs execution in agents", "comparison"),
    ("MCP (Model Context Protocol)", "concept_explainer"),
    ("MCP servers and tools", "how_it_works"),
    ("Claude plugins", "how_it_works"),
    ("Claude Code", "tool_explainer"),
    ("coding agents", "tool_explainer"),
    ("AI coding assistants", "comparison"),
    ("AI API vs chatbot", "comparison"),
    ("open-source LLMs", "concept_explainer"),
    ("local LLMs", "how_to"),
    ("Ollama", "tool_explainer"),
    ("Hugging Face inference", "tool_explainer"),
    ("model quantization", "concept_explainer"),
    ("GPU inference", "concept_explainer"),
    ("latency vs throughput", "comparison"),
    ("fine-tuning", "concept_explainer"),
    ("LoRA", "concept_explainer"),
    ("RLHF", "concept_explainer"),
    ("distillation", "concept_explainer"),
    ("multimodal AI", "concept_explainer"),
    ("vision-language models", "concept_explainer"),
    ("AI image generation", "how_it_works"),
    ("diffusion models", "concept_explainer"),
    ("AI video generation", "how_it_works"),
    ("speech-to-text", "how_it_works"),
    ("text-to-speech", "how_it_works"),
    ("AI automation", "how_to"),
    ("AI workflow automation", "how_to"),
    ("AI evaluation", "concept_explainer"),
    ("LLM benchmarks", "concept_explainer"),
    ("grounded generation", "concept_explainer"),
    ("prompt injection", "security_explainer"),
    ("AI security basics", "security_explainer"),
    ("AI data privacy", "security_explainer"),
    ("guardrails for LLMs", "concept_explainer"),
    ("AI observability", "concept_explainer"),
    ("model routing", "concept_explainer"),
    ("AI cost optimization", "how_to"),
    ("caching LLM responses", "concept_explainer"),
    ("AI workflows for developers", "how_to"),
    ("AI research workflow", "how_to"),
    ("AI document processing", "how_to"),
    ("AI knowledge bases", "how_it_works"),
]

_CONTENT_TYPES = [
    "concept_explainer",
    "how_it_works",
    "how_to",
    "comparison",
    "tool_explainer",
    "security_explainer",
    "practical_workflow",
]

_VISUAL_TYPES = [
    "educational_infographic",
    "process_diagram",
    "architecture_diagram",
    "comparison_card",
    "annotated_ui",
    "technical_illustration",
    "step_by_step_card",
    "concept_map",
]

# Topics that particularly benefit from a process/architecture visual.
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
}


# ---------------------------------------------------------------------------
# Subject deduplication
# ---------------------------------------------------------------------------


def _normalize_subject(subject: str) -> str:
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
        previous_normalized = _normalize_subject(previous)

        if fuzz.token_set_ratio(
            normalized,
            previous_normalized,
        ) > threshold:
            return True

    return False


# ---------------------------------------------------------------------------
# Legacy compatibility helpers
# ---------------------------------------------------------------------------


def flatten_locations(
    locations_dict: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    """Flatten configured locations.

    Kept for compatibility. Educational AI posts do not depend on locations.
    """

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
    """Compatibility helper.

    Locations are no longer editorially important. If configured, one may be
    used as a subtle background environment, but never as the subject.
    """

    available = [
        location
        for location in locations
        if location.get("name", "") not in history_locations
    ]

    if not available:
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
        name = brief.get("location_name")

        if name:
            locations.append(str(name))

    return locations


# ---------------------------------------------------------------------------
# Educational visual direction
# ---------------------------------------------------------------------------


def select_framing(rng: random.Random) -> str:
    """Select an educational visual layout."""

    return rng.choice(
        [
            "vertical 4:5 Instagram infographic with a clear visual hierarchy",
            "vertical 4:5 process diagram with large readable labels",
            "vertical 4:5 architecture diagram with connected components",
            "vertical 4:5 comparison card with two clearly separated sides",
            "vertical 4:5 annotated technical illustration",
            "vertical 4:5 step-by-step educational card",
            "vertical 4:5 concept map with a central AI concept and connected ideas",
        ]
    )


def _pick(
    rng: random.Random,
    data: dict[str, Any],
    key: str,
) -> str:
    """Randomly select a configured value."""

    values = data.get(key)

    if isinstance(values, list) and values:
        return str(rng.choice(values))

    return ""


def _pick_trend(
    rng: random.Random,
    trends: dict[str, Any],
    key: str,
) -> str:
    """Compatibility helper for old visual configuration."""

    values = trends.get(key)

    if isinstance(values, list) and values:
        return str(rng.choice(values))

    return ""


def select_style(
    rng: random.Random,
    visual_data: dict[str, Any],
    topic: str = "",
) -> dict[str, str]:
    """Select an educational visual direction."""

    topic_normalized = topic.lower().strip()

    if topic_normalized in _DIAGRAM_TOPICS:
        preferred_visual = rng.choice(
            [
                "process_diagram",
                "architecture_diagram",
                "concept_map",
                "step_by_step_card",
            ]
        )
    else:
        preferred_visual = rng.choice(_VISUAL_TYPES)

    return {
        "framing": select_framing(rng),
        "visual_format": preferred_visual,
        "context": "educational AI concept visualization",
        "action": "show the underlying AI process, system or concept",
        "ui_style": "clean technical infographic with highly legible labels",
        "composition": "information-first hierarchy with one dominant concept",
        "lighting_style": "flat clean presentation lighting",
        "color_grading": "professional modern technology palette",
        "depth_of_field": "sharp readable graphic elements throughout",
    }


# ---------------------------------------------------------------------------
# Image prompt anchor
# ---------------------------------------------------------------------------


def build_character_block(cfg: Config) -> str:
    """Return the configured prompt anchor.

    Retained for compatibility with existing callers. The active AI-tools
    anchor can still provide a general visual identity, but this module does
    not generate character content.
    """

    return cfg.active_content.prompt_anchor.strip()


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def render_prompts(
    brief: Brief,
    cfg: Config,
    characters_block: str = "",
) -> tuple[str, str]:
    """Render positive and negative image prompts.

    The educational content is injected into the legacy template fields so
    existing image-generation code can remain unchanged.
    """

    fields: dict[str, Any] = brief.model_dump()

    fields["style_modifiers"] = ", ".join(brief.style_modifiers)
    fields["characters"] = characters_block

    educational_text = "; ".join(brief.image_text)
    visual_elements = "; ".join(brief.visual_elements)
    diagram_flow = " -> ".join(brief.diagram_flow)

    # These aliases are useful if an existing image template references them.
    fields["educational_text"] = educational_text
    fields["visual_elements_text"] = visual_elements
    fields["diagram"] = diagram_flow

    positive = cfg.image.positive_template.format(**fields)
    negative = cfg.image.negative_template

    positive = re.sub(
        r"(,\s*){2,}",
        ", ",
        positive,
    ).strip(" ,")

    # Strongly steer the image model away from the old lifestyle-photo failure
    # mode regardless of the configured positive template.
    educational_suffix = (
        ", educational AI infographic, technical diagram, "
        "clear typography, readable labels, information-first composition, "
        "no generic lifestyle photography, no person as the main subject"
    )

    if educational_suffix.strip(" ,") not in positive:
        positive += educational_suffix

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
    topic_hint: str,
    content_type_hint: str,
) -> list[dict[str, str]]:
    """Build the educational AI-content LLM request."""

    schema = """
{
  "subject": "specific AI topic",
  "content_type": "concept_explainer|how_it_works|how_to|comparison|tool_explainer|security_explainer|practical_workflow",
  "hook": "short attention-grabbing educational hook",
  "definition": "accurate one or two sentence definition",
  "explanation": "clear explanation of how or why it works",
  "steps": ["step 1", "step 2", "step 3"],
  "example": "concrete simple example",
  "key_takeaway": "one memorable takeaway",
  "common_mistake": "one common misconception or mistake",
  "practical_use": "where this is useful in real life",
  "visual_type": "educational_infographic|process_diagram|architecture_diagram|comparison_card|annotated_ui|technical_illustration|step_by_step_card|concept_map",
  "visual_title": "short title to appear on the image",
  "visual_elements": ["element 1", "element 2"],
  "diagram_flow": ["A", "B", "C"],
  "image_text": ["headline", "definition or key fact", "step/key takeaway"],
  "setting": "clean educational AI infographic environment",
  "lighting": "clean presentation lighting",
  "mood": "clear and educational",
  "composition": "information-first composition",
  "color_palette": "professional technology palette",
  "time_of_day": "not applicable",
  "style_modifiers": ["educational infographic", "clear typography", "technical diagram"]
}
"""

    system = (
        "You are the educational content engine for an AI-learning Instagram "
        "channel. Your job is to teach one useful AI concept per post. "
        "The post must provide genuine learning value, not merely depict AI "
        "visually. "
        f"{cfg.active_content.system_prompt.strip()} "
        "Respond ONLY with one JSON object matching this schema exactly. "
        "No markdown. No prose outside JSON. No code fences.\n\n"
        f"SCHEMA:\n{schema}"
    )

    ai_rules = """
HARD EDITORIAL RULES:

1. EDUCATION FIRST.
The viewer must learn something concrete from the post.

2. ONE CLEAR TOPIC.
Teach one concept, tool, workflow, mechanism, comparison or practical
technique. Do not make a vague "AI is changing the world" post.

3. ACCURACY.
Do not invent product capabilities, APIs, features, technical terminology,
statistics or claims. Explain established concepts accurately. If a topic
depends on a specific current product feature, keep the explanation scoped
to the known concept rather than inventing version-specific behavior.

4. VISUALS MUST TEACH.
The image should normally be an infographic, diagram, architecture flow,
comparison, annotated interface, concept map or step-by-step card.

5. TEXT ON IMAGE IS IMPORTANT.
Provide concise text that can actually appear in the image:
headline, short definition, steps, labels, arrows, or key takeaway.
Do not fill the image with a paragraph.

6. NO GENERIC PEOPLE.
Do not use an attractive person, developer sitting at a laptop, office worker,
business person or generic human as the visual subject.
A person may appear only when genuinely necessary to explain a workflow.

7. NO ROMANCE OR LIFESTYLE CONTENT.
Absolutely no couples, dating, romance, kissing, hugging, wedding scenes,
fashion/editorial portraits or generic lifestyle photography.

8. USEFUL EXAMPLES.
Whenever possible, explain the concept using a small realistic example.

9. HOW-IT-WORKS CONTENT.
For mechanisms such as RAG, agents, embeddings, tokens, MCP or inference,
show the sequence of components and information flow.

10. COMPARISONS.
For comparisons such as RAG vs fine-tuning, clearly state what each is,
when to use each, and the key difference.

11. DEFINITIONS.
Definitions must answer "what is it?" in plain language before going deeper.

12. STEPS.
For tutorials/workflows, produce 3-6 concrete steps.

13. IMAGE TEXT.
Keep each image_text item short enough for a social-media graphic.
Prefer roughly 2-12 words per item.

14. NO FAKE UI.
If showing a software interface, label it as a conceptual/illustrative
interface unless the exact interface is known.

15. DO NOT REPEAT RECENT TOPICS.
A fresh angle is required, not just a rewording of a previous post.
"""

    location_section = ""
    if selected_location:
        location_section = (
            "\nOPTIONAL VISUAL ENVIRONMENT:\n"
            f"{selected_location.get('name', '')}\n"
            f"{selected_location.get('description', '')}\n"
            "Use this only as a subtle background context. The educational "
            "content must remain the dominant visual element.\n"
        )

    style_section = (
        "\nEDUCATIONAL VISUAL DIRECTION:\n"
        f"Layout: {style.get('framing', '')}\n"
        f"Visual type: {style.get('visual_format', '')}\n"
        f"Context: {style.get('context', '')}\n"
        f"UI style: {style.get('ui_style', '')}\n"
        f"Composition: {style.get('composition', '')}\n"
        f"Presentation: {style.get('lighting_style', '')}\n"
        f"Palette: {style.get('color_grading', '')}\n"
        f"Sharpness: {style.get('depth_of_field', '')}\n"
    )

    hints = "\n".join(
        f"- {key.replace('_', ' ')}: {value}"
        for key, value in axis_hints.items()
    )

    previous = "\n".join(
        (
            f"- {brief.get('subject', '?')} | "
            f"{brief.get('content_type', '?')} | "
            f"{brief.get('visual_type', '?')}"
        )
        for brief in recent_briefs
    ) or "(none yet)"

    user = (
        "ACTIVE WORKFLOW: AI EDUCATION / AI TOOLS\n"
        f"STANDING THEME: {cfg.active_content.theme}\n\n"
        f"{ai_rules}\n"
        f"TOPIC HINT: {topic_hint}\n"
        f"CONTENT TYPE HINT: {content_type_hint}\n"
        f"{location_section}"
        f"{style_section}\n"
        f"CREATIVE AXIS HINTS:\n{hints or '(none)'}\n\n"
        f"RECENT POSTS:\n{previous}\n\n"
        "EDITORIAL INSTRUCTION:\n"
        f"{cfg.active_content.subject_instruction}\n\n"
        "Generate ONE genuinely useful educational AI post. "
        "The topic must be substantially different from every recent post. "
        "Do not merely rename or rephrase a recent topic. "
        "The final result should teach something a person can understand "
        "and remember after seeing the post."
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


# ---------------------------------------------------------------------------
# Deterministic educational fallback
# ---------------------------------------------------------------------------


def _fallback_brief(
    cfg: Config,
    seed: int,
    topic: str,
    content_type: str,
) -> Brief:
    """Create a useful deterministic fallback if Ollama fails."""

    fallbacks: dict[str, dict[str, Any]] = {
        "RAG": {
            "definition": (
                "Retrieval-Augmented Generation lets an AI retrieve relevant "
                "information before generating an answer."
            ),
            "explanation": (
                "Instead of relying only on what the model learned during "
                "training, RAG retrieves relevant documents and gives them "
                "to the model as context."
            ),
            "steps": [
                "User asks a question",
                "Question is converted into a search representation",
                "Relevant information is retrieved",
                "Retrieved context is sent to the LLM",
                "LLM generates a grounded answer",
            ],
            "example": (
                "A company chatbot retrieves its internal HR policy before "
                "answering a question about leave."
            ),
            "key_takeaway": "RAG adds external knowledge at inference time.",
            "visual": "process_diagram",
            "flow": [
                "User query",
                "Retrieve relevant data",
                "Context",
                "LLM",
                "Grounded answer",
            ],
        },
        "AI hallucinations": {
            "definition": (
                "An AI hallucination is an answer that sounds plausible but "
                "contains false, unsupported or fabricated information."
            ),
            "explanation": (
                "LLMs generate likely sequences of tokens; they do not "
                "automatically verify every claim against reality."
            ),
            "steps": [
                "Model receives a prompt",
                "Model predicts likely tokens",
                "Generated answer may sound confident",
                "Facts can still be wrong",
            ],
            "example": (
                "An LLM invents a research paper or citation that does not exist."
            ),
            "key_takeaway": "Confidence in wording does not guarantee factual accuracy.",
            "visual": "educational_infographic",
            "flow": [],
        },
        "LLM tokens": {
            "definition": (
                "Tokens are the pieces of text that an LLM processes rather "
                "than reading an entire sentence as one indivisible unit."
            ),
            "explanation": (
                "A token can represent a word, part of a word, punctuation "
                "or another text fragment. Tokenization converts text into "
                "the units the model can process."
            ),
            "steps": [
                "Text is provided",
                "Tokenizer splits it into tokens",
                "Model processes token IDs",
                "Model predicts the next token",
                "Tokens are converted back into text",
            ],
            "example": "A long prompt uses more tokens and consumes more context.",
            "key_takeaway": "LLMs read and generate text through tokens.",
            "visual": "step_by_step_card",
            "flow": ["Text", "Tokens", "Model", "Next tokens", "Text"],
        },
        "embeddings": {
            "definition": (
                "An embedding is a numerical representation of information "
                "that captures semantic relationships."
            ),
            "explanation": (
                "Texts with related meanings tend to have embeddings that are "
                "closer together in vector space."
            ),
            "steps": [
                "Text is sent to an embedding model",
                "The model produces a vector",
                "Vectors can be stored",
                "A query becomes another vector",
                "Similar vectors can be retrieved",
            ],
            "example": (
                "A support system can find documents about password resets "
                "even when the user uses different wording."
            ),
            "key_takeaway": "Embeddings turn meaning into numbers that systems can compare.",
            "visual": "concept_map",
            "flow": ["Text", "Embedding model", "Vector", "Similarity", "Results"],
        },
        "In-context learning": {
            "definition": (
                "In-context learning is when an LLM adapts its response to "
                "examples or instructions included in the current prompt."
            ),
            "explanation": (
                "The model does not need its weights changed; it uses the "
                "information already present in the context to infer the task."
            ),
            "steps": [
                "Give the model an instruction",
                "Provide one or more examples",
                "Ask for a new input",
                "Model follows the demonstrated pattern",
            ],
            "example": "Give three examples of classifying tickets, then provide a fourth.",
            "key_takeaway": "Examples in the prompt can teach a model the expected pattern temporarily.",
            "visual": "process_diagram",
            "flow": ["Instructions", "Examples", "New input", "Model", "Output"],
        },
    }

    data = fallbacks.get(topic)

    if data is None:
        data = {
            "definition": f"{topic} is an important concept in modern AI systems.",
            "explanation": (
                f"This post explains what {topic} means, how it works, "
                "and where it is useful."
            ),
            "steps": [
                f"Understand the core idea of {topic}",
                "See the main components",
                "Follow the information flow",
                "Apply it to a practical example",
            ],
            "example": f"A practical AI workflow can use {topic} as one component.",
            "key_takeaway": f"Understand the mechanism behind {topic}, not just the buzzword.",
            "visual": "educational_infographic",
            "flow": [],
        }

    image_text = [
        topic,
        data["definition"],
        *data["steps"][:4],
        data["key_takeaway"],
    ]

    return Brief(
        subject=topic,
        content_type=content_type,
        hook=f"How does {topic} actually work?",
        definition=data["definition"],
        explanation=data["explanation"],
        steps=data["steps"],
        example=data["example"],
        key_takeaway=data["key_takeaway"],
        common_mistake="Do not treat the AI concept as magic; understand the underlying process.",
        practical_use="Use the concept when designing or evaluating an AI workflow.",
        visual_type=data["visual"],
        visual_title=topic,
        visual_elements=data["steps"][:5],
        diagram_flow=data["flow"],
        image_text=image_text[:7],
        setting="clean educational AI infographic",
        lighting="clean flat presentation lighting",
        mood="clear and educational",
        composition="information-first vertical infographic",
        color_palette="professional modern technology palette",
        time_of_day="not applicable",
        style_modifiers=[
            "educational infographic",
            "technical diagram",
            "clear typography",
            "readable labels",
            "no generic person",
        ],
        location_name="",
        interaction="",
        framing="vertical 4:5 educational social graphic",
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
    """Generate a fresh educational AI brief."""

    rng = random.Random(
        compute_seed(
            run_date,
            cfg.seed_salt,
        ) ^ seed
    )

    # Creative axes remain supported for compatibility, but are no longer
    # allowed to determine the editorial topic.
    axis_hints = select_axis_hints(
        rng,
        cfg.brief.axes,
    )

    topic_candidates = list(_AI_TOPIC_POOL)
    rng.shuffle(topic_candidates)

    # Prefer a topic not seen in recent history before asking the LLM.
    recent_normalized = {
        _normalize_subject(str(item.get("subject", "")))
        for item in recent_briefs
    }
    recent_normalized.update(
        _normalize_subject(subject)
        for subject in history_subjects
    )

    topic_hint = topic_candidates[0][0]
    content_type_hint = topic_candidates[0][1]

    for candidate_topic, candidate_type in topic_candidates:
        if _normalize_subject(candidate_topic) not in recent_normalized:
            topic_hint = candidate_topic
            content_type_hint = candidate_type
            break

    style = select_style(
        rng,
        cfg.active_content.visual
        if isinstance(cfg.active_content.visual, dict)
        else {},
        topic_hint,
    )

    log.info(
        "AI educational topic hint: %s (%s)",
        topic_hint,
        content_type_hint,
    )

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
                    cfg.active_content.visual
                    if isinstance(cfg.active_content.visual, dict)
                    else {},
                    None,
                    style,
                    topic_hint,
                    content_type_hint,
                ),
                seed=seed + attempt,
                temperature=cfg.llm.temperature,
            )

            brief = Brief.model_validate(raw)

        except (
            OllamaError,
            ValidationError,
        ) as exc:
            error_feedback = str(exc)[:500]

            log.warning(
                "AI educational brief attempt %d/%d invalid: %s",
                attempt,
                cfg.brief.max_retries,
                error_feedback,
            )

            continue

        # ---------------------------------------------------------------
        # Validate educational substance
        # ---------------------------------------------------------------

        if len(brief.definition.strip()) < 20:
            error_feedback = (
                "definition is too short; provide a real educational definition"
            )
            continue

        if len(brief.explanation.strip()) < 30:
            error_feedback = (
                "explanation is too short; explain how or why the concept works"
            )
            continue

        if not brief.key_takeaway.strip():
            error_feedback = "key_takeaway is required"
            continue

        if not brief.image_text:
            error_feedback = (
                "image_text is required; the visual must contain useful educational labels"
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
                f"subject '{brief.subject}' is too similar to a recent post; "
                "choose a substantially different AI topic"
            )

            log.warning(
                "AI educational brief attempt %d rejected as duplicate",
                attempt,
            )

            # Pick a different deterministic topic for the retry.
            topic_index = min(
                attempt,
                len(topic_candidates) - 1,
            )
            topic_hint, content_type_hint = topic_candidates[topic_index]

            continue

        # ---------------------------------------------------------------
        # Hard content sanity checks
        # ---------------------------------------------------------------

        subject_lower = brief.subject.lower()

        forbidden = [
            "romance",
            "romantic",
            "couple",
            "lover",
            "honeymoon",
            "kissing",
            "wedding",
            "dating",
        ]

        if any(word in subject_lower for word in forbidden):
            error_feedback = (
                "invalid topic: romantic or relationship content is forbidden"
            )
            continue

        # Force educational visual language even if the model returns
        # photography-oriented wording.
        brief.setting = "clean educational AI infographic"
        brief.time_of_day = "not applicable"
        brief.framing = style["framing"]
        brief.lighting = style["lighting_style"]
        brief.composition = style["composition"]
        brief.color_palette = style["color_grading"]
        brief.mood = "clear, intelligent and educational"
        brief.location_name = ""
        brief.interaction = "show the AI concept, process or information flow"

        if brief.visual_type not in _VISUAL_TYPES:
            brief.visual_type = style["visual_format"]

        brief.style_modifiers = list(
            dict.fromkeys(
                [
                    *brief.style_modifiers,
                    "educational infographic",
                    "technical diagram",
                    "clear readable typography",
                    "information-first composition",
                    "no generic lifestyle photography",
                    "no person as the main subject",
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

    # -----------------------------------------------------------------------
    # Useful deterministic fallback
    # -----------------------------------------------------------------------

    fallback_topic = topic_hint

    # Prefer a known high-value fallback topic instead of a generic phrase.
    if fallback_topic not in {
        "RAG",
        "AI hallucinations",
        "LLM tokens",
        "embeddings",
        "In-context learning",
    }:
        fallback_topic = [
            "RAG",
            "AI hallucinations",
            "LLM tokens",
            "embeddings",
            "In-context learning",
        ][seed % 5]

    return _fallback_brief(
        cfg,
        seed,
        fallback_topic,
        content_type_hint,
    )
