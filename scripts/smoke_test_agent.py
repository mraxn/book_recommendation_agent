from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from backend.app.schemas import ChatMessage  # noqa: E402
from backend.app.services.book_agent import run_book_agent, stream_book_agent_response  # noqa: E402


@dataclass(frozen=True)
class SmokeScenario:
    name: str
    prompt: str
    follow_up: str | None = None


SMOKE_SCENARIOS = [
    SmokeScenario(
        name="gothic_science_before_1900",
        prompt=(
            "Recommend 4 English books written before 1900 that combine gothic atmosphere, "
            "moral conflict, and science or forbidden knowledge."
        ),
    ),
    SmokeScenario(
        name="twain_travel",
        prompt=(
            "Find books by Mark Twain that are connected to travel, journeys, or observations "
            "about foreign places. Prefer popular works."
        ),
    ),
    SmokeScenario(
        name="like_frankenstein",
        prompt=(
            "Suggest something like Frankenstein, but do not recommend Frankenstein itself. "
            "I want gothic mood, science, ambition, and ethical consequences."
        ),
    ),
    SmokeScenario(
        name="french_revolution",
        prompt=(
            "Find French-language books about political revolution, social unrest, or rebellion, "
            "preferably older classics."
        ),
    ),
    SmokeScenario(
        name="philosophical_guilt_before_1900",
        prompt=(
            "Recommend one strong philosophical novel or story before 1900 about guilt, moral "
            "responsibility, or the consequences of obsession."
        ),
    ),
    SmokeScenario(
        name="teen_survival_adventure",
        prompt=(
            "I want adventure books for a teenager who likes survival, wilderness, shipwrecks, "
            "danger, and fast-moving plots. Prefer English books."
        ),
    ),
    SmokeScenario(
        name="shakespeare_power_betrayal",
        prompt=(
            "Find Shakespeare works that involve political power, betrayal, ambition, or murder. "
            "Recommend the best 3 matches."
        ),
        follow_up="More like the second one, but darker and more focused on ambition.",
    ),
    SmokeScenario(
        name="count_of_monte_cristo",
        prompt=(
            "Suggest books for someone who loved The Count of Monte Cristo, especially revenge, "
            "imprisonment, justice, disguise, and long-term plotting."
        ),
        follow_up=(
            "Give me only one from those that feels most like a revenge story, and explain "
            "briefly why."
        ),
    ),
    SmokeScenario(
        name="ghost_horror_before_1920",
        prompt=(
            "Find English ghost or supernatural horror books before 1920, but avoid anything "
            "that sounds comedic or light."
        ),
        follow_up="Now make it more psychological and less about literal ghosts.",
    ),
    SmokeScenario(
        name="social_class_satire",
        prompt=(
            "Recommend popular classic books about social class, wealth, marriage, reputation, "
            "or hypocrisy, ideally with sharp social criticism."
        ),
    ),
]


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run live smoke prompts against the book agent.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Skip the configured LLM provider and use deterministic fallback answers.",
    )
    parser.add_argument(
        "--no-follow-ups",
        action="store_true",
        help="Run only the first prompt in each scenario.",
    )
    parser.add_argument(
        "--only",
        choices=[scenario.name for scenario in SMOKE_SCENARIOS],
        help="Run one named scenario.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N scenarios after applying --only.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Maximum characters to print per answer.",
    )
    args = parser.parse_args()

    load_dotenv()
    scenarios = [scenario for scenario in SMOKE_SCENARIOS if args.only in {None, scenario.name}]
    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    failures = 0
    for index, scenario in enumerate(scenarios, start=1):
        print(f"\n=== {index}. {scenario.name} ===")
        messages = [ChatMessage(role="user", content=scenario.prompt)]
        try:
            answer = await _ask(messages, deterministic=args.deterministic)
            _print_turn("Prompt", scenario.prompt, answer, args.max_chars)

            if scenario.follow_up and not args.no_follow_ups:
                messages.append(ChatMessage(role="assistant", content=answer))
                messages.append(ChatMessage(role="user", content=scenario.follow_up))
                follow_up_answer = await _ask(messages, deterministic=args.deterministic)
                _print_turn("Follow-up", scenario.follow_up, follow_up_answer, args.max_chars)
        except Exception as exc:
            failures += 1
            print(f"FAILED: {type(exc).__name__}: {exc}")

    if failures:
        print(f"\nSmoke suite finished with {failures} failure(s).")
        return 1

    print("\nSmoke suite finished successfully.")
    return 0


async def _ask(messages: list[ChatMessage], *, deterministic: bool) -> str:
    if deterministic:
        return await run_book_agent(messages, provider=None)

    chunks: list[str] = []
    async for chunk in stream_book_agent_response(messages):
        chunks.append(chunk)
    return "".join(chunks)


def _print_turn(label: str, prompt: str, answer: str, max_chars: int) -> None:
    print(f"{label}: {prompt}")
    print("Answer:")
    print(answer[:max_chars])
    if len(answer) > max_chars:
        print("...")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
