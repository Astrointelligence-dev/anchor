"""Minimal eval harness for skills: with/without-skill A/B baseline.

Eval-driven skill development (agentskills.io/skill-creation/evaluating-skills):
run the same prompts through an agent with and without the skill, judge each
output, and compare pass rates. The judge is a user callback — anchor never
calls an LLM itself.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from anchor.agent.agent import Agent
    from anchor.agent.skills.models import Skill


class SkillEvalCase(BaseModel):
    """One eval scenario: a prompt plus what a passing answer looks like."""

    model_config = ConfigDict(frozen=True)

    prompt: str
    expected: str = ""
    name: str = ""


class SkillEvalResult(BaseModel):
    """Outcome of one case, run with and without the skill."""

    model_config = ConfigDict(frozen=True)

    case: SkillEvalCase
    output_with_skill: str
    output_without_skill: str
    passed_with_skill: bool
    passed_without_skill: bool


class SkillEvalReport(BaseModel):
    """Aggregate pass rates across all cases."""

    results: tuple[SkillEvalResult, ...] = ()
    pass_rate_with_skill: float = Field(default=0.0, ge=0.0, le=1.0)
    pass_rate_without_skill: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def lift(self) -> float:
        """Pass-rate improvement the skill delivers over baseline."""
        return self.pass_rate_with_skill - self.pass_rate_without_skill


def run_skill_eval(
    make_agent: Callable[[], Agent],
    skill: Skill,
    cases: Iterable[SkillEvalCase],
    judge_fn: Callable[[SkillEvalCase, str], bool],
) -> SkillEvalReport:
    """Run each case through a fresh agent with and without *skill*.

    Parameters
    ----------
    make_agent:
        Factory returning a fresh, fully configured :class:`Agent` (LLM,
        system prompt, memory). Called twice per case so runs don't share
        conversation state.
    skill:
        The skill under evaluation; attached only to the "with" run.
    cases:
        Eval scenarios. The spec guidance is at least 3.
    judge_fn:
        Callback ``(case, output) -> passed``. Typically an LLM judge or a
        keyword assertion — the caller decides.

    Returns
    -------
    SkillEvalReport with per-case results and aggregate pass rates.
    """
    results: list[SkillEvalResult] = []
    for case in cases:
        with_agent = make_agent().with_skill(skill)
        output_with = "".join(with_agent.chat(case.prompt))

        without_agent = make_agent()
        output_without = "".join(without_agent.chat(case.prompt))

        results.append(
            SkillEvalResult(
                case=case,
                output_with_skill=output_with,
                output_without_skill=output_without,
                passed_with_skill=judge_fn(case, output_with),
                passed_without_skill=judge_fn(case, output_without),
            )
        )

    n = len(results)
    return SkillEvalReport(
        results=tuple(results),
        pass_rate_with_skill=(
            sum(r.passed_with_skill for r in results) / n if n else 0.0
        ),
        pass_rate_without_skill=(
            sum(r.passed_without_skill for r in results) / n if n else 0.0
        ),
    )
