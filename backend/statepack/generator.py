"""Generate draft State Packs from machine profiles.

Two generators share one interface:
  - LLMGenerator: Claude reads the profile and writes the states (Mode 1 proper).
  - TemplateGenerator: deterministic rules; used offline, in tests, and as a
    baseline to compare the LLM against.

Both return a draft. Nothing is used at runtime until a human approves it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Protocol

from .schema import (
    Action,
    Condition,
    MachineProfile,
    MaintenanceTiming,
    Signal,
    SignalRule,
    StateDefinition,
    StatePack,
    StatePackDraft,
    Urgency,
)
from .validator import ValidationReport, validate

MODEL = "claude-opus-5"


class Generator(Protocol):
    name: str

    def generate(self, profile: MachineProfile, feedback: list[str] | None = None) -> list[StateDefinition]: ...


def build_draft(
    profile: MachineProfile, generator: Generator, version: int, max_attempts: int = 2
) -> tuple[StatePack, ValidationReport]:
    """Generate, validate, and give the generator one chance to fix its errors."""
    feedback: list[str] | None = None
    for _ in range(max_attempts):
        states = generator.generate(profile, feedback)
        report = validate(states, profile)
        if report.ok:
            break
        feedback = report.errors

    pack = StatePack(
        machine_id=profile.machine_id,
        version=version,
        profile_fingerprint=profile.fingerprint(),
        generated_by=generator.name,
        created_at=datetime.now(timezone.utc),
        states=states,
    )
    return pack, report


# ── LLM generator ─────────────────────────────────────────────────────────────


SYSTEM_PROMPT = f"""\
You are an industrial reliability engineer configuring a real-time monitoring
system. Given one machine's profile, define the operating states a fast
classifier (JEV) should recognise from live sensor data.

Your output is reviewed by a human and then frozen. At runtime no LLM is in the
loop, so each state must be recognisable purely from its signature.

Signals available at runtime (use these exact names):
- power_kw: measured electrical power
- power_deviation_pct: % above (+) or below (-) the energy model's expected power
- temperature_c
- vibration: normalised vibration score, same units as the profile
- load_percent: commanded machine load, 0-100
- units_per_hour: production rate

Each signature rule may also require a trend: rising, falling, stable or any.
All rules in a signature must hold for the state to match.

Allowed values:
- condition: {", ".join(c.value for c in Condition)}
- action: {", ".join(a.value for a in Action)}
- urgency: {", ".join(u.value for u in Urgency)}
- maintenance_timing: {", ".join(m.value for m in MaintenanceTiming)}

Requirements:
- Exactly one state with condition NORMAL, named NORMAL, action CONTINUE,
  covering the profile's normal ranges.
- Cover the failure and waste modes realistic for this machine type and the
  notes provided (e.g. bearing wear, idle waste, overheating, overload,
  cooling or lubrication problems). Typically 5-8 states in total.
- Include at least one CRITICAL state for breaching the manufacturer hard
  limits (max_temperature_c, max_vibration), with action STOP or INSPECT.
- No state with action CONTINUE may allow temperature or vibration to reach
  the hard limits.
- Only use SHIFT_WORKLOAD if the profile lists backup machines.
- Keep bounds physically plausible for the machine's ratings.
- Make states as mutually exclusive as practical so the classifier is not
  torn between two.
- description and operator_guidance are read by shop-floor operators: plain,
  specific, one or two sentences each.
"""


class LLMGenerator:
    def __init__(self, model: str = MODEL, client: object | None = None):
        import anthropic

        self.name = model
        self.model = model
        self.client = client or anthropic.Anthropic()

    def generate(self, profile: MachineProfile, feedback: list[str] | None = None) -> list[StateDefinition]:
        content = "Machine profile:\n" + json.dumps(profile.model_dump(mode="json"), indent=2)
        if feedback:
            content += (
                "\n\nA previous attempt failed automated validation with these errors. "
                "Produce a corrected set of states:\n- " + "\n- ".join(feedback)
            )

        response = self.client.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
            output_format=StatePackDraft,
            # Server-side fallback: if the primary model declines, the API
            # retries on a fallback model inside the same call.
            extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
            extra_body={"fallbacks": "default"},
        )

        if response.stop_reason == "refusal":
            raise RuntimeError(f"Model declined to generate a pack for {profile.machine_id}.")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"Output truncated for {profile.machine_id}; raise max_tokens.")
        return response.parsed_output.states


# ── Template generator ────────────────────────────────────────────────────────


class TemplateGenerator:
    """Rule-of-thumb states derived from the profile. No network needed."""

    name = "template"

    def generate(self, profile: MachineProfile, feedback: list[str] | None = None) -> list[StateDefinition]:
        p = profile
        t, v = p.normal_temperature_c, p.normal_vibration
        can_shift = bool(p.backup_machines)
        relieve = Action.SHIFT_WORKLOAD if can_shift else Action.REDUCE_LOAD

        def rule(signal: Signal, lo: float | None = None, hi: float | None = None, trend="any") -> SignalRule:
            return SignalRule(signal=signal, min=lo, max=hi, trend=trend)

        return [
            StateDefinition(
                name="NORMAL",
                condition=Condition.NORMAL,
                description=f"{p.name} running within its normal energy, temperature and vibration ranges.",
                signature=[
                    rule(Signal.POWER_DEVIATION_PCT, -15, 15),
                    rule(Signal.TEMPERATURE_C, t.min, t.max),
                    rule(Signal.VIBRATION, v.min, v.max),
                ],
                likely_causes=[],
                action=Action.CONTINUE,
                urgency=Urgency.LOW,
                maintenance_timing=MaintenanceTiming.NONE,
                operator_guidance="No action needed.",
            ),
            StateDefinition(
                name="ENERGY_INEFFICIENT",
                condition=Condition.INEFFICIENT,
                description="Drawing noticeably more power than expected while temperature and vibration look normal.",
                signature=[
                    rule(Signal.POWER_DEVIATION_PCT, 15, None),
                    rule(Signal.TEMPERATURE_C, None, t.max),
                    rule(Signal.VIBRATION, None, v.max),
                ],
                likely_causes=["process setting change", "worn tooling", "air or fluid leak"],
                action=Action.INSPECT,
                urgency=Urgency.MEDIUM,
                maintenance_timing=MaintenanceTiming.NEXT_IDLE_WINDOW,
                operator_guidance="Check recent setting changes and tooling condition at the next idle window.",
            ),
            StateDefinition(
                name="IDLE_WASTE",
                condition=Condition.INEFFICIENT,
                description="Powered and drawing energy but doing no useful work.",
                signature=[
                    # Utility machines (compressors, pumps) produce no units; use load instead.
                    rule(Signal.UNITS_PER_HOUR, None, 0)
                    if p.rated_units_per_hour > 0
                    else rule(Signal.LOAD_PERCENT, None, 10),
                    rule(Signal.POWER_KW, p.idle_power_kw * 0.8, None),
                ],
                likely_causes=["waiting for material", "left running between batches"],
                action=Action.REDUCE_LOAD,
                urgency=Urgency.MEDIUM,
                maintenance_timing=MaintenanceTiming.NONE,
                operator_guidance="Switch to standby or power down until the next batch is ready.",
            ),
            StateDefinition(
                name="BEARING_WEAR",
                condition=Condition.DEGRADING,
                description="Vibration and power both rising above normal: typical early bearing or alignment wear.",
                signature=[
                    rule(Signal.VIBRATION, v.max, p.max_vibration, trend="rising"),
                    rule(Signal.POWER_DEVIATION_PCT, 10, None),
                ],
                likely_causes=["bearing wear", "misalignment", "lubrication breakdown"],
                action=relieve,
                urgency=Urgency.HIGH,
                maintenance_timing=MaintenanceTiming.NEXT_IDLE_WINDOW,
                operator_guidance=(
                    f"Move work to {', '.join(p.backup_machines)} and inspect bearings."
                    if can_shift
                    else "Reduce load and inspect bearings at the next idle window."
                ),
            ),
            StateDefinition(
                name="OVERHEATING",
                condition=Condition.DEGRADING,
                description="Temperature above normal range and climbing, still below the hard limit.",
                signature=[rule(Signal.TEMPERATURE_C, t.max, p.max_temperature_c, trend="rising")],
                likely_causes=["cooling fault", "blocked airflow", "overload"],
                action=Action.REDUCE_LOAD,
                urgency=Urgency.HIGH,
                maintenance_timing=MaintenanceTiming.NEXT_IDLE_WINDOW,
                operator_guidance="Reduce load and check cooling and airflow.",
            ),
            StateDefinition(
                name="OVER_TEMPERATURE_LIMIT",
                condition=Condition.CRITICAL,
                description=f"Temperature at or beyond the {p.max_temperature_c}°C manufacturer limit.",
                signature=[rule(Signal.TEMPERATURE_C, p.max_temperature_c, None)],
                likely_causes=["cooling failure", "severe overload"],
                action=Action.STOP,
                urgency=Urgency.CRITICAL,
                maintenance_timing=MaintenanceTiming.NOW,
                operator_guidance="Stop the machine safely and call maintenance.",
            ),
            StateDefinition(
                name="OVER_VIBRATION_LIMIT",
                condition=Condition.CRITICAL,
                description=f"Vibration at or beyond the {p.max_vibration} manufacturer limit.",
                signature=[rule(Signal.VIBRATION, p.max_vibration, None)],
                likely_causes=["bearing failure", "loose mounting", "rotor imbalance"],
                action=Action.STOP,
                urgency=Urgency.CRITICAL,
                maintenance_timing=MaintenanceTiming.NOW,
                operator_guidance="Stop the machine safely and call maintenance.",
            ),
        ]
