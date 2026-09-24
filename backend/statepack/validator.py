"""Deterministic checks a State Pack must pass before a human can approve it.

This is not JEV. It is a gate that catches LLM mistakes: missing NORMAL state,
impossible ranges, CONTINUE while past a hard limit, SHIFT_WORKLOAD with no
backup machine, and so on. It also probes the pack with a few synthetic
readings (nominal operation, hard-limit breaches) to make sure they land in
sensible states.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schema import (
    Action,
    Condition,
    MachineProfile,
    Signal,
    StateDefinition,
    Urgency,
)

NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# ── Matching (used only for probes here; JEV does the real runtime matching) ──


Reading = dict[str, float]  # signal value -> reading
Trends = dict[str, str]  # signal value -> "rising" | "falling" | "stable"


def matches(state: StateDefinition, reading: Reading, trends: Trends | None = None) -> bool:
    trends = trends or {}
    for rule in state.signature:
        value = reading.get(rule.signal.value)
        if value is None:
            return False
        if rule.min is not None and value < rule.min:
            return False
        if rule.max is not None and value > rule.max:
            return False
        if rule.trend != "any" and trends.get(rule.signal.value, "stable") != rule.trend:
            return False
    return True


def probe_readings(p: MachineProfile) -> dict[str, tuple[Reading, Trends]]:
    """Synthetic readings every pack must handle sensibly."""
    nominal = {
        Signal.POWER_KW.value: p.rated_power_kw * 0.8,
        Signal.POWER_DEVIATION_PCT.value: 0.0,
        Signal.TEMPERATURE_C.value: (p.normal_temperature_c.min + p.normal_temperature_c.max) / 2,
        Signal.VIBRATION.value: (p.normal_vibration.min + p.normal_vibration.max) / 2,
        Signal.LOAD_PERCENT.value: 75.0,
        Signal.UNITS_PER_HOUR.value: p.rated_units_per_hour * 0.95,
    }
    stable = {s: "stable" for s in nominal}
    return {
        "nominal": (nominal, stable),
        "over_temperature": (
            {**nominal, Signal.TEMPERATURE_C.value: p.max_temperature_c + 5},
            {**stable, Signal.TEMPERATURE_C.value: "rising"},
        ),
        "over_vibration": (
            {**nominal, Signal.VIBRATION.value: p.max_vibration * 1.1},
            {**stable, Signal.VIBRATION.value: "rising"},
        ),
    }


# ── Validation ────────────────────────────────────────────────────────────────


def validate(states: list[StateDefinition], profile: MachineProfile) -> ValidationReport:
    r = ValidationReport()
    _check_structure(states, r)
    for s in states:
        _check_state(s, profile, r)
    _check_probes(states, profile, r)
    return r


def _check_structure(states: list[StateDefinition], r: ValidationReport) -> None:
    if not states:
        r.errors.append("Pack has no states.")
        return

    names = [s.name for s in states]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        r.errors.append(f"Duplicate state names: {sorted(dupes)}")

    normal = [s for s in states if s.condition == Condition.NORMAL]
    if len(normal) != 1:
        r.errors.append(f"Expected exactly one NORMAL-condition state, found {len(normal)}.")
    elif normal[0].action != Action.CONTINUE:
        r.errors.append("The NORMAL state must use action CONTINUE.")

    if not any(s.condition == Condition.CRITICAL for s in states):
        r.warnings.append("No CRITICAL state defined; hard-limit breaches rely solely on safety rules.")


def _check_state(s: StateDefinition, p: MachineProfile, r: ValidationReport) -> None:
    where = f"State {s.name}"

    if not NAME_RE.match(s.name):
        r.errors.append(f"{where}: name must be UPPER_SNAKE_CASE.")
    if not s.signature:
        r.errors.append(f"{where}: signature is empty, so it would match everything.")

    for rule in s.signature:
        lo, hi = rule.min, rule.max
        if lo is None and hi is None and rule.trend == "any":
            r.errors.append(f"{where}: rule on {rule.signal.value} constrains nothing.")
        if lo is not None and hi is not None and lo > hi:
            r.errors.append(f"{where}: {rule.signal.value} min {lo} > max {hi}.")
        _check_plausible(where, rule.signal, lo, hi, p, r)

    if s.condition == Condition.CRITICAL and s.urgency not in (Urgency.HIGH, Urgency.CRITICAL):
        r.errors.append(f"{where}: CRITICAL condition needs HIGH or CRITICAL urgency.")
    if s.condition == Condition.NORMAL and s.urgency != Urgency.LOW:
        r.warnings.append(f"{where}: NORMAL state usually has LOW urgency.")
    if s.action == Action.SHIFT_WORKLOAD and not p.backup_machines:
        r.errors.append(f"{where}: SHIFT_WORKLOAD but machine has no backup machines.")

    # A state that says "keep running" must never cover a hard-limit breach.
    if s.action == Action.CONTINUE:
        if _allows_at_least(s, Signal.TEMPERATURE_C, p.max_temperature_c):
            r.errors.append(f"{where}: CONTINUE while temperature can reach the {p.max_temperature_c}°C hard limit.")
        if _allows_at_least(s, Signal.VIBRATION, p.max_vibration):
            r.errors.append(f"{where}: CONTINUE while vibration can reach the {p.max_vibration} hard limit.")


def _check_plausible(
    where: str, signal: Signal, lo: float | None, hi: float | None, p: MachineProfile, r: ValidationReport
) -> None:
    bounds = {
        Signal.POWER_KW: (0.0, p.rated_power_kw * 2),
        Signal.POWER_DEVIATION_PCT: (-100.0, 300.0),
        Signal.TEMPERATURE_C: (-20.0, p.max_temperature_c + 60),
        Signal.VIBRATION: (0.0, p.max_vibration * 3),
        Signal.LOAD_PERCENT: (0.0, 150.0),
        Signal.UNITS_PER_HOUR: (0.0, p.rated_units_per_hour * 2),
    }[signal]
    for v in (lo, hi):
        if v is not None and not bounds[0] <= v <= bounds[1]:
            r.errors.append(f"{where}: {signal.value}={v} outside plausible range {bounds}.")


def _allows_at_least(s: StateDefinition, signal: Signal, limit: float) -> bool:
    rules = [rule for rule in s.signature if rule.signal == signal]
    if not rules:
        return True  # unconstrained
    return all(rule.max is None or rule.max >= limit for rule in rules)


def _check_probes(states: list[StateDefinition], p: MachineProfile, r: ValidationReport) -> None:
    by_probe = {
        name: [s for s in states if matches(s, reading, trends)]
        for name, (reading, trends) in probe_readings(p).items()
    }

    nominal = by_probe["nominal"]
    if not any(s.condition == Condition.NORMAL for s in nominal):
        r.errors.append("Probe 'nominal': normal operating point does not match the NORMAL state.")
    if any(s.condition != Condition.NORMAL for s in nominal):
        others = [s.name for s in nominal if s.condition != Condition.NORMAL]
        r.warnings.append(f"Probe 'nominal': also matches non-normal states {others}.")

    for probe in ("over_temperature", "over_vibration"):
        hit = by_probe[probe]
        if any(s.action == Action.CONTINUE for s in hit):
            r.errors.append(f"Probe '{probe}': hard-limit breach matches a CONTINUE state.")
        if not hit:
            r.warnings.append(f"Probe '{probe}': matches no state; JEV would return UNKNOWN.")
