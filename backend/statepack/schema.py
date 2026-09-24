"""Data models for Mode 1: machine profiles in, State Packs out.

A State Pack is the frozen, human-approved knowledge JEV uses at runtime.
The LLM writes the `states`; everything else is filled in by code.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ── Enums shared with the runtime decision layer (PLAN.md §16) ────────────────


class Condition(str, Enum):
    NORMAL = "NORMAL"
    INEFFICIENT = "INEFFICIENT"
    DEGRADING = "DEGRADING"
    CRITICAL = "CRITICAL"


class Action(str, Enum):
    CONTINUE = "CONTINUE"
    REDUCE_LOAD = "REDUCE_LOAD"
    SHIFT_WORKLOAD = "SHIFT_WORKLOAD"
    INSPECT = "INSPECT"
    MAINTENANCE = "MAINTENANCE"
    STOP = "STOP"


class Urgency(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class MaintenanceTiming(str, Enum):
    NONE = "NONE"
    NOW = "NOW"
    NEXT_IDLE_WINDOW = "NEXT_IDLE_WINDOW"
    NEXT_SHIFT = "NEXT_SHIFT"
    PLANNED_MAINTENANCE = "PLANNED_MAINTENANCE"


class Signal(str, Enum):
    """Signals the State Engine produces for every machine."""

    POWER_KW = "power_kw"
    POWER_DEVIATION_PCT = "power_deviation_pct"  # vs. energy-baseline model
    TEMPERATURE_C = "temperature_c"
    VIBRATION = "vibration"
    LOAD_PERCENT = "load_percent"
    UNITS_PER_HOUR = "units_per_hour"


Trend = Literal["rising", "falling", "stable", "any"]


# ── Input: what we know about a machine ───────────────────────────────────────


class Range(BaseModel):
    min: float
    max: float


class MachineProfile(BaseModel):
    machine_id: str
    name: str
    type: str
    rated_power_kw: float
    idle_power_kw: float
    rated_units_per_hour: float
    normal_temperature_c: Range
    max_temperature_c: float  # manufacturer hard limit
    normal_vibration: Range
    max_vibration: float  # manufacturer hard limit
    backup_machines: list[str] = []
    age_years: float | None = None
    notes: str = ""  # manual excerpts, known failure modes, operator knowledge

    def fingerprint(self) -> str:
        return _sha256(self.model_dump(mode="json"))


# ── Output: what the LLM writes ───────────────────────────────────────────────


class SignalRule(BaseModel):
    signal: Signal
    min: float | None = Field(None, description="Inclusive lower bound, or null for unbounded")
    max: float | None = Field(None, description="Inclusive upper bound, or null for unbounded")
    trend: Trend = "any"


class StateDefinition(BaseModel):
    name: str = Field(description="UPPER_SNAKE_CASE, e.g. BEARING_WEAR")
    condition: Condition
    description: str = Field(description="One or two sentences: what this state means physically")
    signature: list[SignalRule] = Field(description="All rules must hold for the state to match")
    likely_causes: list[str]
    action: Action
    urgency: Urgency
    maintenance_timing: MaintenanceTiming
    operator_guidance: str = Field(description="What a human operator should check or do")


class StatePackDraft(BaseModel):
    """The part of a pack the LLM is asked to produce."""

    states: list[StateDefinition]


# ── Stored artefact ───────────────────────────────────────────────────────────


class StatePack(BaseModel):
    machine_id: str
    version: int
    status: Literal["draft", "approved"] = "draft"
    profile_fingerprint: str
    generated_by: str  # model id, or "template"
    created_at: datetime
    approved_by: str | None = None
    approved_at: datetime | None = None
    content_hash: str | None = None  # set on approval; covers states + profile fingerprint
    states: list[StateDefinition]

    def compute_content_hash(self) -> str:
        return _sha256(
            {
                "machine_id": self.machine_id,
                "version": self.version,
                "profile_fingerprint": self.profile_fingerprint,
                "states": [s.model_dump(mode="json") for s in self.states],
            }
        )

    def state(self, name: str) -> StateDefinition | None:
        return next((s for s in self.states if s.name == name), None)


def _sha256(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
