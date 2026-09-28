import json
import os
from types import SimpleNamespace

import pytest

from backend.statepack.generator import LLMGenerator, TemplateGenerator, build_draft
from backend.statepack.profiles import load_profiles
from backend.statepack.schema import Action, Condition, Signal, SignalRule, StatePackDraft
from backend.statepack.store import StatePackError, StatePackStore
from backend.statepack.validator import validate

PROFILES = load_profiles("config/machines.yaml")


@pytest.fixture
def profile():
    return PROFILES["M01"]


@pytest.fixture
def states(profile):
    return TemplateGenerator().generate(profile)


# ── template generator ───────────────────────────────────────────────────────


@pytest.mark.parametrize("machine_id", sorted(PROFILES))
def test_template_pack_is_valid_for_every_machine(machine_id):
    report = validate(TemplateGenerator().generate(PROFILES[machine_id]), PROFILES[machine_id])
    assert report.ok, report.errors
    assert not report.warnings, report.warnings


def test_template_avoids_shift_workload_without_backup():
    states = TemplateGenerator().generate(PROFILES["M05"])
    assert all(s.action != Action.SHIFT_WORKLOAD for s in states)


# ── validator catches LLM mistakes ───────────────────────────────────────────


def test_rejects_missing_normal(states, profile):
    report = validate([s for s in states if s.condition != Condition.NORMAL], profile)
    assert any("NORMAL" in e for e in report.errors)


def test_rejects_continue_past_hard_limit(states, profile):
    normal = states[0]
    normal.signature = [r for r in normal.signature if r.signal != Signal.TEMPERATURE_C]
    report = validate(states, profile)
    assert any("CONTINUE while temperature" in e for e in report.errors)


def test_rejects_shift_workload_without_backup(states):
    no_backup = PROFILES["M01"].model_copy(update={"backup_machines": []})
    states[3].action = Action.SHIFT_WORKLOAD
    assert any("SHIFT_WORKLOAD" in e for e in validate(states, no_backup).errors)


def test_rejects_normal_that_misses_nominal_operation(states, profile):
    states[0].signature.append(SignalRule(signal=Signal.LOAD_PERCENT, min=95, max=100))
    assert any("Probe 'nominal'" in e for e in validate(states, profile).errors)


def test_rejects_implausible_range(states, profile):
    states[1].signature.append(SignalRule(signal=Signal.POWER_KW, min=0, max=500))
    assert any("plausible" in e for e in validate(states, profile).errors)


# ── generation loop ──────────────────────────────────────────────────────────


class FlakyGenerator:
    """First attempt omits NORMAL, second attempt is correct."""

    name = "flaky"

    def __init__(self):
        self.feedback_seen = []

    def generate(self, profile, feedback=None):
        self.feedback_seen.append(feedback)
        states = TemplateGenerator().generate(profile)
        return states if feedback else states[1:]


def test_build_draft_feeds_errors_back_for_repair(profile):
    gen = FlakyGenerator()
    pack, report = build_draft(profile, gen, version=1)
    assert report.ok
    assert gen.feedback_seen[0] is None
    assert any("NORMAL" in e for e in gen.feedback_seen[1])


def test_llm_generator_request_shape(profile):
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        draft = StatePackDraft(states=TemplateGenerator().generate(profile))
        return SimpleNamespace(stop_reason="end_turn", parsed_output=draft)

    client = SimpleNamespace(messages=SimpleNamespace(parse=parse))
    states = LLMGenerator(client=client).generate(profile, feedback=["missing NORMAL"])

    assert len(states) == 7
    req = calls[0]
    assert req["model"] == "claude-opus-5"
    assert req["output_format"] is StatePackDraft
    content = req["messages"][0]["content"]
    assert json.loads(content.split("Machine profile:\n")[1].split("\n\nA previous")[0])["machine_id"] == "M01"
    assert "missing NORMAL" in content


def test_llm_generator_raises_on_refusal(profile):
    client = SimpleNamespace(
        messages=SimpleNamespace(parse=lambda **_: SimpleNamespace(stop_reason="refusal", parsed_output=None))
    )
    with pytest.raises(RuntimeError, match="declined"):
        LLMGenerator(client=client).generate(profile)


# ── store: approval, freezing, integrity ─────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    return StatePackStore(tmp_path)


def _draft(store, profile):
    pack, _ = build_draft(profile, TemplateGenerator(), store.next_version(profile.machine_id))
    store.save_draft(pack)
    return pack


def test_approve_freezes_pack(store, profile):
    _draft(store, profile)
    path = store.approve("M01", profile, "Shift Lead")

    assert not os.access(path, os.W_OK)
    active = store.load_active("M01", profile)
    assert active.status == "approved" and active.approved_by == "Shift Lead"
    assert store.list() == [("M01", 1, "approved")]


def test_versions_increment_and_latest_approved_wins(store, profile):
    _draft(store, profile)
    store.approve("M01", profile, "A")
    _draft(store, profile)
    store.approve("M01", profile, "B")
    assert store.load_active("M01").version == 2


def test_cannot_approve_invalid_draft(store, profile):
    pack = _draft(store, profile)
    pack.states = pack.states[1:]
    store.save_draft(pack)
    with pytest.raises(StatePackError, match="validation"):
        store.approve("M01", profile, "A")


def test_cannot_approve_after_profile_change(store, profile):
    _draft(store, profile)
    changed = profile.model_copy(update={"max_temperature_c": 80})
    with pytest.raises(StatePackError, match="profile changed"):
        store.approve("M01", changed, "A")


def test_detects_tampering(store, profile):
    _draft(store, profile)
    path = store.approve("M01", profile, "A")
    os.chmod(path, 0o644)
    path.write_text(path.read_text().replace('"STOP"', '"CONTINUE"'))
    with pytest.raises(StatePackError, match="hash mismatch"):
        store.load_active("M01")


def test_detects_stale_pack(store, profile):
    _draft(store, profile)
    store.approve("M01", profile, "A")
    with pytest.raises(StatePackError, match="stale"):
        store.load_active("M01", profile.model_copy(update={"backup_machines": []}))
