"""Tests for the deterministic office-scenario generator (Phase 36).

Verifies the emitted detection records conform to the REAL batched-detection
schema, are reproducible under a fixed seed, never fabricate unsupported AI
families, and always keep ``Unknown`` distinct from employees.
"""

import pytest

from src.domain import UNKNOWN_ID
from src.simulation.scenario import SCENARIOS, ScenarioPlan

# Event families that MUST NOT be fabricated without a real model.
_FORBIDDEN_FAMILIES = ("fire", "smoke", "fight", "fall", "weapon", "ppe", "violence")


@pytest.mark.simulation
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_all_scenarios_known_and_load(name):
    plan = ScenarioPlan.for_scenario(name, seed=42)
    assert plan.cameras
    assert plan.steps > 0
    assert plan.fps > 0


@pytest.mark.simulation
def test_detections_reproducible_same_seed():
    a = ScenarioPlan.for_scenario("A_normal_occupancy", seed=11)
    b = ScenarioPlan.for_scenario("A_normal_occupancy", seed=11)
    for step in range(a.steps):
        assert a.detections_at_step(step) == b.detections_at_step(step)


@pytest.mark.simulation
def test_different_seed_still_valid_schema():
    rng1 = ScenarioPlan.for_scenario("F_unknown_presence", seed=1)
    rng2 = ScenarioPlan.for_scenario("F_unknown_presence", seed=2)
    for step in range(rng1.steps):
        for d in rng1.detections_at_step(step):
            assert set(["cam", "emp_id", "phone", "person_present"]).issubset(d), d
    # both run without error on a different seed
    for step in range(rng2.steps):
        rng2.detections_at_step(step)


@pytest.mark.simulation
def test_cross_camera_move_uses_expected_cameras():
    plan = ScenarioPlan.for_scenario("J_cross_camera_move", seed=42)
    seen = set()
    for step in range(plan.steps):
        for d in plan.detections_at_step(step):
            if d["emp_id"] == "EMP003":
                seen.add(d["cam"])
    assert "cam_01" in seen and "cam_02" in seen and "cam_03" in seen


@pytest.mark.simulation
def test_unknown_used_only_where_scripted_and_never_employee():
    plan = ScenarioPlan.for_scenario("F_unknown_presence", seed=42)
    seen_unknown = False
    seen_employee = False
    for step in range(plan.steps):
        for d in plan.detections_at_step(step):
            if d["emp_id"] == UNKNOWN_ID:
                seen_unknown = True
            elif d["emp_id"] not in ("__person__",):
                if not d["emp_id"].startswith("EMP"):
                    raise AssertionError(f"unexpected id {d['emp_id']!r}")
                seen_employee = True
    assert seen_unknown
    assert not seen_employee  # F scenario scripts no employees at all


@pytest.mark.simulation
def test_no_forbidden_ai_families_emitted_any_scenario():
    for name in SCENARIOS:
        plan = ScenarioPlan.for_scenario(name, seed=99)
        for step in range(plan.steps):
            for d in plan.detections_at_step(step):
                blob = f"{d['emp_id']}|{d.get('cam')}".lower()
                for fam in _FORBIDDEN_FAMILIES:
                    assert fam not in blob, f"{name} leaked {fam!r}"