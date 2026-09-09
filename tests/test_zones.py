"""Phase 31 -- security zones (definition, persistence, evaluation)."""

import pytest

from src import database as db
from src.domain import UNKNOWN_ID, is_time_in_window
from src.zones import (
    POLICY_AFTER_HOURS_INTRUSION,
    POLICY_BYPASS,
    POLICY_INTRUSION,
    POLICY_UNKNOWN_ONLY,
    Zone,
    ZoneStore,
    parse_polygon,
    seed_zones,
)

CLOCK_10AM = 10 * 60
CLOCK_6PM = 18 * 60
CLOCK_7PM = 19 * 60


class TestParsePolygon:
    def test_full_frame(self):
        assert parse_polygon("full") is None
        assert parse_polygon(None) is None

    def test_json_polygon(self):
        pts = parse_polygon('[[0.2,0.2],[0.8,0.2],[0.8,0.8],[0.2,0.8]]')
        assert pts == [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]

    def test_list_polygon(self):
        pts = parse_polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert len(pts) == 4

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            parse_polygon('[[0.2,0.2],[1.5,0.2],[0.8,0.8],[0.2,0.8]]')

    def test_too_few_points(self):
        with pytest.raises(ValueError):
            parse_polygon('[[0,0],[1,0]]')


class TestZoneGeometry:
    def test_full_frame_contains_any_normalised(self):
        z = Zone(zone_name="ALL", polygon=None)
        assert z.contains(0.0, 0.0)
        assert z.contains(0.5, 0.5)
        assert z.contains(1.0, 1.0)

    def test_square_polygon(self):
        z = Zone(zone_name="SQ", polygon=[(0.2, 0.2), (0.8, 0.2),
                                          (0.8, 0.8), (0.2, 0.8)])
        assert z.contains(0.5, 0.5)
        assert not z.contains(0.1, 0.1)
        assert not z.contains(0.9, 0.9)

    def test_schedule_window(self):
        z = Zone(zone_name="Z", polygon=None,
                 schedule={"start": "09:00", "end": "18:00"})
        assert z.time_is_active(CLOCK_10AM)
        assert not z.time_is_active(CLOCK_6PM)  # end exclusive
        assert not z.time_is_active(CLOCK_7PM)

    def test_no_schedule_is_24x7(self):
        z = Zone(zone_name="Z", polygon=None)
        assert z.time_is_active(CLOCK_7PM)

    def test_overnight_window_via_domain_helper(self):
        assert is_time_in_window(23 * 60, "22:00", "06:00")
        assert is_time_in_window(2 * 60, "22:00", "06:00")
        assert not is_time_in_window(12 * 60, "22:00", "06:00")


class TestAllowList:
    def test_empty_allows_everyone(self):
        z = Zone(zone_name="Z", allowed=set())
        assert z.allows("E1")
        assert z.allows(UNKNOWN_ID)

    def test_explicit_allows_only_members(self):
        z = Zone(zone_name="Z", allowed={"E1", "E2"})
        assert z.allows("E1")
        assert not z.allows("E3")
        # Unknown is never auto-exempt -- policy decides handling.
        assert not z.allows(UNKNOWN_ID)


class TestZoneStore:
    def test_add_list(self, tmp_db):
        store = ZoneStore(tmp_db)
        store.add(zone_name="SERVER_ROOM", camera="CAM1",
                  polygon='[[0.2,0.2],[0.8,0.2],[0.8,0.8],[0.2,0.8]]',
                  allowed=["E1"], policy=POLICY_INTRUSION)
        zones = store.list()
        assert len(zones) == 1
        assert zones[0].zone_name == "SERVER_ROOM"
        assert zones[0].allows("E1")

    def test_add_roundtrip_persists(self, tmp_db):
        store = ZoneStore(tmp_db)
        store.add(zone_name="Z", camera="CAM1", polygon=None)
        z = store.get("Z")
        assert z.full_frame
        assert z.policy == POLICY_INTRUSION

    def test_invalid_policy_rejected(self, tmp_db):
        store = ZoneStore(tmp_db)
        with pytest.raises(ValueError):
            store.add(zone_name="BAD", camera="CAM1", policy="NOPE")

    def test_remove(self, tmp_db):
        store = ZoneStore(tmp_db)
        store.add(zone_name="Z", camera="CAM1")
        store.remove("Z")
        assert store.get("Z") is None

    def test_update_enabled(self, tmp_db):
        store = ZoneStore(tmp_db)
        store.add(zone_name="Z", camera="CAM1")
        store.update_enabled("Z", False)
        assert store.get("Z").enabled is False
        assert store.list(enabled_only=True) == []

    def test_evaluate_intrusion_policy(self, tmp_db):
        store = ZoneStore(tmp_db)
        store.add(zone_name="SERVER", camera="CAM1",
                  polygon='[[0,0],[1,0],[1,1],[0,1]]', allowed=["E1"])
        hits = store.evaluate_presence("CAM1", 0.5, 0.5, "E2", CLOCK_10AM)
        assert len(hits) == 1
        assert hits[0]["intrusion"] is True
        # Allowed employee in the same zone is present-but-allowed.
        ok = store.evaluate_presence("CAM1", 0.5, 0.5, "E1", CLOCK_10AM)
        assert ok[0]["intrusion"] is False

    def test_evaluate_respects_camera_scope(self, tmp_db):
        store = ZoneStore(tmp_db)
        store.add(zone_name="SERVER", camera="CAM1", polygon=None)
        assert store.evaluate_presence("CAM2", 0.5, 0.5, "E2", CLOCK_10AM) == []
        # Blank camera on the zone => matches any camera.
        store.add(zone_name="GLOBAL", camera="", polygon=None)
        assert len(store.evaluate_presence("CAM9", 0.5, 0.5, "E2", CLOCK_10AM)) == 1

    def test_evaluate_unknown_only_policy(self, tmp_db):
        store = ZoneStore(tmp_db)
        store.add(zone_name="VAULT", camera="CAM1", polygon=None,
                  policy=POLICY_UNKNOWN_ONLY)
        hits = store.evaluate_presence("CAM1", 0.5, 0.5, UNKNOWN_ID, CLOCK_10AM)
        assert hits[0]["intrusion"] is True
        no_hit = store.evaluate_presence("CAM1", 0.5, 0.5, "E1", CLOCK_10AM)
        assert no_hit[0]["intrusion"] is False

    def test_evaluate_after_hours_policy(self, tmp_db):
        store = ZoneStore(tmp_db)
        store.add(zone_name="HQ", camera="CAM1", polygon=None,
                  schedule={"start": "09:00", "end": "18:00"},
                  policy=POLICY_AFTER_HOURS_INTRUSION)
        # Inside working hours: allowed, no intrusion.
        in_hours = store.evaluate_presence("CAM1", 0.5, 0.5, "E1", CLOCK_10AM)
        assert in_hours[0]["intrusion"] is False
        # After hours: intrusion regardless of identity.
        after = store.evaluate_presence("CAM1", 0.5, 0.5, "E1", CLOCK_7PM)
        assert after[0]["intrusion"] is True

    def test_evaluate_bypass_never_intrudes(self, tmp_db):
        store = ZoneStore(tmp_db)
        store.add(zone_name="LOBBY", camera="CAM1", polygon=None,
                  policy=POLICY_BYPASS)
        hits = store.evaluate_presence("CAM1", 0.5, 0.5, "E2", CLOCK_7PM)
        assert hits == []


class TestSeed:
    def test_seed_when_empty(self, tmp_db, tmp_path):
        cfg = tmp_path / "zones.json"
        cfg.write_text(
            '{"zones": [{"zone_name": "A", "camera": "CAM1", '
            '"polygon": "full", "alert_policy": "INTRUSION"}]}',
            encoding="utf-8",
        )
        store = ZoneStore(tmp_db)
        added = seed_zones(tmp_db, store, str(cfg))
        assert added == 1
        assert store.get("A") is not None

    def test_no_seed_when_table_populated(self, tmp_db, tmp_path):
        store = ZoneStore(tmp_db)
        store.add(zone_name="EXISTING", camera="CAM1")
        cfg = tmp_path / "zones.json"
        cfg.write_text('{"zones": [{"zone_name": "B", "camera": "CAM1"}]}',
                       encoding="utf-8")
        assert seed_zones(tmp_db, store, str(cfg)) == 0
        assert store.get("B") is None

    def test_missing_file_is_noop(self, tmp_db, tmp_path):
        store = ZoneStore(tmp_db)
        assert seed_zones(tmp_db, store, str(tmp_path / "missing.json")) == 0