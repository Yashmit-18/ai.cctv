"""Tests for detector geometry helpers (pure, no YOLO/insightface load)."""

import pytest

from src.detector import (
    _boxes_overlap,
    _expand_box,
    _phone_near_face,
)


def test_boxes_overlap_basic():
    assert _boxes_overlap((0, 0, 10, 10), (5, 5, 15, 15)) is True
    assert _boxes_overlap((0, 0, 10, 10), (20, 20, 30, 30)) is False


def test_box_center_midpoint():
    assert _expand_box((10, 10, 20, 20), (100, 100), ratio=1.0) == (10, 10, 20, 20)


def test_expand_box_ratio_and_clamp():
    # face 100x100 at center of 400x400 frame, ratio 2 -> 200x200 centered
    box = _expand_box((100, 100, 200, 200), (400, 400), ratio=2.0)
    assert box == (50, 50, 250, 250)
    # near edge -> clamped to frame
    edge = _expand_box((0, 0, 100, 100), (400, 400), ratio=2.0)
    assert edge[0] == 0 and edge[1] == 0
    assert edge[2] <= 400 and edge[3] <= 400


def test_phone_near_face_overlap():
    face = (50, 50, 150, 150)
    phone = (120, 120, 180, 180)   # overlaps expanded face
    assert _phone_near_face(face, [phone], (400, 400)) is True


def test_phone_near_face_proximity():
    face = (50, 50, 150, 150)
    phone = (300, 50, 360, 110)    # outside expansion, far -> False
    assert _phone_near_face(face, [phone], (400, 400)) is False


def test_phone_near_face_empty_phones():
    assert _phone_near_face((50, 50, 150, 150), [], (400, 400)) is False


def test_phone_on_desk_far_below_face_is_false():
    # face at top of frame; phone "on the desk" well below near the chin-less
    # area of an expanded box -> not phone use
    assert _phone_near_face((50, 50, 150, 150), [(320, 380, 380, 430)], (640, 640)) is False


def test_phone_touching_expanded_edge_is_true():
    # phone just inside the 1.5x expanded face region -> call behaviour
    assert _phone_near_face((200, 200, 300, 300), [(320, 240, 360, 300)], (640, 640)) is True


def test_phone_close_to_expanded_centre_within_proximity_is_true():
    # face 100x100 at centre; expanded threshold = max(w,h)*0.4 = 60px
    # phone centre ~59px from expanded centre -> True (proximity band)
    assert _phone_near_face((200, 200, 300, 300), [(301, 242, 317, 258)], (640, 640)) is True


def test_phone_beyond_proximity_and_no_overlap_is_false():
    # phone centre 90px from expanded centre, fully outside expanded box
    assert _phone_near_face((200, 200, 300, 300), [(332, 242, 348, 258)], (640, 640)) is False


def test_phone_at_corner_face_desk_is_false():
    # face pinned to top-left; expanded box is clamped; desk phone bottom-right
    assert _phone_near_face((0, 0, 100, 100), [(310, 310, 370, 370)], (400, 400)) is False


def test_phone_overlapping_face_corner_is_true():
    # phone pokes into the clamped expanded box at the frame corner
    assert _phone_near_face((0, 0, 100, 100), [(110, 110, 140, 140)], (400, 400)) is True


def test_multiple_phones_any_near_counts():
    face = (50, 50, 150, 150)
    far = (500, 500, 520, 520)
    near = (120, 110, 140, 130)
    assert _phone_near_face(face, [far, near], (640, 640)) is True
    assert _phone_near_face(face, [far, (80, 80, 100, 100)], (640, 640)) is True


def test_large_phone_mostly_overlapping_is_true():
    # oversized phone spanning the face region -> definitely phone use
    assert _phone_near_face((200, 200, 300, 300), [(190, 190, 360, 360)], (640, 640)) is True