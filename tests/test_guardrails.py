"""Tests for guardrails."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.guardrails import (
    run_input_guards,
    percentile_95,
    validate_topic_on_handbook,
)


def test_topic_blocks_random():
    ok, _ = validate_topic_on_handbook("Who won the superbowl in quantum mars colony")
    assert ok is False


def test_topic_allows_handbook_query():
    ok, _ = validate_topic_on_handbook("Nghỉ phép năm được bao nhiêu ngày?")
    assert ok is True


def test_pii_regex_detects_email():
    g = run_input_guards("Liên hệ attacker@test.com về lương.")
    assert g.ok is False


def test_percentile_95():
    assert percentile_95([100.0, 200.0, 300.0]) == 290.0
