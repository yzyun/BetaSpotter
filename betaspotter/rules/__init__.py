"""Technique rules. Each module exposes check(climb) -> list[Finding]."""
from __future__ import annotations

from ..types import Finding
from . import bent_arms, body_tension, footwork, pull_direction

RULES = [bent_arms, footwork, body_tension, pull_direction]


def run_rules(climb) -> list[Finding]:
    findings: list[Finding] = []
    for rule in RULES:
        findings.extend(rule.check(climb))
    findings.sort(key=lambda f: f.t_start)
    return findings
