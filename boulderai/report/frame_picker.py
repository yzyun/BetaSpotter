"""Choose which findings become screenshots: merge repeats, keep climb order."""
from __future__ import annotations

from dataclasses import replace

from ..types import Finding


def select(findings: list[Finding], cfg: dict) -> list[Finding]:
    """Merge findings of the same rule and limb that are close in time.

    Each finding already carries its worst frame; a merged finding keeps the frame
    of the worse of the two. The result is in climb order.
    """
    rc = cfg["report"]
    merged: list[Finding] = []
    for f in sorted(findings, key=lambda f: (f.rule, str(f.limb), f.t_start)):
        last = merged[-1] if merged else None
        if (last is not None and last.rule == f.rule and last.limb == f.limb
                and f.t_start - last.t_end < rc["merge_gap_s"]):
            keep = f if f.score > last.score else last
            merged[-1] = replace(keep, t_start=last.t_start, t_end=max(last.t_end, f.t_end))
        else:
            merged.append(f)
    if len(merged) > rc["max_findings"]:
        merged = sorted(merged, key=lambda f: -f.score)[:rc["max_findings"]]
    return sorted(merged, key=lambda f: (f.frame, f.rule))
