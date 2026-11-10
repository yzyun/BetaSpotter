"""Write the output folder: one screenshot per finding plus findings.json."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2

from ..types import Finding, Suggestion
from .annotate import render_clean, render_finding, timestamp


def make_out_dir(out_root: str | Path, video: str | Path) -> Path:
    path = Path(out_root) / f"{Path(video).stem}_{datetime.now():%Y%m%d-%H%M%S}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def filename(index: int, t: float, f: Finding) -> str:
    limb = f"_{f.limb}" if f.limb else ""
    return f"{index:02d}_{timestamp(t)}_{f.rule}{limb}.png"


def write_report(out_dir: Path, climb, frames: dict, items: list[tuple[Finding, Suggestion | None, Suggestion | None]],
                 hold_bgr, layout: str = "panels") -> list[Path]:
    """frames: climb frame index -> BGR image, for every finding's frame."""
    written, index = [], []
    for n, (f, fix, beta) in enumerate(items, start=1):
        t = float(climb.t[f.frame])
        name = filename(n, t, f)
        image = render_finding(frames[f.frame], climb, f, fix, beta, hold_bgr, layout)
        cv2.imwrite(str(out_dir / name), image)
        written.append(out_dir / name)
        index.append({
            "file": name, "rule": f.rule, "title": f.title, "limb": f.limb,
            "time_s": round(t, 2), "start_s": round(f.t_start, 2), "end_s": round(f.t_end, 2),
            "severity": f.severity, "measured": f.measured, "tip": f.tip,
            "fix": fix.tip if fix else None, "alternative_beta": beta.tip if beta else None,
        })
    (out_dir / "findings.json").write_text(json.dumps(index, indent=2))
    return written


def write_clean(out_dir: Path, climb, frame, i: int, hold_bgr) -> Path:
    path = out_dir / "00_no-errors.png"
    cv2.imwrite(str(path), render_clean(frame, climb, i, hold_bgr))
    (out_dir / "findings.json").write_text("[]")
    return path
