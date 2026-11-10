"""Footwork and balance: centre of mass outside the contacts, feet being re-placed."""
from __future__ import annotations

import numpy as np

from ..analysis import geometry as G
from ..analysis.contacts import episodes
from ..types import Finding
from .common import in_contact, severity


def com_offsets(climb) -> np.ndarray:
    """Per frame: signed sideways distance (torso lengths) of the CoM outside the contacts."""
    placed = in_contact(climb)
    out = np.zeros(len(climb.t))
    for i in range(len(out)):
        if placed[i].sum() < 2:
            continue
        xs = climb.limb_pts[i, placed[i], 0]
        out[i] = G.com_offset(climb.com[i, 0], xs, climb.scale[i])
    return out


def check_com(climb) -> list[Finding]:
    cfg = climb.cfg["footwork"]
    off = com_offsets(climb)
    mask = climb.static & climb.on_wall & (np.abs(off) > cfg["com_margin"])
    findings = []
    for s, e in G.runs(mask, climb.frames(cfg["min_duration_s"])):
        frame = s + int(np.argmax(np.abs(off[s:e])))
        worst = float(off[frame])
        side = "right" if worst > 0 else "left"
        findings.append(Finding(
            rule="com-outside-support", title="OFF BALANCE",
            t_start=float(climb.t[s]), t_end=float(climb.t[e - 1]), frame=frame,
            severity=severity(abs(worst) - cfg["com_margin"], 0.2, 0.45), limb=None,
            measured=f"centre of mass is {abs(worst) * 100:.0f}% of a torso length to the {side} "
                     f"of your hands and feet (aim for under {cfg['com_margin'] * 100:.0f}%)",
            tip=f"Bring your weight back between your holds, or put a foot out to the {side} "
                "to stop the barn-door.",
            evidence={"offset": worst, "side": side},
            score=abs(worst) - cfg["com_margin"],
        ))
    return findings


def check_readjust(climb) -> list[Finding]:
    cfg = climb.cfg["footwork"]
    findings = []
    for limb, side in ((2, "left"), (3, "right")):
        eps = [ep for ep in episodes(climb.contacts, limb) if climb.on_wall[ep[0]]]
        chain = []
        for ep in eps + [None]:
            if ep is not None and chain:
                prev = chain[-1]
                pos_prev = np.median(climb.limb_pts[prev[0]:prev[1], limb], axis=0)
                pos = np.median(climb.limb_pts[ep[0]:ep[1], limb], axis=0)
                near = np.linalg.norm(pos - pos_prev) <= cfg["readjust_radius"] * climb.scale[ep[0]]
                quick = climb.t[ep[0]] - climb.t[prev[1] - 1] <= cfg["readjust_gap_s"]
                if near and quick:
                    chain.append(ep)
                    continue
            if len(chain) - 1 >= cfg["readjust_count"]:
                n = len(chain) - 1
                findings.append(Finding(
                    rule="foot-readjust", title="SLOPPY FOOT PLACEMENT",
                    t_start=float(climb.t[chain[0][0]]), t_end=float(climb.t[chain[-1][1] - 1]),
                    frame=chain[-1][0], severity=severity(n, 3, 5), limb=side,
                    measured=f"{side} foot re-placed {n} times on the same spot "
                             f"(aim to place it once)",
                    tip="Look at the foothold until your toe is on it, place it precisely once, then trust it.",
                    evidence={"side": side, "count": n},
                    score=float(n),
                ))
            chain = [ep] if ep is not None else []
    return findings


def check(climb) -> list[Finding]:
    return check_com(climb) + check_readjust(climb)
