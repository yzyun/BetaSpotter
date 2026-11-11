"""Draw one finding as a comparison image: your climb | suggested fix | alternative beta."""
from __future__ import annotations

import cv2
import numpy as np

from ..analysis import geometry as G
from ..fixes.corrections import describe_change
from ..pose import landmarks as L
from ..types import NO_CONTACT, Finding, Suggestion

RED = (50, 50, 235)
GREEN = (90, 210, 90)
BLUE = (240, 160, 50)
GREY = (185, 185, 185)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_MAX_H = 1100


def _pt(p) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def _th(scale: float, frac: float, minimum: int = 1) -> int:
    return max(minimum, int(round(scale * frac)))


def put_label(img, text: str, org, fs: float, colour=WHITE, bg=BLACK) -> None:
    """Text on a solid box, kept inside the image."""
    thick = max(1, int(round(fs * 2)))
    (tw, th), base = cv2.getTextSize(text, FONT, fs, thick)
    pad = int(6 * fs) + 2
    h, w = img.shape[:2]
    x = int(np.clip(org[0], pad, max(pad, w - tw - pad)))
    y = int(np.clip(org[1], th + pad, max(th + pad, h - base - pad)))
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), bg, -1)
    cv2.putText(img, text, (x, y), FONT, fs, colour, thick, cv2.LINE_AA)


def dashed_line(img, a, b, colour, thick: int, dash: float) -> None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(1, int(np.linalg.norm(b - a) / max(dash, 1.0)))
    for i in range(0, n, 2):
        cv2.line(img, _pt(a + (b - a) * i / n), _pt(a + (b - a) * min(i + 1, n) / n),
                 colour, thick, cv2.LINE_AA)


def draw_holds(img, holds, colour, thick: int, numbers: bool = False, fs: float = 0.5) -> None:
    for h in holds:
        cv2.polylines(img, [h.polygon.astype(np.int32).reshape(-1, 1, 2)], True, colour, thick, cv2.LINE_AA)
        if numbers:
            put_label(img, f"#{h.id}", h.centroid + np.array([8, -8]), fs)


def draw_wall(img, wall, scale: float, fs: float) -> None:
    """Side view: the wall surface as a dashed line, labelled with its angle."""
    if wall is None:
        return
    h, w = img.shape[:2]
    far = 2.0 * (h + w)
    ok, a, b = cv2.clipLine((0, 0, w, h), _pt(wall.p0 - far * wall.up), _pt(wall.p0 + far * wall.up))
    if not ok:
        return
    dashed_line(img, a, b, BLACK, _th(scale, 0.04, 3), 0.15 * scale)
    dashed_line(img, a, b, WHITE, _th(scale, 0.02, 1), 0.15 * scale)
    low = max(a, b, key=lambda p: p[1])  # label the bottom end: the zoom inset sits in a top corner
    put_label(img, f"wall: {wall.describe()}", np.array(low, float) + np.array([8, -12 * fs]), fs * 0.8, BLACK, WHITE)


def draw_skeleton(img, kp, colour, thick: int) -> None:
    for a, b in L.SKELETON:
        cv2.line(img, _pt(kp[a]), _pt(kp[b]), colour, thick, cv2.LINE_AA)
    for j in L.MAIN_JOINTS:
        cv2.circle(img, _pt(kp[j]), thick + 1, colour, -1, cv2.LINE_AA)


def draw_figure(img, kp, colour, scale: float, alpha: float = 0.8) -> None:
    """A filled stick-person: thick limbs, solid torso, head circle. Blended onto img."""
    layer = img.copy()
    limb = _th(scale, 0.13, 3)
    torso = np.array([kp[11], kp[12], kp[24], kp[23]], dtype=np.int32)
    cv2.fillPoly(layer, [torso], colour, cv2.LINE_AA)
    for a, b in L.SKELETON:
        cv2.line(layer, _pt(kp[a]), _pt(kp[b]), colour, limb, cv2.LINE_AA)
    neck = G.shoulder_mid(kp)
    head = neck + 0.42 * scale * G.unit(kp[L.NOSE] - neck) if np.linalg.norm(kp[L.NOSE] - neck) > 1 \
        else neck - np.array([0, 0.42 * scale])
    cv2.circle(layer, _pt(head), _th(scale, 0.2, 3), colour, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, dst=img)
    # dark outline so the figure reads on any wall colour
    edge = _th(scale, 0.02)
    cv2.polylines(img, [torso.reshape(-1, 1, 2)], True, BLACK, edge, cv2.LINE_AA)
    cv2.circle(img, _pt(head), _th(scale, 0.2, 3), BLACK, edge, cv2.LINE_AA)


def _arrow(img, a, b, colour, thick: int) -> None:
    length = float(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float)))
    if length < 2:
        return
    cv2.arrowedLine(img, _pt(a), _pt(b), BLACK, thick + 2, cv2.LINE_AA, tipLength=min(0.35, 14 * thick / length))
    cv2.arrowedLine(img, _pt(a), _pt(b), colour, thick, cv2.LINE_AA, tipLength=min(0.35, 14 * thick / length))


# ---------------------------------------------------------------- per-rule markup

def _focus(climb, f: Finding) -> tuple[np.ndarray, float]:
    """The spot to ring / zoom in on, and its radius."""
    kp, s = climb.kp[f.frame], float(climb.scale[f.frame])
    if f.rule == "bent-arms":
        elbows = np.array([kp[L.ARM[side][1]] for side in f.evidence["sides"]])
        centre = elbows.mean(axis=0)
        return centre, 0.6 * s + float(np.linalg.norm(elbows - centre, axis=1).max())
    if f.rule == "wrong-pull-direction":
        return kp[L.ARM[f.evidence["side"]][2]], 0.75 * s
    if f.rule == "com-outside-support":
        return climb.com[f.frame], 0.5 * s
    if f.rule == "feet-cut":
        ankles = kp[[L.L_ANKLE, L.R_ANKLE]]
        centre = ankles.mean(axis=0)
        return centre, 0.5 * s + float(np.linalg.norm(ankles - centre, axis=1).max())
    if f.rule == "foot-readjust":
        return kp[L.LEG[f.evidence["side"]][3]], 0.45 * s
    if f.rule == "passive-shoulders":
        return G.shoulder_mid(kp), 0.55 * s
    return G.hip_mid(kp), 0.6 * s


def _mark_bent_arms(img, climb, f, fs):
    kp, s = climb.kp[f.frame], float(climb.scale[f.frame])
    for side, ang in f.evidence["sides"].items():
        sh, el, wr = (kp[i] for i in L.ARM[side])
        cv2.line(img, _pt(sh), _pt(el), RED, _th(s, 0.07, 3), cv2.LINE_AA)
        cv2.line(img, _pt(el), _pt(wr), RED, _th(s, 0.07, 3), cv2.LINE_AA)
        a0 = np.degrees(np.arctan2(*(sh - el)[::-1]))
        a1 = np.degrees(np.arctan2(*(wr - el)[::-1]))
        if abs(a1 - a0) > 180:  # draw the inside of the elbow, not the long way round
            a1 += 360 if a1 < a0 else -360
        cv2.ellipse(img, _pt(el), (_th(s, 0.22, 6),) * 2, 0, a0, a1, WHITE, _th(s, 0.03, 2), cv2.LINE_AA)
        out = G.unit(el - 0.5 * (sh + wr))
        put_label(img, f"{ang:.0f} deg", el + out * 0.35 * s + np.array([-20 * fs, 0]), fs, WHITE, RED)


def _mark_pull(img, climb, f, fs):
    kp, s = climb.kp[f.frame], float(climb.scale[f.frame])
    side = f.evidence["side"]
    hold = climb.hold(f.evidence["hold_id"])
    _, el, wr = (kp[i] for i in L.ARM[side])
    cv2.polylines(img, [hold.polygon.astype(np.int32).reshape(-1, 1, 2)], True, GREEN, _th(s, 0.04, 2), cv2.LINE_AA)
    start = G.hand_point(kp, side)
    _arrow(img, start, start + hold.direction * 0.9 * s, GREEN, _th(s, 0.045, 2))
    _arrow(img, start, start + G.pull_vector(kp, side) * 0.9 * s, RED, _th(s, 0.045, 2))
    put_label(img, "hold wants", start + hold.direction * 1.0 * s, fs * 0.8, BLACK, GREEN)
    put_label(img, f"you pull ({f.evidence['angle']:.0f} deg off)",
              start + G.pull_vector(kp, side) * 1.0 * s + np.array([0, 22 * fs]), fs * 0.8, WHITE, RED)


def _mark_com(img, climb, f, fs):
    i = f.frame
    s = float(climb.scale[i])
    placed = climb.contacts[i] != NO_CONTACT
    pts = climb.limb_pts[i, placed]
    if len(pts) >= 2:
        lo, hi = float(pts[:, 0].min()), float(pts[:, 0].max())
        top, bottom = float(pts[:, 1].min()), float(pts[:, 1].max())
        layer = img.copy()
        cv2.rectangle(layer, _pt((lo, top)), _pt((hi, bottom)), GREEN, -1)
        cv2.addWeighted(layer, 0.25, img, 0.75, 0, dst=img)
        cv2.rectangle(img, _pt((lo, top)), _pt((hi, bottom)), GREEN, _th(s, 0.02), cv2.LINE_AA)
        put_label(img, "balanced zone", (lo, top - 10 * fs), fs * 0.8, BLACK, GREEN)
        c = climb.com[i]
        edge = np.array([np.clip(c[0], lo, hi), c[1]])
        _arrow(img, c, edge, RED, _th(s, 0.04, 2))
    c = climb.com[i]
    cv2.circle(img, _pt(c), _th(s, 0.09, 5), BLACK, -1, cv2.LINE_AA)
    cv2.circle(img, _pt(c), _th(s, 0.07, 4), RED, -1, cv2.LINE_AA)
    put_label(img, "centre of mass", c + np.array([0.15 * s, 0.3 * s]), fs * 0.8, WHITE, RED)


def _mark_hips(img, climb, f, fs):
    i = f.frame
    kp, s = climb.kp[i], float(climb.scale[i])
    hands = climb.limb_pts[i, :2].mean(axis=0)
    feet = climb.limb_pts[i, 2:].mean(axis=0)
    dashed_line(img, hands, feet, WHITE, _th(s, 0.025, 2), 0.12 * s)
    hip, sh = G.hip_mid(kp), G.shoulder_mid(kp)
    cv2.line(img, _pt(sh), _pt(hip), RED, _th(s, 0.07, 3), cv2.LINE_AA)
    cv2.circle(img, _pt(hip), _th(s, 0.08, 4), RED, -1, cv2.LINE_AA)
    put_label(img, f"torso looks {f.evidence['ratio'] * 100:.0f}% length: hips out",
              hip + np.array([0.2 * s, 0.1 * s]), fs * 0.8, WHITE, RED)


def _mark_cut(img, climb, f, fs):
    kp, s = climb.kp[f.frame], float(climb.scale[f.frame])
    for limb, target in f.evidence["foot_targets"].items():
        ankle = kp[L.LIMB_END_JOINT[int(limb)]]
        cv2.circle(img, _pt(ankle), _th(s, 0.14, 5), RED, _th(s, 0.035, 2), cv2.LINE_AA)
        cv2.circle(img, _pt(target), _th(s, 0.14, 5), GREEN, _th(s, 0.03, 2), cv2.LINE_AA)
        dashed_line(img, ankle, target, WHITE, _th(s, 0.02, 1), 0.1 * s)
    put_label(img, "feet off", kp[[L.L_ANKLE, L.R_ANKLE]].mean(axis=0) + np.array([0.2 * s, 0]), fs * 0.8, WHITE, RED)


def _mark_swing(img, climb, f, fs):
    i = f.frame
    s = float(climb.scale[i])
    hip = G.hip_mid(climb.kp[i])
    j = max(0, i - climb.frames(0.25))
    _arrow(img, G.hip_mid(climb.kp[j]), hip, RED, _th(s, 0.05, 2))
    cv2.circle(img, _pt(hip), _th(s, 0.08, 4), RED, -1, cv2.LINE_AA)
    put_label(img, "hips swinging", hip + np.array([0.2 * s, 0.1 * s]), fs * 0.8, WHITE, RED)


def _mark_readjust(img, climb, f, fs):
    kp, s = climb.kp[f.frame], float(climb.scale[f.frame])
    toe = kp[L.LEG[f.evidence["side"]][3]]
    put_label(img, f"re-placed x{f.evidence['count']}", toe + np.array([0.3 * s, 0]), fs * 0.8, WHITE, RED)


def _mark_wall_gap(img, climb, f, fs):
    kp, s = climb.kp[f.frame], float(climb.scale[f.frame])
    hip = G.hip_mid(kp)
    cv2.line(img, _pt(climb.wall.foot(hip)), _pt(hip), RED, _th(s, 0.05, 3), cv2.LINE_AA)
    cv2.circle(img, _pt(hip), _th(s, 0.08, 4), RED, -1, cv2.LINE_AA)
    what = "square hips, " if f.rule == "square-reach" else "hips "
    put_label(img, f"{what}{f.evidence['dist']:.2f} torso lengths out (aim under {f.evidence['limit']:.2f})",
              hip + np.array([0.2 * s, 0.25 * s]), fs * 0.8, WHITE, RED)


def _mark_swing_out(img, climb, f, fs):
    i = f.frame
    s = float(climb.scale[i])
    hip = G.hip_mid(climb.kp[i])
    _arrow(img, G.hip_mid(climb.kp[f.evidence["from_frame"]]), hip, RED, _th(s, 0.05, 2))
    cv2.circle(img, _pt(hip), _th(s, 0.08, 4), RED, -1, cv2.LINE_AA)
    put_label(img, f"swung {f.evidence['out']:.2f} torso lengths out",
              hip + np.array([0.2 * s, 0.25 * s]), fs * 0.8, WHITE, RED)


def _mark_shoulders(img, climb, f, fs):
    kp, s = climb.kp[f.frame], float(climb.scale[f.frame])
    sh = G.shoulder_mid(kp)
    ears = 0.5 * (kp[L.L_EAR] + kp[L.R_EAR])
    cv2.line(img, _pt(ears), _pt(sh), RED, _th(s, 0.07, 3), cv2.LINE_AA)
    put_label(img, f"shoulders up by your ears ({f.evidence['ratio'] * 100:.0f}% of usual gap)",
              sh + np.array([0.3 * s, 0.1 * s]), fs * 0.8, WHITE, RED)


_MARKERS = {
    "hips-far-from-wall": _mark_wall_gap, "square-reach": _mark_wall_gap,
    "swing-out": _mark_swing_out, "passive-shoulders": _mark_shoulders,
    "bent-arms": _mark_bent_arms, "wrong-pull-direction": _mark_pull,
    "com-outside-support": _mark_com, "hips-off-wall": _mark_hips,
    "feet-cut": _mark_cut, "body-swing": _mark_swing, "foot-readjust": _mark_readjust,
}


# ---------------------------------------------------------------- panels

def _zoom_inset(img, centre, radius: float) -> None:
    """Paste a 2x close-up of the fault into the top corner furthest from it."""
    h, w = img.shape[:2]
    r = int(radius * 1.3)
    x0, y0 = int(centre[0]) - r, int(centre[1]) - r
    x0, y0 = max(0, min(x0, w - 2 * r)), max(0, min(y0, h - 2 * r))
    crop = img[y0:y0 + 2 * r, x0:x0 + 2 * r]
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return
    size = int(min(4 * r, 0.42 * min(h, w)))
    inset = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(inset, (0, 0), (size - 1, size - 1), WHITE, max(2, size // 120))
    m = int(0.02 * w)
    left = centre[0] > w / 2
    top = centre[1] > h / 2
    ix = m if left else w - size - m
    iy = m if top else h - size - m
    img[iy:iy + size, ix:ix + size] = inset


def your_panel(frame, climb, f: Finding, hold_bgr) -> np.ndarray:
    img = frame.copy()
    i = f.frame
    kp, s = climb.kp[i], float(climb.scale[i])
    fs = max(0.45, img.shape[1] / 900)
    draw_holds(img, climb.holds, hold_bgr, _th(s, 0.02))
    draw_wall(img, climb.wall, s, fs)
    draw_skeleton(img, kp, GREY, _th(s, 0.025))
    centre, radius = _focus(climb, f)
    cv2.circle(img, _pt(centre), int(radius), RED, _th(s, 0.03, 2), cv2.LINE_AA)
    _MARKERS[f.rule](img, climb, f, fs)  # after the ring, so labels stay readable
    ys = kp[L.MAIN_JOINTS, 1]
    if ys.max() - ys.min() < 0.35 * img.shape[0]:
        _zoom_inset(img, centre, radius)
    return img


def suggestion_panel(frame, climb, f: Finding, sug: Suggestion, colour, hold_bgr) -> np.ndarray:
    img = (frame.astype(np.float32) * 0.6).astype(np.uint8)
    i = f.frame
    kp0, s = climb.kp[i], float(climb.scale[i])
    fs = max(0.45, img.shape[1] / 900)
    draw_holds(img, climb.holds, hold_bgr, _th(s, 0.02))
    draw_wall(img, climb.wall, s, fs)
    if sug.kind == "beta" and sug.hold_id is not None:
        hold = climb.hold(sug.hold_id)
        r = int(1.7 * np.sqrt(hold.area / np.pi)) + _th(s, 0.05)
        cv2.circle(img, _pt(hold.centroid), r, colour, _th(s, 0.04, 2), cv2.LINE_AA)
        put_label(img, f"use #{hold.id}", hold.centroid + np.array([r + 4, 0]), fs * 0.8, BLACK, colour)
    draw_figure(img, sug.keypoints, colour, s)
    draw_skeleton(img, kp0, GREY, _th(s, 0.025))
    # arrows from where you were to where the suggestion puts you
    groups = [("hips", [L.L_HIP, L.R_HIP]), ("shoulders", [L.L_SHOULDER, L.R_SHOULDER]),
              ("", [L.L_ELBOW]), ("", [L.R_ELBOW]), ("", [L.L_KNEE]), ("", [L.R_KNEE]),
              ("", [L.L_ANKLE]), ("", [L.R_ANKLE]), ("", [L.L_WRIST]), ("", [L.R_WRIST])]
    for name, joints in groups:
        a, b = kp0[joints].mean(axis=0), sug.keypoints[joints].mean(axis=0)
        if np.linalg.norm(b - a) > 0.15 * s:
            _arrow(img, a, b, WHITE, _th(s, 0.03, 2))
            if name == "hips":
                text = describe_change(kp0, sug.keypoints, s, climb.wall)
                if text:
                    put_label(img, text, b + np.array([0.25 * s, 0.25 * s]), fs * 0.8, BLACK, colour)
    return img


def _fit(img, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= max_h:
        return img
    return cv2.resize(img, (int(round(w * max_h / h)), max_h), interpolation=cv2.INTER_AREA)


def _wrap(text: str, fs: float, thick: int, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and cv2.getTextSize(trial, FONT, fs, thick)[0][0] > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + [cur] if cur else lines


def compose(panels: list[tuple[str, tuple, np.ndarray]], caption: list[tuple[str, tuple, float]]) -> np.ndarray:
    """Panels side by side under title bars, with a caption banner underneath.

    panels: (title, title colour, image). caption: (text, colour, relative size).
    """
    imgs = [_fit(p[2], PANEL_MAX_H) for p in panels]
    h = min(im.shape[0] for im in imgs)
    imgs = [im[:h] for im in imgs]
    gap = 6
    width = sum(im.shape[1] for im in imgs) + gap * (len(imgs) - 1)
    fs = max(0.5, min(imgs[0].shape[1], 900) / 900 * 0.8)
    thick = max(1, int(round(fs * 2)))
    title_h = int(46 * fs) + 8

    # lay out the caption first so we know its height
    rows = []
    for text, colour, rel in caption:
        for line in _wrap(text, fs * rel, max(1, int(round(fs * rel * 2))), width - 40):
            rows.append((line, colour, rel))
    line_h = int(34 * fs) + 6
    cap_h = 20 + sum(int(line_h * rel) for _, _, rel in rows)

    canvas = np.full((title_h + h + cap_h, width, 3), 24, dtype=np.uint8)
    x = 0
    for (title, colour, _), im in zip(panels, imgs):
        cv2.rectangle(canvas, (x, 0), (x + im.shape[1], title_h), colour, -1)
        (tw, th), _ = cv2.getTextSize(title, FONT, fs, thick)
        cv2.putText(canvas, title, (x + (im.shape[1] - tw) // 2, (title_h + th) // 2), FONT, fs,
                    BLACK, thick, cv2.LINE_AA)
        canvas[title_h:title_h + h, x:x + im.shape[1]] = im
        x += im.shape[1] + gap
    y = title_h + h + 12
    for line, colour, rel in rows:
        y += int(line_h * rel)
        cv2.putText(canvas, line, (20, y - 8), FONT, fs * rel, colour,
                    max(1, int(round(fs * rel * 2))), cv2.LINE_AA)
    return canvas


def timestamp(t: float) -> str:
    return f"{int(t // 60):02d}m{t % 60:04.1f}s"


def render_finding(frame, climb, f: Finding, fix: Suggestion | None, beta: Suggestion | None,
                   hold_bgr=(0, 230, 255), layout: str = "panels") -> np.ndarray:
    who = f" ({f.limb})" if f.limb else ""
    caption = [
        (f"{f.title}{who} - {f.severity}", (90, 90, 255), 1.25),
        (f"At {timestamp(climb.t[f.frame])}: {f.measured}.", WHITE, 1.0),
        (f"Why it matters: {f.tip}", (210, 210, 210), 1.0),
    ]
    if fix is not None:
        caption.append((f"Fix (green figure): {fix.tip}", GREEN, 1.0))
    if beta is not None:
        caption.append((f"Alternative beta (blue figure): {beta.tip}", BLUE, 1.0))

    mine = your_panel(frame, climb, f, hold_bgr)
    if layout == "overlay":
        s = float(climb.scale[f.frame])
        if fix is not None:
            draw_figure(mine, fix.keypoints, GREEN, s, alpha=0.5)
        if beta is not None:
            draw_skeleton(mine, beta.keypoints, BLUE, _th(s, 0.04, 2))
        return compose([("YOUR CLIMB + SUGGESTED FIX", (200, 200, 200), mine)], caption)

    panels = [("YOUR CLIMB", (200, 200, 200), mine)]
    if fix is not None:
        panels.append(("SUGGESTED FIX", GREEN, suggestion_panel(frame, climb, f, fix, GREEN, hold_bgr)))
    if beta is not None:
        panels.append(("ALTERNATIVE BETA", BLUE, suggestion_panel(frame, climb, f, beta, BLUE, hold_bgr)))
    return compose(panels, caption)


def render_clean(frame, climb, i: int, hold_bgr=(0, 230, 255)) -> np.ndarray:
    img = frame.copy()
    s = float(climb.scale[i])
    draw_holds(img, climb.holds, hold_bgr, _th(s, 0.02))
    draw_wall(img, climb.wall, s, max(0.45, img.shape[1] / 900))
    draw_skeleton(img, climb.kp[i], GREEN, _th(s, 0.03, 2))
    return compose([("YOUR CLIMB", (200, 200, 200), img)],
                   [("NO TECHNIQUE ERRORS FLAGGED", GREEN, 1.25),
                    ("Nothing crossed the thresholds in this clip.", WHITE, 1.0)])


def render_holds_map(image, holds, hold_bgr=(0, 230, 255)) -> np.ndarray:
    """Numbered holds with their guessed pull direction, for checking holds.yaml."""
    img = image.copy()
    fs = max(0.45, img.shape[1] / 1100)
    size = 0.04 * max(img.shape[:2])
    draw_holds(img, holds, hold_bgr, 2, numbers=True, fs=fs)
    for h in holds:
        _arrow(img, h.centroid, h.centroid + h.direction * size,
               GREEN if h.confidence >= 0.5 else GREY, 2)
    return img
