import numpy as np

import synth
from betaspotter.holds.orientation import estimate_orientations, load_holds, save_holds
from betaspotter.holds.segment import HSV_PRESETS, build_background, segment_holds


def wall_holds():
    return [
        synth.make_hold(0, (320, 150)),                 # round
        synth.make_hold(1, (400, 400), rx=45, ry=12),   # flat edge
        synth.make_hold(2, (120, 600), rx=12, ry=45),   # upright, left of the route
        synth.make_hold(3, (330, 800)),
    ]


def test_segment_finds_only_the_problem_colour():
    img = synth.draw_wall(holds=wall_holds())
    img[50:90, 500:560] = (255, 80, 0)  # a blue hold from another problem
    found = segment_holds(img, HSV_PRESETS["yellow"])
    assert len(found) == 4
    for hold, truth in zip(found, wall_holds()):  # numbered top to bottom
        assert np.linalg.norm(hold.centroid - truth.centroid) < 4
    assert len(segment_holds(img, HSV_PRESETS["blue"])) == 1


def test_orientation_heuristic():
    found = segment_holds(synth.draw_wall(holds=wall_holds()), HSV_PRESETS["yellow"])
    estimate_orientations(found)
    assert [h.kind for h in found] == ["round", "edge", "sidepull", "round"]
    assert found[1].direction[1] > 0.95 and found[1].confidence >= 0.5   # pull down
    assert found[2].direction[0] > 0.8                                   # pull towards the route
    assert found[0].confidence < 0.5


def test_background_removes_the_climber():
    wall = synth.draw_wall(holds=wall_holds())
    frames, boxes = [], []
    for x in (100, 250, 400):
        frame = wall.copy()
        frame[300:500, x:x + 80] = (0, 220, 255)  # a "climber" wearing hold-coloured clothes
        frames.append(frame)
        boxes.append((x - 10, 290, x + 90, 510))
    bg = build_background(frames, boxes)
    assert np.abs(bg.astype(int) - wall.astype(int)).max() <= 2


def test_holds_yaml_round_trip(tmp_path):
    found = segment_holds(synth.draw_wall(holds=wall_holds()), HSV_PRESETS["yellow"])
    estimate_orientations(found)
    save_holds(tmp_path / "holds.yaml", found, "yellow")
    loaded, colour = load_holds(tmp_path / "holds.yaml")
    assert colour == "yellow" and len(loaded) == 4
    for a, b in zip(found, loaded):
        assert np.allclose(a.direction, b.direction, atol=0.01)
        assert (a.confidence >= 0.5) == (b.confidence >= 0.5)
