"""Command line interface."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="Analyse bouldering technique from a video.")


@app.command()
def analyse(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, help="Video of the climb, filmed from behind."),
    hold_colour: str = typer.Option("yellow", "--hold-colour", "--hold-color", help="Colour of the problem's holds."),
    out: Path = typer.Option(Path("out"), help="Folder to create the results folder in."),
    config: Optional[Path] = typer.Option(None, exists=True, help="YAML file overriding rule thresholds."),
    holds: Optional[Path] = typer.Option(None, exists=True, help="holds.yaml to use instead of detecting holds."),
    hsv: Optional[str] = typer.Option(None, help="Custom hold colour range: h_lo,s_lo,v_lo,h_hi,s_hi,v_hi (OpenCV HSV)."),
    layout: str = typer.Option("panels", help="'panels' (side by side) or 'overlay' (fix drawn over you)."),
    stride: int = typer.Option(1, min=1, help="Analyse every Nth frame (faster on long clips)."),
    model: str = typer.Option("full", help="Pose model: lite, full or heavy."),
):
    """Write a folder of annotated screenshots, one per technique error."""
    from .io.video import VideoError
    from .pipeline import AnalysisError, analyse as run

    if layout not in ("panels", "overlay"):
        raise typer.BadParameter("layout must be 'panels' or 'overlay'")
    try:
        run(video, hold_colour, out, config, holds, hsv, layout, stride, model, log=typer.echo)
    except (AnalysisError, VideoError, ValueError) as err:
        typer.secho(f"Error: {err}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


@app.command()
def holds(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    hold_colour: str = typer.Option("yellow", "--hold-colour", "--hold-color"),
    out: Path = typer.Option(Path("holds.yaml"), help="Where to write the editable hold list."),
    hsv: Optional[str] = typer.Option(None, help="Custom hold colour range (OpenCV HSV)."),
    stride: int = typer.Option(3, min=1),
    model: str = typer.Option("lite"),
):
    """Detect holds and write holds.yaml plus a numbered picture, so you can correct pull directions."""
    import cv2

    from .holds.orientation import save_holds
    from .holds.segment import DISPLAY_BGR
    from .io.video import VideoError
    from .pipeline import AnalysisError, find_holds, track_video
    from .pose.estimator import PoseEstimator
    from .report.annotate import render_holds_map

    try:
        estimator = PoseEstimator(model)
        try:
            track = track_video(video, estimator, stride, typer.echo)
        finally:
            estimator.close()
        found, bg = find_holds(track, hold_colour, hsv)
    except (AnalysisError, VideoError, ValueError) as err:
        typer.secho(f"Error: {err}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    save_holds(out, found, hold_colour)
    picture = out.with_suffix(".png")
    full = cv2.resize(bg, None, fx=1 / track["bg_factor"], fy=1 / track["bg_factor"])
    cv2.imwrite(str(picture), render_holds_map(full, found, DISPLAY_BGR.get(hold_colour, (0, 230, 255))))
    typer.echo(f"Found {len(found)} {hold_colour} holds. Wrote {out} and {picture}.")


if __name__ == "__main__":
    app()
