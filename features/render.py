"""Render a STEP solid with its faces coloured by the features of its features JSON.

    .venv\\Scripts\\python.exe -m features.render runs\\features\\part_features.json
    .venv\\Scripts\\python.exe -m features.render runs\\features\\part_features.json --out part.png --views above below side

The STEP path is read from the JSON (``part``); ``--step`` overrides it. One panel per view
(offscreen VTK, the renderer of ``identify.py``), a legend with one entry per feature, and
unrecognised faces in yellow. Default output: ``<json without _features.json>_render.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from identify import CLASS_COLORS, _vtk_render  # noqa: E402
from step_to_graph import load_step  # noqa: E402

UNRECOGNISED = "unrecognised"
# (title, elevation, azimuth) in degrees; negative elevation looks from below
VIEWS = {"above": ("view from above", 28.0, -55.0), "below": ("view from below", -30.0, -55.0),
         "above_back": ("view from above, back", 28.0, 125.0), "below_back": ("view from below, back", -30.0, 125.0),
         "side": ("side view", 0.0, -90.0), "top": ("top view", 89.0, -90.0)}
TYPE_CLASS = {"extrude": "extrude_side", "cut": "cut_side", "fillet": "fillet", "chamfer": "chamfer"}


def face_class(feature: dict | None) -> str:
    """Colour class (``identify.CLASS_COLORS`` key) of a feature."""
    if feature is None:
        return UNRECOGNISED
    if feature["type"] == "hole":
        cls = f"hole_{feature['subtype']}"
        return cls if cls in CLASS_COLORS else "hole_counterbore"  # stepped bore
    return TYPE_CLASS[feature["type"]]


def describe(f: dict) -> str:
    if f["type"] == "hole":
        d = f"{f['id']}: {f['subtype']} hole Ø{f['diameter']:g}"
        if f.get("cb_diameter"):
            d += f", cb Ø{f['cb_diameter']:g}×{f['cb_depth']:g}"
        if f.get("cs_diameter"):
            d += f", cs Ø{f['cs_diameter']:g} {f['cs_angle']:g}°"
        return d + (", through" if f["through"] else f", depth {f['depth']:g}")
    if f["type"] == "fillet":
        return f"{f['id']}: fillet " + ("free-form" if f["radius"] is None else f"R{f['radius']:g}")
    if f["type"] == "chamfer":
        size = f"{f['distance']:g}" if f["distance"] else "×".join(f"{v:g}" for v in f["distances"])
        return f"{f['id']}: chamfer {size}"
    ext = "through" if f.get("through") else f"{f.get('length', f.get('depth')):g} mm"
    return f"{f['id']}: {f['type']} {f['subtype']}, {ext}"


def render_features(res: dict, step: Path, out: Path, views: list[str], max_legend: int = 12) -> None:
    # _vtk_render looks colours up in identify.CLASS_COLORS: register the unrecognised colour there
    CLASS_COLORS.setdefault(UNRECOGNISED, (1.0, 0.85, 0.0))
    by_id = {f["id"]: f for f in res["features"]}
    cls = [face_class(by_id.get(r["feature"])) for r in res["faces"]]
    shape = load_step(step)
    fig, axes = plt.subplots(1, len(views), figsize=(8 * len(views), 7), dpi=110, squeeze=False)
    for ax, v in zip(axes[0], views):
        title, el, az = VIEWS[v]
        img, _ = _vtk_render(shape, cls, size=(900, 750), elev_deg=el, azim_deg=az)
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    handles = [Patch(color=CLASS_COLORS[face_class(f)], label=describe(f)) for f in res["features"][:max_legend]]
    if len(res["features"]) > max_legend:
        handles.append(Patch(color="white", label=f"… {len(res['features']) - max_legend} more"))
    if res["unrecognised_faces"]:
        handles.append(Patch(color=CLASS_COLORS[UNRECOGNISED], label=f"unrecognised ({len(res['unrecognised_faces'])} faces)"))
    ncol = min(len(handles), 3) if len(handles) > 4 else max(1, len(handles))
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=10, frameon=False)
    size = " × ".join(f"{v:g}" for v in res["geometry"]["size"])
    fig.suptitle(f"{step.name}  ({size} mm, {res['summary']['faces']} faces)", fontsize=13)
    rows = -(-len(handles) // ncol)
    fig.tight_layout(rect=(0, min(0.04 + 0.035 * rows, 0.4), 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", type=Path, help="<part>_features.json written by features.extract")
    ap.add_argument("--step", type=Path, default=None, help="STEP file (default: the JSON's 'part')")
    ap.add_argument("--out", type=Path, default=None, help="PNG path (default: <part>_render.png next to the JSON)")
    ap.add_argument("--views", nargs="+", default=["above", "below"], choices=sorted(VIEWS))
    args = ap.parse_args()

    res = json.loads(args.json.read_text(encoding="utf-8"))
    step = args.step or Path(res["part"])
    out = args.out or args.json.with_name(args.json.name.replace("_features.json", "") + "_render.png")
    render_features(res, step, out, args.views)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
