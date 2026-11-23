"""STEP solid → structured JSON of its features (rule-based, cross-checked by the segmenter).

    .venv\\Scripts\\python.exe -m features.extract part.stp --out runs\\features\\part
    .venv\\Scripts\\python.exe -m features.extract part.stp --no-seg          # rules only
    .venv\\Scripts\\python.exe -m features.extract part.stp --no-extrusions   # holes / blends only (fast)

Writes <out>_features.json and <out>_features.step (input solid, faces coloured by feature type;
unrecognised faces in yellow). The JSON layout is documented in features/README.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from recon.geom import analyse_faces
from recon.reconstruct import load_target

from .recognize import (adjacency, chamfer_features, chamfer_items, fillet_features, fillet_items, find_patterns, r6,
                        recognize_extrusions, recognize_holes, symmetry)

SCHEMA_VERSION = "features-1.0"
COLORS = {"extrude": (0.62, 0.70, 0.80), "cut": (0.98, 0.62, 0.25), "hole:simple": (0.90, 0.20, 0.20),
          "hole:counterbore": (0.55, 0.25, 0.75), "hole:countersink": (0.90, 0.30, 0.65),
          "hole:stepped": (0.55, 0.25, 0.75), "fillet": (0.25, 0.70, 0.35), "chamfer": (0.15, 0.65, 0.70),
          None: (1.0, 0.85, 0.0)}
ID_PREFIX = {"extrude": "E", "cut": "K", "hole": "H", "fillet": "F", "chamfer": "C"}
PATTERN_TOL = 0.05  # mm: hole positions in a pattern (CAD exports and the synthetic grid are within this)


def clean(o):
    """numpy → JSON types, floats rounded to 1e-6 mm (``r6``)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v) for v in o.tolist()]
    if isinstance(o, (float, np.floating, np.integer, np.bool_)) and not isinstance(o, bool):
        return r6(o)
    return o


def _hole_as_pocket(h: dict) -> dict:
    """A simple hole re-expressed as a circular cut (same exact geometry)."""
    d = np.asarray(h["direction"], float)
    p = np.asarray(h["position"], float)
    f = {"type": "cut", "subtype": "circle", "direction": d,
         "sketch_plane": {"origin": p, "normal": -d},
         "profile": {"shape": "circle", "diameter": h["diameter"], "area": np.pi * h["diameter"] ** 2 / 4},
         "inner_loops": 0, "start_level": float(p @ -d)}
    if h["through"]:
        f["through"] = True
    else:
        f.update(depth=h["depth"], through=False)
    f.update(faces=list(h["faces"]), end_faces=[], _hole=h)
    return f


def extract(step: Path, n_uniform: int = 16384, extrusions: bool = True, ckpt: Path | None = None) -> tuple:
    """Returns (result, target solid, segmenter labels or None)."""
    t0 = time.time()
    target = load_target(step)
    faces = analyse_faces(target)
    fmap, shared = adjacency(target)
    bb = target.BoundingBox()
    diag = bb.DiagonalLength
    tol = 1e-6 * max(1.0, diag)
    log: list[str] = []

    seg = None
    if ckpt is not None:
        from .crosscheck import segment_faces

        try:
            seg = segment_faces(step, ckpt, faces)
            seg["model"] = str(ckpt)
        except Exception as e:  # missing checkpoint / torch: the rules-only result is still valid
            log.append(f"segmenter cross-check skipped: {type(e).__name__}: {e}")

    holes = recognize_holes(faces, shared, tol)
    hole_faces = {i for h in holes for i in h["faces"]}
    f_items = fillet_items(target, faces, hole_faces, shared, tol)
    c_items = chamfer_items(target, faces, hole_faces | {it["face"] for it in f_items}, shared, fmap, tol)
    extr, sym, ambiguous = [], None, []
    # A plain blind / through hole is geometrically identical to a circular pocket; the segmenter
    # (trained on modelling histories) decides which a designer would call it. Without it: hole.
    pockets = []
    if seg:
        keep = []
        for h in holes:
            cls = [seg["faces"][i][0] for i in h["faces"]]
            if h["subtype"] == "simple" and sum(c.startswith("cut") for c in cls) > len(cls) / 2:
                pockets.append(_hole_as_pocket(h))
                ambiguous.append({"faces": h["faces"], "readings": ["hole", "circular pocket"],
                                  "chosen": "circular pocket", "by": "segmenter"})
            else:
                keep.append(h)
        holes = keep
    if extrusions:
        extr, rfaces, ps = recognize_extrusions(target, hole_faces, holes + [p["_hole"] for p in pockets],
                                                n_uniform, tol, log)
        sym = symmetry(target, rfaces, ps)
        # A fillet on an edge parallel to an extrusion is geometrically identical to an arc in the
        # extrusion's sketch (a chamfer to a sketch line). The segmenter breaks the tie; without it
        # the blend reading is kept (more useful in review) and the extrusion is listed as alternative.
        claim = {i: k for k, e in enumerate(extr) for i in e["faces"]}
        for items, kind in ((f_items, "fillet"), (c_items, "chamfer")):
            keep = []
            for it in items:
                k = claim.get(it["face"])
                if k is None:
                    keep.append(it)
                    continue
                seg_cls = seg["faces"][it["face"]][0] if seg else None
                as_extrusion = seg_cls is not None and seg_cls.startswith(("extrude", "cut"))
                ambiguous.append({"faces": [it["face"]], "readings": [kind, ("side of", k)],
                                  "chosen": "extrusion side" if as_extrusion else kind,
                                  "by": "segmenter" if seg else "default"})
                if not as_extrusion:
                    extr[k]["faces"].remove(it["face"])
                    it["alternatives"] = [k]
                    keep.append(it)
            items[:] = keep
        for e in extr:
            e["end_faces"] = [i for i in e["end_faces"] if i in e["faces"]]
    for p in pockets:
        del p["_hole"]
    extr += pockets
    fillets = fillet_features(f_items, shared)
    chamfers = chamfer_features(c_items, shared)
    # blend corner patches: small faces surrounded only by fillet / chamfer faces (where a chamfer
    # meets a fillet, or three blends meet) belong to the adjacent blend (chamfer first)
    taken = {i for f in extr + holes + fillets + chamfers for i in f.get("faces", [])}
    blend_of = {i: f for f in chamfers + fillets for i in f["faces"]}
    for fi in faces:
        nbs = list(shared.get(fi.index, {}))
        if fi.index in taken or len(nbs) < 2 or not all(j in blend_of for j in nbs):
            continue
        owner_f = next((blend_of[j] for j in nbs if blend_of[j]["type"] == "chamfer"), blend_of[nbs[0]])
        owner_f["faces"] = sorted(owner_f["faces"] + [fi.index])
        owner_f.setdefault("corner_faces", []).append(fi.index)

    features = extr + holes + fillets + chamfers
    counts: dict = {}
    for f in features:
        counts[f["type"]] = counts.get(f["type"], 0) + 1
        f["id"] = f"{ID_PREFIX[f['type']]}{counts[f['type']]}"
    for f in fillets + chamfers:
        f["alternatives"] = [extr[k]["id"] for k in f["alternatives"]]
    for a in ambiguous:
        a["readings"] = [f"side of {extr[r[1]]['id']}" if isinstance(r, tuple) else r for r in a["readings"]]
    for f in features:  # id first, for readability
        fid = f.pop("id")
        items = list(f.items())
        f.clear()
        f.update({"id": fid, **dict(items)})
    patterns = find_patterns(holes, PATTERN_TOL)
    owner = {i: f["id"] for f in features for i in f.get("faces", [])}
    face_rows = [{"index": f.index, "surface": f.kind, "area": f.area, "feature": owner.get(f.index)} for f in faces]
    unrecognised = [{"face": f.index, "surface": f.kind, "area": f.area} for f in faces if f.index not in owner]

    summary: dict = {"faces": len(faces), "faces_explained": len(faces) - len(unrecognised),
                     "features": dict(counts), "hole_patterns": len(patterns)}
    for h in holes:
        k = f"hole_{h['subtype']}"
        summary[k] = summary.get(k, 0) + 1
    summary["fillet_radii"] = sorted({round(f["radius"], 4) for f in fillets if f["radius"]})
    summary["chamfer_sizes"] = sorted({round(c["distance"], 4) for c in chamfers if c["distance"]})
    summary["hole_diameters"] = sorted({round(h["diameter"], 4) for h in holes})
    return {
        "schema": SCHEMA_VERSION, "part": str(step), "units": "mm",
        "geometry": {"bbox": {"min": [bb.xmin, bb.ymin, bb.zmin], "max": [bb.xmax, bb.ymax, bb.zmax]},
                     "size": [bb.xlen, bb.ylen, bb.zlen], "volume": target.Volume(), "area": target.Area(),
                     "surface_types": {k: sum(f.kind == k for f in faces) for k in sorted({f.kind for f in faces})}},
        "symmetry_planes": sym,
        "features": features, "patterns": patterns,
        "ambiguous_faces": ambiguous,
        "unrecognised_faces": unrecognised, "faces": face_rows,
        "summary": summary, "log": log, "seconds": time.time() - t0,
    }, target, seg


def write_colored(target, result: dict, path: Path) -> None:
    from identify import write_colored_step

    by_id = {f["id"]: f for f in result["features"]}
    cols = []
    for row in result["faces"]:
        f = by_id.get(row["feature"])
        key = None if f is None else f"hole:{f['subtype']}" if f["type"] == "hole" else f["type"]
        cols.append(COLORS.get(key, COLORS[None]))
    write_colored_step(target.wrapped, cols, path)


def print_summary(res: dict) -> None:
    g = res["geometry"]
    print(f"{Path(res['part']).name}: {res['summary']['faces']} faces, size "
          f"{' x '.join(f'{v:.4g}' for v in g['size'])} mm, {res['summary']['faces_explained']} faces explained")
    for f in res["features"]:
        if f["type"] == "hole":
            extra = f"Ø{f['diameter']:.4g} {'through' if f['through'] else 'depth %.4g' % f['depth']}"
            if f["subtype"] == "counterbore":
                extra += f" cb Ø{f['cb_diameter']:.4g}×{f['cb_depth']:.4g}"
            elif f["subtype"] == "countersink":
                extra += f" cs Ø{f['cs_diameter']:.4g} {f['cs_angle']:.4g}°"
            desc = f"hole {f['subtype']} {extra}"
        elif f["type"] == "fillet":
            r = "freeform" if f["radius"] is None else f"R{f['radius']:.4g}"
            desc = f"fillet {r} {f['convexity'] or ''} on {len(f['edges'])} edge(s)"
        elif f["type"] == "chamfer":
            d = f"{f['distance']:.4g}" if f["distance"] else "×".join(f"{v:.4g}" for v in f["distances"])
            desc = f"chamfer {d} on {len(f['edges'])} edge(s)"
        else:
            p = f["profile"]
            dims = {k: p[k] for k in ("diameter", "width", "height", "length", "corner_radius", "sides") if k in p}
            ext = "through" if f.get("through") else f"{f.get('length', f.get('depth')):.4g}"
            desc = (f"{f['type']} {f['subtype']} {', '.join(f'{k} {v:.4g}' for k, v in dims.items())} "
                    f"along {[round(v, 3) for v in f['direction']]} {ext}")
        seg = f.get("seg", {})
        s = f" [seg: {seg['status']}]" if seg else ""
        print(f"  {f['id']:<4} {desc}  faces {f['faces']}{s}")
    for p in res["patterns"]:
        extra = {"linear": lambda: f"pitch {p['pitch']:.4g}",
                 "grid": lambda: f"{p['counts'][0]}×{p['counts'][1]} pitch {p['pitches'][0]:.4g}/{p['pitches'][1]:.4g}",
                 "circular": lambda: f"PCD {p['pcd']:.4g} step {p['angle_step']:.4g}°",
                 "identical_set": lambda: ""}[p["kind"]]()
        print(f"  pattern {p['kind']} {p['count']}× Ø{p['hole']['diameter']:.4g} {extra} {p['holes']}")
    if res.get("symmetry_planes"):
        print("  symmetry planes, normals:", [[round(v, 3) for v in s["normal"]] for s in res["symmetry_planes"]])
    if res["unrecognised_faces"]:
        print("  unrecognised faces:", [u["face"] for u in res["unrecognised_faces"]])
    for a in res.get("ambiguous_faces", []):
        print(f"  ambiguous faces {a['faces']}: {' or '.join(a['readings'])} → {a['chosen']} ({a['by']})")
    cc = res.get("cross_check")
    if cc:
        print(f"  segmenter cross-check: {cc['features']}, {len(cc['conflicts'])} conflict(s)")
        for c in cc["conflicts"]:
            print(f"    conflict: face {c['face']} in {c['rule_feature']}, segmenter says {c['seg_class']} ({c['seg_prob']:.2f})")
    print(f"  {res['seconds']:.1f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="output prefix (default runs/features/<name>)")
    ap.add_argument("--ckpt", type=Path, default=Path("runs/seg/last.pt"), help="segmenter checkpoint")
    ap.add_argument("--no-seg", action="store_true", help="skip the segmenter cross-check")
    ap.add_argument("--no-extrusions", action="store_true", help="skip extrude / cut recognition")
    ap.add_argument("--uniform", type=int, default=16384, help="uniform sample points (extrusions)")
    args = ap.parse_args()

    out = args.out or Path("runs/features") / args.step.stem
    out.parent.mkdir(parents=True, exist_ok=True)
    res, target, seg = extract(args.step, args.uniform, not args.no_extrusions, None if args.no_seg else args.ckpt)
    if seg is not None:
        from .crosscheck import annotate

        annotate(res, seg)
    elif not args.no_seg:
        res["cross_check"] = {"error": next((l for l in res["log"] if "cross-check skipped" in l), "unavailable")}
    res = clean(res)
    (out.parent / f"{out.name}_features.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    write_colored(target, res, out.parent / f"{out.name}_features.step")
    print_summary(res)
    print(f"wrote {out}_features.json / _features.step")


if __name__ == "__main__":
    main()
