"""Reconstruct a design history from a STEP solid using its exact geometry (no learning).

    .venv\\Scripts\\python.exe -m recon.reconstruct part.stp --out out/part

Pipeline (see ``recon.candidates`` for how candidates are read from the B-Rep):

1. Sample the target (uniform + just inside / outside every face) and classify the samples.
2. Generate exact pad / pocket / hole candidates; pick them greedily by the reduction of the
   weighted point error, then drop candidates that became redundant.
3. Rebuild with the CAD kernel (``cadseq.build.Builder``); features the kernel rejects are skipped.
4. Blends: target faces still unexplained that are fillets (tangent cylinder / torus) or chamfers
   (angled planar strip) are applied on the sharp edges recovered from their support faces
   (``cadseq.blends``), kept only if they reduce the error.
5. Mirror: if the target is symmetric about a plane through its bounding-box centre, the half
   part is reconstructed and mirrored; the shorter history of equal quality wins.

Writes <out>.json (design history, strictly rebuildable), <out>_rebuilt.step,
<out>_report.json and <out>_unexplained.step (target faces coloured: grey explained, red not).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cadquery as cq
import numpy as np

from cadseq.blends import sharp_edges
from cadseq.build import BuildError, Builder, build, edge_ref, resolve_edge
from cadseq.metrics import mc_iou
from cadseq.schema import ID_PREFIX, CADSequence, EdgeRef

from .candidates import Candidate, PointSet, directions, extrusion_candidates, hole_candidates, make_points
from .geom import analyse_faces, classify

EXPLAINED = 0.9  # a target face is explained if ≥ 90 % of its inside / outside samples are right


def _error(ps: PointSet, pred: np.ndarray) -> float:
    return float(ps.w[pred != ps.y].sum())


def simulate(cands: list[Candidate], seq: list[int], n: int) -> np.ndarray:
    b = np.zeros(n, dtype=bool)
    for k in seq:
        b = b | cands[k].mask if cands[k].op == "pad" else b & ~cands[k].mask
    return b


def _delta_errors(cands, is_pad, M_chunks, b, ps) -> np.ndarray:
    y, w = ps.y.astype(float), ps.w
    s_pad = ((~b) * (1 - 2 * y) * w).astype(np.float32)
    s_cut = (b * (2 * y - 1) * w).astype(np.float32)
    dE = np.empty(len(cands))
    for i, M in M_chunks:
        n = len(M)
        dE[i:i + n] = np.where(is_pad[i:i + n], M @ s_pad, M @ s_cut)
    return dE


def greedy(cands: list[Candidate], ps: PointSet, beam: int = 6, max_features: int = 40,
           feature_cost: float = 2e-4, chunk: int = 256) -> list[int]:
    """Beam search over candidate sequences minimising the weighted point error plus a small cost
    per feature (prefers short histories). A plain greedy pick is myopic: e.g. a pad over the full
    height that slightly overshoots a lower rib beats the exact pads, and no candidate can remove
    the overshoot afterwards."""
    if not cands:
        return []
    y, w = ps.y, ps.w
    is_pad = np.array([c.op == "pad" for c in cands])
    prefer = np.array([c.prefer for c in cands])
    M_chunks = [(i, np.stack([c.mask for c in cands[i:i + chunk]]).astype(np.float32))
                for i in range(0, len(cands), chunk)]
    W = float(w.sum())
    eps, cost = 1e-4 * W, feature_cost * W
    start = (float((w * y).sum()), [], np.zeros(len(y), dtype=bool))
    states, finished = [start], [start]
    for _ in range(max_features):
        expansions = []
        for E, seq, b in states:
            dE = _delta_errors(cands, is_pad, M_chunks, b, ps)
            used = set(seq)
            for k in np.argsort(dE - prefer)[: beam + len(used)]:
                k = int(k)
                if k in used or dE[k] > -eps:
                    continue
                nb = b | cands[k].mask if is_pad[k] else b & ~cands[k].mask
                expansions.append((E + float(dE[k]), seq + [k], nb))
        if not expansions:
            break
        expansions.sort(key=lambda s: s[0] + cost * len(s[1]))
        states, seen = [], set()
        for s in expansions:
            key = hash(s[2].tobytes())
            if key not in seen:
                seen.add(key)
                states.append(s)
            if len(states) == beam:
                break
        finished += states
    chosen = min(finished, key=lambda s: s[0] + cost * len(s[1]))[1]
    # drop features whose removal does not increase the error
    changed = True
    while changed and len(chosen) > 1:
        changed = False
        base = _error(ps, simulate(cands, chosen, len(y)))
        for j in range(len(chosen)):
            trial = chosen[:j] + chosen[j + 1:]
            if cands[trial[0]].op == "pad" and _error(ps, simulate(cands, trial, len(y))) <= base + 1e-12:
                chosen, changed = trial, True
                break
    return chosen


class History:
    """Feature groups (sketch + pad / pocket, hole, fillet, ...) with canonical ids."""

    def __init__(self):
        self.groups: list[list[dict]] = []
        self.descs: list[str] = []

    def sequence(self, groups=None) -> CADSequence:
        groups = self.groups if groups is None else groups
        count: dict = {}
        feats = []
        for g in groups:
            sketch_id = None
            for f in g:
                f = dict(f)
                count[f["type"]] = count.get(f["type"], 0) + 1
                f["id"] = f"{ID_PREFIX[f['type']]}{count[f['type']]}"
                if f["type"] == "sketch":
                    sketch_id = f["id"]
                elif f["type"] in ("pad", "pocket"):
                    f["sketch"] = sketch_id
                feats.append(f)
        return CADSequence.model_validate({"features": feats})

    def n_features(self) -> int:
        return sum(1 for g in self.groups for f in g if f["type"] != "sketch")


def kernel_build(groups: list[list[dict]], descs: list[str], log: list) -> tuple:
    """Apply groups one by one; groups the kernel rejects are skipped.
    Returns (body, History, indices of the skipped groups)."""
    hist, builder, skipped = History(), Builder(check_each=True), []
    for gi, (g, desc) in enumerate(zip(groups, descs)):
        snap = builder.snapshot()
        try:
            seq = hist.sequence(hist.groups + [g])
            for f in seq.features[sum(len(x) for x in hist.groups):]:
                builder.apply(f)
        except (BuildError, ValueError) as e:
            builder.restore(snap)
            log.append(f"skipped {desc}: {str(e)[:120]}")
            skipped.append(gi)
            continue
        hist.groups.append(g)
        hist.descs.append(desc)
    return builder.body, hist, skipped


# ----------------------------------------------------------------- blends --

def _chamfer_distance(shape, face_idx: int, ve) -> float | None:
    """Distance from the recovered sharp edge to the chamfer face's edges on its supports."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    from .geom import face_list

    faces = face_list(shape)
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape.wrapped, TopAbs_EDGE, TopAbs_FACE, m)
    ds = []
    exp = TopExp_Explorer(faces[face_idx], TopAbs_EDGE)
    while exp.More():
        e = TopoDS.Edge_s(exp.Current())
        exp.Next()
        if not m.Contains(e):
            continue
        if any(any(g.IsSame(faces[s]) for g in m.FindFromKey(e)) for s in ve.supports):
            c = BRepAdaptor_Curve(e)
            p = c.Value(0.5 * (c.FirstParameter() + c.LastParameter()))
            proj = GeomAPI_ProjectPointOnCurve(p, ve.geom)
            if proj.NbPoints():
                ds.append(proj.LowerDistance())
    return float(np.mean(ds)) if ds else None


def blend_candidates(shape, faces, explained: np.ndarray) -> list[dict]:
    kinds = {}
    for f in faces:
        if explained[f.index]:
            continue
        if f.kind in ("cylinder", "torus") and not (f.kind == "cylinder" and f.span and f.span > 2 * math.pi - 1e-3):
            kinds[f.index] = "fillet"
        elif f.kind == "plane":
            kinds[f.index] = "chamfer"
    out = []
    for ve in sharp_edges(shape, kinds) if kinds else []:
        f = faces[ve.face]
        kind = kinds[ve.face]
        if kind == "fillet":
            size = f.minor if f.kind == "torus" else f.radius
        else:
            size = _chamfer_distance(shape, ve.face, ve)
        if size and size > 1e-6:
            out.append({"type": kind, "size": float(size), "point": ve.point, "curve": ve.curve, "face": ve.face})
    return out


def apply_blends(target, faces, ps: PointSet, body, hist: History, log: list):
    pred = classify(body, ps.p)
    acc = face_accuracy(ps, pred, len(faces))
    blends = blend_candidates(target, faces, acc >= EXPLAINED)
    diag = target.BoundingBox().DiagonalLength
    groups: dict = {}
    for b in blends:
        groups.setdefault((b["type"], round(b["size"], 6)), []).append(b)
    order = sorted(groups, key=lambda k: (k[0] != "fillet", -k[1]))
    for key in order:
        items = groups[key]
        trials = [items] + ([[it] for it in items] if len(items) > 1 else [])
        done = set()
        for trial in trials:
            if any(id(it) in done for it in trial):
                continue
            try:
                edges, seen = [], set()
                for it in trial:
                    e = resolve_edge(body, EdgeRef(point=tuple(it["point"]), curve=it["curve"]), 1e-4 * diag + 1e-3)
                    if e.hashCode() not in seen:
                        seen.add(e.hashCode())
                        edges.append(e)
                new = body.fillet(key[1], edges) if key[0] == "fillet" else body.chamfer(key[1], None, edges)
                solids = new.clean().Solids()
                if len(solids) != 1 or not solids[0].isValid():
                    raise BuildError("blend result is not one valid solid")
                new = solids[0]
            except Exception as e:  # kernel failures
                log.append(f"blend {key[0]} {key[1]:.4g} × {len(trial)} failed: {str(e)[:80]}")
                continue
            # score on the samples near the blended faces only
            near = np.zeros(len(ps.p), dtype=bool)
            for it in trial:
                fp = faces[it["face"]].points
                lo, hi = fp.min(0) - 2 * key[1] - 1e-3, fp.max(0) + 2 * key[1] + 1e-3
                near |= np.all((ps.p >= lo) & (ps.p <= hi), axis=1)
            new_pred = pred.copy()
            new_pred[near] = classify(new, ps.p[near])
            if _error(ps, new_pred) < _error(ps, pred) - 1e-9:
                feat = {"type": key[0], "edges": [edge_ref(e).model_dump() for e in edges]}
                feat["radius" if key[0] == "fillet" else "distance"] = key[1]
                hist.groups.append([feat])
                hist.descs.append(f"{key[0]} {key[1]:.4g} on {len(edges)} edge(s)")
                body, pred = new, new_pred
                done.update(id(it) for it in trial)
            else:
                log.append(f"blend {key[0]} {key[1]:.4g} × {len(trial)} did not improve")
    return body, hist, pred


def face_accuracy(ps: PointSet, pred: np.ndarray, n_faces: int) -> np.ndarray:
    acc = np.ones(n_faces)
    for i in range(n_faces):
        sel = ps.face == i
        if sel.any():
            acc[i] = float((pred[sel] == ps.y[sel]).mean())
    return acc


# ----------------------------------------------------------------- mirror --

def symmetry_planes(shape, faces, ps: PointSet, n_check: int = 3000, seed: int = 0) -> list[tuple]:
    c = np.array(shape.BoundingBox().center.toTuple())
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ps.p), min(n_check, len(ps.p)), replace=False)
    out = []
    for d in directions([f for f in faces if f.kind == "plane"]):
        p = ps.p[idx]
        q = p - 2 * np.outer((p - c) @ d, d)
        if (classify(shape, q) == ps.y[idx]).mean() > 0.999:
            out.append((c, d))
    return out


def half_space(shape, c, d):
    bb = shape.BoundingBox()
    size = 3.0 * bb.DiagonalLength + 10.0
    face = cq.Face.makePlane(size, size, cq.Vector(*c), cq.Vector(*d))
    box = cq.Solid.extrudeLinear(face, cq.Vector(*(d * size)))
    solids = shape.intersect(box).Solids()
    return solids[0] if len(solids) == 1 else None


# ------------------------------------------------------------------- main --

def prismatic_history(target, log: list, stats: dict, n_uniform: int, rounds: int = 4):
    """Steps 1–3 on ``target``: returns (body, History, faces, PointSet). Candidates the kernel
    rejects (e.g. the point model misses that a cut disconnects the body) are banned and the
    search re-planned; the plan with the lowest kernel-verified error is kept."""
    faces = analyse_faces(target)
    ps = make_points(target, faces, n_uniform)
    cands = extrusion_candidates(target, faces, ps, stats) + hole_candidates(target, faces, ps)
    stats["candidates"] = stats.get("candidates", 0) + len(cands)
    banned: set = set()
    best = None
    for _ in range(rounds):
        idx = [i for i in range(len(cands)) if i not in banned]
        chosen = [idx[k] for k in greedy([cands[i] for i in idx], ps)]
        rlog: list[str] = []
        body, hist, skipped = kernel_build([cands[k].features for k in chosen], [cands[k].desc for k in chosen], rlog)
        err = _error(ps, classify(body, ps.p)) if body is not None else math.inf
        if best is None or err < best[0] - 1e-12:
            best = (err, body, hist, rlog)
        if not skipped:
            break
        banned.update(chosen[k] for k in skipped)
    log += best[3]
    return best[1], best[2], faces, ps


def reconstruct(target, n_uniform: int = 16384, try_mirror: bool = True) -> dict:
    t0 = time.time()
    log: list[str] = []
    stats: dict = {}
    body, hist, faces, ps = prismatic_history(target, log, stats, n_uniform)
    results = []
    if body is not None:
        body, hist, pred = apply_blends(target, faces, ps, body, hist, log)
        results.append({"body": body, "hist": hist, "pred": pred, "mirror": None})
    if try_mirror:
        for c, d in symmetry_planes(target, faces, ps):
            half = half_space(target, c, d)
            if half is None:
                continue
            mlog: list[str] = []
            hbody, hhist, _, _ = prismatic_history(half, mlog, stats, n_uniform)
            if hbody is None:
                continue
            mirror = {"type": "mirror", "plane": {"origin": tuple(map(float, c)), "normal": tuple(map(float, d))}}
            try:
                b2 = hbody.fuse(hbody.mirror(cq.Vector(*d), cq.Vector(*c))).clean().Solids()
            except Exception:
                continue
            if len(b2) != 1:
                continue
            hhist.groups.append([mirror])
            hhist.descs.append(f"mirror normal {np.round(d, 3).tolist()}")
            mbody, mhist, mpred = apply_blends(target, faces, ps, b2[0], hhist, mlog)
            results.append({"body": mbody, "hist": mhist, "pred": mpred, "mirror": d, "log": mlog})
    if not results:
        raise RuntimeError("no body could be built: " + "; ".join(log[-3:]))
    for r in results:
        r["error"] = _error(ps, r["pred"])
    best_err = min(r["error"] for r in results)
    ok = [r for r in results if r["error"] <= best_err + 2e-3]
    best = min(ok, key=lambda r: (r["hist"].n_features(), r["error"]))
    log += best.get("log", [])
    acc = face_accuracy(ps, best["pred"], len(faces))
    return {"body": best["body"], "history": best["hist"], "faces": faces, "face_acc": acc,
            "error": best["error"], "log": log, "stats": stats, "seconds": time.time() - t0,
            "mirror": best["mirror"] is not None}


def write_outputs(target, res: dict, out: Path) -> dict:
    from identify import write_colored_step

    out.parent.mkdir(parents=True, exist_ok=True)
    seq = res["history"].sequence()
    (out.parent / f"{out.name}.json").write_text(json.dumps(seq.model_dump(), indent=1), encoding="utf-8")
    body = res["body"]
    cq.exporters.export(cq.Workplane().add(body), str(out.parent / f"{out.name}_rebuilt.step"),
                        cq.exporters.ExportTypes.STEP)
    try:
        strict = build(seq)
        strict_ok = abs(strict.Volume() - body.Volume()) <= 1e-6 * max(1.0, abs(body.Volume()))
    except BuildError as e:
        strict_ok = False
        res["log"].append(f"strict rebuild of the JSON failed: {e}")
    acc = res["face_acc"]
    unexplained = [int(i) for i in np.nonzero(acc < EXPLAINED)[0]]
    colors = [(0.85, 0.15, 0.15) if a < EXPLAINED else (0.75, 0.75, 0.78) for a in acc]
    write_colored_step(target.wrapped, colors, out.parent / f"{out.name}_unexplained.step")
    report = {
        "iou": round(mc_iou(body, target), 4),
        "volume_target": round(target.Volume(), 3), "volume_rebuilt": round(body.Volume(), 3),
        "strict_rebuild": strict_ok, "features": res["history"].n_features(),
        "history": res["history"].descs, "mirror": res["mirror"],
        "faces": len(acc), "unexplained_faces": unexplained,
        "unexplained_kinds": sorted({res["faces"][i].kind for i in unexplained}),
        "candidates": res["stats"].get("candidates", 0),
        "unsupported_sections": res["stats"].get("unsupported_sections", 0),
        "seconds": round(res["seconds"], 1), "log": res["log"],
    }
    (out.parent / f"{out.name}_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def load_target(path: Path):
    solids = cq.importers.importStep(str(path)).solids().vals()
    if len(solids) != 1:
        raise SystemExit(f"expected one solid in {path}, found {len(solids)}")
    return solids[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="output path prefix")
    ap.add_argument("--uniform", type=int, default=16384, help="uniform sample points")
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()

    target = load_target(args.step)
    res = reconstruct(target, args.uniform, not args.no_mirror)
    rep = write_outputs(target, res, args.out)
    for i, d in enumerate(rep["history"], 1):
        print(f"  {i:>2}. {d}")
    print(f"features {rep['features']} | IoU {rep['iou']:.4f} | strict rebuild {rep['strict_rebuild']} | "
          f"unexplained faces {len(rep['unexplained_faces'])}/{rep['faces']} {rep['unexplained_kinds']} | "
          f"{rep['candidates']} candidates | {rep['seconds']} s")
    for line in rep["log"]:
        print("  log:", line)
    print(f"wrote {args.out}.json / _rebuilt.step / _report.json / _unexplained.step")


if __name__ == "__main__":
    main()
