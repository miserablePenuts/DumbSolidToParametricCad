"""Feature candidates read from the exact B-Rep of the target, and the weighted point set that
scores them.

Every candidate is a *tool* whose parameters come from the target's own geometry — nothing is
estimated:

* **pad / pocket** — for each extrusion direction d (normals of planar faces, axes of cylinders and
  cones) the planar faces perpendicular to d define *levels*. Between consecutive levels the target
  is a prism along d, so its cross-section at the slab middle (an exact OCC section) is a sketch.
  Pad = section outer loop (optionally with its inner loops) extruded over any span of levels that
  contains the slab; pocket = an inner loop cut over such a span (through-all if it spans the part);
  trim = cut everything *outside* the outer loop over such a span (sketch: a frame with the outline
  as island) — intersects prisms of different directions, e.g. a side profile with a front outline.
* **hole** — stacks of coaxial concave full cylinders (+ cones): simple, counterbore and countersink
  holes, blind or through, entered from either end.

A tool's effect is evaluated on sample points: uniform points in the bounding box plus points just
inside / outside every target face (each face gets the same total weight, so small features such as
holes and chamfers matter as much as the bulk). The membership test of each tool is analytic
(point-in-polygon in the sketch frame, distance to the hole axis), so no boolean is needed to score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cadseq.schema import Line

from .geom import (FaceInfo, Unsupported, canonical_dir, classify, face_loops, frame, in_loop,
                   section_faces)


@dataclass
class PointSet:
    p: np.ndarray        # (N, 3)
    y: np.ndarray        # (N,) inside the target
    w: np.ndarray        # (N,) weight
    face: np.ndarray     # (N,) target face of a surface sample, −1 for uniform points
    delta: float         # offset of the surface samples


def make_points(shape, faces: list[FaceInfo], n_uniform: int = 16384, seed: int = 0) -> PointSet:
    bb = shape.BoundingBox()
    lo, hi = np.array([bb.xmin, bb.ymin, bb.zmin]), np.array([bb.xmax, bb.ymax, bb.zmax])
    pad = 0.01 * (hi - lo) + 1e-3
    delta = float(min(0.05, 1e-3 * bb.DiagonalLength))
    rng = np.random.default_rng(seed)
    uni = lo - pad + (hi - lo + 2 * pad) * rng.random((n_uniform, 3))
    surf, fid = [], []
    for f in faces:
        if len(f.points):
            surf += [f.points - delta * f.normals, f.points + delta * f.normals]
            fid += [f.index] * (2 * len(f.points))
    surf = np.concatenate(surf) if surf else np.zeros((0, 3))
    p = np.concatenate([uni, surf])
    face = np.concatenate([np.full(len(uni), -1), np.array(fid, dtype=int)])
    w = np.empty(len(p))
    w[: len(uni)] = 1.0 / len(uni)
    nf = max(1, len(set(fid)))
    for i in set(fid):
        sel = face == i
        w[sel] = 1.0 / nf / sel.sum()
    return PointSet(p, classify(shape, p), w, face, delta)


@dataclass
class Candidate:
    op: str                    # pad | pocket | hole
    mask: np.ndarray           # (N,) points inside the tool
    features: list[dict]       # schema feature dicts without ids
    desc: str
    prefer: float = 0.0        # tie-break bonus (fewer / more meaningful features)
    meta: dict = field(default_factory=dict)


def _dedupe_dirs(dirs, tol=1e-9):
    out = []
    for d in dirs:
        d = canonical_dir(d)
        if not any(abs(abs(d @ e) - 1) < tol for e in out):
            out.append(d)
    return out


def directions(faces: list[FaceInfo]) -> list[np.ndarray]:
    dirs = [f.normal for f in faces if f.kind == "plane"]
    dirs += [f.axis for f in faces if f.kind in ("cylinder", "cone")]
    return _dedupe_dirs(dirs)


def _levels(faces: list[FaceInfo], d: np.ndarray, extent: tuple, tol: float) -> list[float]:
    """Sorted distinct levels along d of the planar faces perpendicular to d, plus the part's
    extent along d where no such plane exists (e.g. a cylinder tangent to the bounding box)."""
    planes = sorted(float(f.points[0] @ d) for f in faces
                    if f.kind == "plane" and len(f.points) and abs(abs(f.normal @ d) - 1) < 1e-9)
    if not planes:
        return []
    out: list[float] = []
    for v in planes:
        if not out or v - out[-1] > tol:
            out.append(v)
    out += [v for v in extent if all(abs(v - u) > tol for u in out)]
    return sorted(out)


def _plane(d, x, h) -> dict:
    return {"origin": tuple(float(v) for v in d * h), "normal": tuple(float(v) for v in d),
            "x_dir": tuple(float(v) for v in x)}


def _profile(outer, inner=()) -> dict:
    return {"outer": [c.model_dump() for c in outer], "inner": [[c.model_dump() for c in l] for l in inner]}


def extrusion_candidates(shape, faces, ps: PointSet, stats: dict, max_span: int = 6) -> list[Candidate]:
    out: list[Candidate] = []
    verts = np.array([v.toTuple() for v in shape.Vertices()]) if shape.Vertices() else ps.p
    tess = np.array([v.toTuple() for v in shape.tessellate(0.01, 0.1)[0]])
    allp = np.concatenate([verts, tess])
    diag = shape.BoundingBox().DiagonalLength
    tol = 1e-7 * max(1.0, diag)
    seen = set()
    for d in directions(faces):
        proj = allp @ d
        L = _levels(faces, d, (float(proj.min()), float(proj.max())), max(tol, 1e-4))
        if len(L) < 2:
            continue
        x, y, _ = frame(d)
        uv = np.stack([ps.p @ x, ps.p @ y], 1)
        w = ps.p @ d
        K = len(L)
        pu, pv, mg = allp @ x, allp @ y, 0.05 * diag + 1.0
        u0, u1, v0, v1 = pu.min() - mg, pu.max() + mg, pv.min() - mg, pv.max() + mg
        corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
        rect = [Line(type="line", start=corners[k], end=corners[(k + 1) % 4]) for k in range(4)]
        for i in range(K - 1):
            for F in section_faces(shape, d, 0.5 * (L[i] + L[i + 1])):
                try:
                    outer, inner = face_loops(F, x, y)
                except Unsupported as e:
                    stats["unsupported_sections"] = stats.get("unsupported_sections", 0) + 1
                    stats.setdefault("unsupported_reasons", set()).add(str(e)[:60])
                    continue
                m_outer = in_loop(outer, uv)
                m_inner = [in_loop(l, uv) for l in inner]
                profiles = [("pad", outer, [], m_outer)]
                if inner:
                    m_full = m_outer.copy()
                    for m in m_inner:
                        m_full &= ~m
                    profiles.append(("pad", outer, inner, m_full))
                profiles += [("pocket", l, [], m) for l, m in zip(inner, m_inner)]
                # trim: cut everything outside the outline (sketch = frame with the outline as island);
                # combines with a pad along another direction (e.g. side profile × front outline)
                profiles.append(("trim", rect, [outer], ~m_outer))
                for op, o, inn, m2 in profiles:
                    sig = (op, str([c.model_dump() for c in o]), str([[c.model_dump() for c in l] for l in inn]))
                    for a in range(max(0, i - max_span + 1), i + 1):
                        for b in range(i + 1, min(K, a + max_span + 1)):
                            key = (tuple(np.round(d, 9)), sig, a, b)
                            if key in seen:
                                continue
                            seen.add(key)
                            through = op != "pad" and a == 0 and b == K - 1
                            lo_ = -np.inf if through else L[a]
                            mask = m2 & (w >= lo_) & (w <= L[b])
                            if not mask.any():
                                continue
                            if op == "pad":
                                feats = [{"type": "sketch", "plane": _plane(d, x, L[a]), "profiles": [_profile(o, inn)]},
                                         {"type": "pad", "length": L[b] - L[a], "direction": 1}]
                            else:
                                pk = {"type": "pocket", "direction": -1}
                                pk.update({"through_all": True} if through else {"depth": L[b] - L[a]})
                                feats = [{"type": "sketch", "plane": _plane(d, x, L[b]), "profiles": [_profile(o, inn)]}, pk]
                            n_curves = len(o) + sum(len(l) for l in inn)
                            out.append(Candidate("pad" if op == "pad" else "pocket", mask, feats,
                                                 f"{op} d={np.round(d, 3).tolist()} [{L[a]:.4g},{L[b]:.4g}] {n_curves} curves",
                                                 prefer=-1e-7 * n_curves - (1e-6 if op == "trim" else 0.0),
                                                 meta={"dir": d, "levels": (L[a], L[b])}))
    return out


# ------------------------------------------------------------------ holes --

def _same_axis(f: FaceInfo, g: FaceInfo, tol: float) -> bool:
    if abs(abs(f.axis @ g.axis) - 1) > 1e-9:
        return False
    dv = g.loc - f.loc
    return np.linalg.norm(dv - (dv @ f.axis) * f.axis) < tol


def hole_candidates(shape, faces: list[FaceInfo], ps: PointSet) -> list[Candidate]:
    diag = shape.BoundingBox().DiagonalLength
    tol = 1e-6 * max(1.0, diag)
    conc = [f for f in faces if f.kind in ("cylinder", "cone") and f.concave]
    # merge faces of one surface (split cylinders) into segments: (kind, r_range, s_range, span)
    axes: list[list[FaceInfo]] = []
    for f in conc:
        for grp in axes:
            if _same_axis(grp[0], f, tol):
                grp.append(f)
                break
        else:
            axes.append([f])
    out: list[Candidate] = []
    for grp in axes:
        a, loc = grp[0].axis, grp[0].loc
        segs = []  # merged segments with full angular coverage
        for f in grp:
            for sg in segs:
                if (sg["kind"] == f.kind and np.allclose(sg["r"], f.r_range, atol=tol)
                        and np.allclose(sg["s"], f.s_range, atol=tol)):
                    sg["span"] += f.span
                    break
            else:
                segs.append({"kind": f.kind, "r": f.r_range, "s": f.s_range, "span": f.span,
                             "semi": f.semi_angle})
        segs = sorted([s for s in segs if s["span"] > 2 * math.pi - 1e-3], key=lambda s: s["s"][0])
        rel = ps.p - loc
        s_pts = rel @ a + float(loc @ a)
        r_pts = np.linalg.norm(rel - np.outer(rel @ a, a), axis=1)
        for k, sg in enumerate(segs):
            if sg["kind"] != "cylinder":
                continue
            r, (s0, s1) = sg["r"][0], sg["s"]
            # neighbours on the axis: a larger cylinder (counterbore) or a cone (countersink)
            for sign in (1, -1):
                entry = s0 if sign == 1 else s1
                nb = [t for t in segs if t is not sg and abs((t["s"][1] if sign == 1 else t["s"][0]) - entry) < 1e-6 * max(1, diag)]
                variants = [("simple", entry, None)]
                for t in nb:
                    if t["kind"] == "cylinder" and t["r"][0] > r + tol:
                        variants.append(("counterbore", t["s"][0] if sign == 1 else t["s"][1], t))
                    elif t["kind"] == "cone":
                        r_far = t["r"][0] if sign == 1 else t["r"][1]
                        if r_far > r + tol:
                            variants.append(("countersink", t["s"][0] if sign == 1 else t["s"][1], t))
                for kind, s_e, t in variants:
                    far = s1 if sign == 1 else s0
                    depth = abs(far - s_e)
                    for through in (False, True):
                        rel_s = (s_pts - s_e) * sign
                        m = (r_pts <= r) & (rel_s >= -1e-9) & (through | (rel_s <= depth + 1e-9))
                        h = {"type": "hole", "kind": kind,
                             "position": tuple(float(v) for v in loc + a * (s_e - loc @ a)),
                             "direction": tuple(float(v) for v in a * sign), "diameter": 2 * r}
                        h.update({"through_all": True} if through else {"depth": depth})
                        if kind == "counterbore":
                            rc, dc = t["r"][0], abs(t["s"][1] - t["s"][0])
                            m |= (r_pts <= rc) & (rel_s >= -1e-9) & (rel_s <= dc + 1e-9)
                            h.update({"cb_diameter": 2 * rc, "cb_depth": dc})
                        elif kind == "countersink":
                            rc = t["r"][0] if sign == 1 else t["r"][1]
                            ang = 2 * math.degrees(t["semi"])
                            cone_r = rc - rel_s * math.tan(t["semi"])
                            m |= (rel_s >= -1e-9) & (r_pts <= cone_r)
                            h.update({"cs_diameter": 2 * rc, "cs_angle": ang})
                        out.append(Candidate("hole", m, [h], f"hole {kind} Ø{2 * r:.4g} "
                                             f"{'through' if through else f'depth {depth:.4g}'} dir {np.round(a * sign, 3).tolist()}",
                                             prefer=1e-6))
    return out
