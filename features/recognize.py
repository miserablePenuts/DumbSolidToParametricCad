"""Rule-based feature recognizers on the exact B-Rep.

Every recognizer reads its parameters from the analytic surfaces of the input (no estimation):

* ``recognize_holes``      coaxial concave cylinders / cones → simple, counterbore, countersink,
                           stepped bores; blind (flat floor or drill point) or through
* ``fillet_items``         cylinder / torus / free-form faces tangent to two neighbours; radius,
                           convex / concave, the sharp edge they replaced (``cadseq.blends``);
                           ``fillet_features`` groups connected faces of equal radius
* ``chamfer_items``        narrow planar / conical strips meeting two neighbours at an angle;
                           distances to the recovered sharp edge (``chamfer_features`` groups them)
* ``recognize_extrusions`` the pad / pocket / trim decomposition of ``recon`` (exact sketch
                           profiles, any direction) with classified profiles (rectangle, circle,
                           slot, …) and the faces each one explains
* ``find_patterns``        linear / grid / circular patterns of identical holes
* ``symmetry``             mirror planes of the part

Face indices are ``recon.geom.face_list`` order (TopExp explorer order, the order used by the
graph / segmenter pipeline as well).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from cadseq.blends import _edge_length, _normal, sharp_edges
from cadseq.schema import Arc, Circle, Line
from recon.geom import FaceInfo, canonical_dir, frame, loop_polygon

FULL_TURN = 2 * math.pi - 1e-3
PARALLEL = 1 - 1e-6      # |cos| above this: parallel directions


def r6(v):
    """JSON-friendly rounding (removes 24.000000000000004-style noise, keeps exact values)."""
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return [r6(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    return round(float(v), 6) + 0.0


# --------------------------------------------------------------- topology --

def adjacency(shape):
    """(face map, shared) — ``shared[i][j]`` = B-Rep edges shared by faces i and j."""
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape

    s = shape.wrapped if hasattr(shape, "wrapped") else shape
    fmap = TopTools_IndexedMapOfShape()
    exp = TopExp_Explorer(s, TopAbs_FACE)
    while exp.More():
        fmap.Add(exp.Current())
        exp.Next()
    e2f = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(s, TopAbs_EDGE, TopAbs_FACE, e2f)
    shared: dict = defaultdict(lambda: defaultdict(list))
    for k in range(1, e2f.Extent() + 1):
        e = TopoDS.Edge_s(e2f.FindKey(k))
        if BRep_Tool.Degenerated_s(e):
            continue
        idx = sorted({fmap.FindIndex(f) - 1 for f in e2f.FindFromIndex(k)} - {-1})
        for i in idx:
            for j in idx:
                if i != j:
                    shared[i][j].append(e)
    return fmap, shared


def _face(fmap, i):
    from OCP.TopoDS import TopoDS
    return TopoDS.Face_s(fmap.FindKey(i + 1))


# ------------------------------------------------------------------ holes --

def _same_axis(f: FaceInfo, g: FaceInfo, tol: float) -> bool:
    if abs(f.axis @ g.axis) < PARALLEL:
        return False
    dv = g.loc - f.loc
    return np.linalg.norm(dv - (dv @ f.axis) * f.axis) < tol


def recognize_holes(faces: list[FaceInfo], shared, tol: float) -> list[dict]:
    """Chains of touching full (360°) concave cylinder / cone segments on one axis."""
    conc = [f for f in faces if f.kind in ("cylinder", "cone") and f.concave and f.s_range]
    axes: list[list[FaceInfo]] = []
    for f in conc:
        for grp in axes:
            if _same_axis(grp[0], f, tol):
                grp.append(f)
                break
        else:
            axes.append([f])
    holes = []
    for grp in axes:
        a, loc = grp[0].axis, grp[0].loc
        segs: list[dict] = []
        for f in grp:  # merge faces of one split surface into a segment
            for s in segs:
                if (s["kind"] == f.kind and np.allclose(s["r"], f.r_range, atol=tol)
                        and np.allclose(s["s"], f.s_range, atol=tol)):
                    s["span"] += f.span
                    s["faces"].append(f.index)
                    break
            else:
                segs.append({"kind": f.kind, "r": tuple(f.r_range), "s": tuple(f.s_range), "span": f.span,
                             "faces": [f.index], "semi": f.semi_angle})
        segs = sorted([s for s in segs if s["span"] > FULL_TURN], key=lambda s: s["s"][0])
        chains: list[list[dict]] = []
        for s in segs:
            if chains and abs(chains[-1][-1]["s"][1] - s["s"][0]) < tol:
                chains[-1].append(s)
            else:
                chains.append([s])
        for ch in chains:
            h = _describe_hole(ch, a, loc, faces, shared, tol)
            if h:
                holes.append(h)
    return holes


def _trim_edge_cones(ch: list[dict], r: float, tol: float) -> tuple[list[dict], dict]:
    """Cones at a chain end that are chamfers on the hole's edges, not part of the hole:
    a cone on a countersink cone or on a counterbore (entry chamfer), and a cone narrowing towards a
    flat floor (bottom-edge chamfer; a drill point narrows to ~0 and is kept)."""
    dropped = {"lo": [], "hi": []}
    changed = True
    while changed and len(ch) > 1:
        changed = False
        for end, name in ((0, "lo"), (-1, "hi")):
            seg = ch[end]
            if len(ch) < 2 or seg["kind"] != "cone":
                continue
            inner = ch[1] if end == 0 else ch[-2]
            outer_r = seg["r"][0] if end == 0 else seg["r"][1]
            on_head = inner["kind"] == "cone" or (inner["kind"] == "cylinder" and inner["r"][0] > r + tol)
            bottom_chamfer = tol < outer_r < r - tol
            if on_head or bottom_chamfer:
                dropped[name].append(ch.pop(end))
                changed = True
    return ch, dropped


def _describe_hole(ch, a, loc, faces, shared, tol) -> dict | None:
    cyls = [s for s in ch if s["kind"] == "cylinder"]
    if not cyls:
        return None  # a lone cone ring is a chamfer, not a hole
    r = min(s["r"][0] for s in cyls)
    ch, dropped = _trim_edge_cones(list(ch), r, tol)
    chain_faces = {i for s in ch for i in s["faces"]}
    r_max = max(max(s["r"]) for s in ch)
    s_lo, s_hi = ch[0]["s"][0], ch[-1]["s"][1]

    def axial_planes(seg_faces):
        """Planar neighbours ⊥ axis lying within ``r_max`` of the axis: (face, s, max radius)."""
        out = {}
        for i in seg_faces:
            for j in shared.get(i, {}):
                g = faces[j]
                if j in chain_faces or g.kind != "plane" or not len(g.points) or abs(g.normal @ a) < PARALLEL:
                    continue
                d = g.points - loc
                out[j] = (float(g.points[0] @ a), float(np.linalg.norm(d - np.outer(d @ a, a), axis=1).max()))
        return out

    # counterbore / stepped-bore shoulders: annular planes between the chain ends
    steps = [j for j, (s, rr) in axial_planes(chain_faces).items() if s_lo + tol < s < s_hi - tol and rr <= r_max + tol]
    ends = {}
    for end, seg, s_end, r_end in (("lo", ch[0], s_lo, ch[0]["r"][0]), ("hi", ch[-1], s_hi, ch[-1]["r"][1])):
        drop = dropped[end]
        near = set(seg["faces"]) | {i for d in drop for i in d["faces"]}
        s_out = (drop[-1]["s"][0] if end == "lo" else drop[-1]["s"][1]) if drop else s_end
        lo_s, hi_s = sorted((s_end, s_out))
        floors = [j for j, (s, rr) in axial_planes(near).items()
                  if lo_s - 10 * tol <= s <= hi_s + 10 * tol and rr <= r_end + tol]
        drill = seg["kind"] == "cone" and r_end < tol
        ends[end] = {"seg": seg, "s": s_end, "s_floor": s_out, "r": r_end, "floors": sorted(set(floors)),
                     "drill": drill, "closed": bool(floors) or drill}
    if ends["lo"]["closed"] and ends["hi"]["closed"]:
        return None  # closed cavity
    open_ = [e for e in ("lo", "hi") if not ends[e]["closed"]]
    headed = [e for e in open_ if ends[e]["r"] > r + tol]
    entry = headed[0] if headed else ("lo" if open_ == ["lo"] else "hi")
    other = "hi" if entry == "lo" else "lo"
    sign = 1 if entry == "lo" else -1
    e, o = ends[entry], ends[other]
    h = {"type": "hole", "diameter": 2 * r, "through": not o["closed"],
         "axis": canonical_dir(a), "direction": a * sign,
         "position": loc + a * (e["s"] - loc @ a),
         "length": ch[-1]["s"][1] - ch[0]["s"][0]}
    radii = sorted({round(s["r"][0], 6) for s in cyls})
    head = e["seg"]
    if len(radii) > 2:
        h["subtype"] = "stepped"
    elif head["kind"] == "cylinder" and head["r"][0] > r + tol:
        h.update({"subtype": "counterbore", "cb_diameter": 2 * head["r"][0], "cb_depth": head["s"][1] - head["s"][0]})
    elif head["kind"] == "cone" and e["r"] > r + tol:
        h.update({"subtype": "countersink", "cs_diameter": 2 * e["r"], "cs_angle": 2 * math.degrees(head["semi"])})
    else:
        h["subtype"] = "simple"
    if o["closed"]:
        bottom_s = o["seg"]["s"][1] if (o["drill"] and other == "lo") else (o["seg"]["s"][0] if o["drill"] else o["s_floor"])
        h["depth"] = abs(bottom_s - e["s"])
        h["bottom"] = "drill_point" if o["drill"] else "flat"
        if o["drill"]:
            h["drill_point_angle"] = 2 * math.degrees(o["seg"]["semi"])
    elif o["r"] > r + tol:
        h["exit_feature"] = "countersink/chamfer" if o["seg"]["kind"] == "cone" else "counterbore"
    if dropped["lo"] or dropped["hi"]:
        h["edge_chamfers"] = len(dropped["lo"]) + len(dropped["hi"])  # reported as chamfer features
    h["segments"] = [{"surface": s["kind"], "diameter_start": 2 * s["r"][0], "diameter_end": 2 * s["r"][1],
                      "length": s["s"][1] - s["s"][0]} for s in ch]
    h["faces"] = sorted(chain_faces | set(e["floors"]) | set(o["floors"]) | set(steps))
    return h


# ----------------------------------------------------------------- blends --

def _blend_convexity(f: FaceInfo, r: float) -> str | None:
    """Convex if the rolling-ball centre p − r·n (n outward) lies on the blend's spine."""
    if not len(f.points) or r is None:
        return None
    c = f.points - r * f.normals
    d = c - f.loc
    along = np.outer(d @ f.axis, f.axis)
    radial = d - along
    if f.kind == "cylinder":
        dist = np.linalg.norm(radial, axis=1)
    else:  # torus: distance to the major circle
        dist = np.hypot(np.linalg.norm(radial, axis=1) - f.radius, np.linalg.norm(along, axis=1))
    return "convex" if float(np.median(dist)) < 0.5 * r else "concave"


def _chains(items: list[dict], shared, key) -> list[list[dict]]:
    """Group blend faces into connected chains of equal ``key`` (e.g. radius + convexity)."""
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_face = {it["face"]: k for k, it in enumerate(items)}
    for k, it in enumerate(items):
        for j in shared.get(it["face"], {}):
            m = by_face.get(j)
            if m is not None and key(items[m]) == key(it):
                parent[find(k)] = find(m)
    groups = defaultdict(list)
    for k, it in enumerate(items):
        groups[find(k)].append(it)
    return list(groups.values())


def fillet_items(shape, faces: list[FaceInfo], used: set, shared, tol: float) -> list[dict]:
    """One item per fillet face (grouped into features by ``fillet_features``)."""
    cands = {f.index: "fillet" for f in faces if f.index not in used and (
        f.kind == "torus" or f.kind == "other" or (f.kind == "cylinder" and (f.span or 0) < FULL_TURN))}
    if not cands:
        return []
    ves = {}
    for ve in sharp_edges(shape, cands):
        ves.setdefault(ve.face, ve)
    items = []
    for i, ve in ves.items():
        f = faces[i]
        r = f.minor if f.kind == "torus" else f.radius if f.kind == "cylinder" else None
        items.append({"face": i, "radius": r, "convexity": _blend_convexity(f, r) if r else None,
                      "surface": f.kind, "edge": {"curve": ve.curve, "point": ve.point}, "supports": ve.supports})
    # spherical corner patches between fillets of the same radius
    for f in faces:
        if f.kind == "sphere" and f.index not in used:
            nb_r = [it for it in items if it["face"] in shared.get(f.index, {})
                    and it["radius"] and abs(it["radius"] - f.radius) < tol]
            if len(nb_r) >= 2:
                items.append({"face": f.index, "radius": f.radius, "convexity": nb_r[0]["convexity"],
                              "surface": "sphere", "edge": None, "supports": ()})
    return items


def fillet_features(items: list[dict], shared) -> list[dict]:
    out = []
    for ch in _chains(items, shared, lambda it: (None if it["radius"] is None else round(it["radius"], 5),
                                                  it["convexity"])):
        out.append({"type": "fillet", "radius": ch[0]["radius"], "convexity": ch[0]["convexity"],
                    "variable_or_freeform": ch[0]["radius"] is None,
                    "edges": [it["edge"] for it in ch if it["edge"]],
                    "surfaces": sorted({it["surface"] for it in ch}),
                    "faces": sorted(it["face"] for it in ch),
                    "alternatives": sorted({a for it in ch for a in it.get("alternatives", [])})})
    return out


def _distance_to_curve(edges, curve) -> float | None:
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve

    ds = []
    for e in edges:
        c = BRepAdaptor_Curve(e)
        proj = GeomAPI_ProjectPointOnCurve(c.Value(0.5 * (c.FirstParameter() + c.LastParameter())), curve)
        if proj.NbPoints():
            ds.append(proj.LowerDistance())
    return float(np.mean(ds)) if ds else None


def chamfer_items(shape, faces: list[FaceInfo], used: set, shared, fmap, tol: float,
                  max_width_ratio: float = 0.5, max_width_diag: float = 0.06, max_asymmetry: float = 4.0) -> list[dict]:
    """A chamfer is a narrow planar / conical strip whose two long sides meet support faces at an
    angle (neither tangent nor perpendicular), e.g. 45° to both supports of a 90° edge. Width limits
    (≤ 0.5 × its length, ≤ 6 % of the part diagonal) keep bevelled profile sides out — geometrically
    identical to a chamfer, but modelled in the sketch."""
    diag = shape.BoundingBox().DiagonalLength
    cands = {}
    for f in faces:
        if f.index in used or f.kind not in ("plane", "cone"):
            continue
        face = _face(fmap, f.index)
        angled = 0
        for j, edges in shared.get(f.index, {}).items():
            e = max(edges, key=_edge_length)
            n1, n2 = _normal(face, e), _normal(_face(fmap, j), e)
            if n1 is not None and n2 is not None and 0.05 < float(n1 @ n2) < 0.999:
                angled += 1
        if angled >= 2:
            cands[f.index] = "chamfer"
    items = []
    for ve in sharp_edges(shape, cands) if cands else []:
        if any(it["face"] == ve.face for it in items):
            continue
        sides = [shared[ve.face].get(s, []) for s in ve.supports]
        ds = [_distance_to_curve(es, ve.geom) for es in sides]
        if any(d is None or d < tol for d in ds):
            continue
        length = float(np.mean([sum(_edge_length(e) for e in es) for es in sides]))
        if max(ds) > max_width_ratio * length or max(ds) > max_width_diag * diag or max(ds) / min(ds) > max_asymmetry:
            continue
        n1 = faces[ve.supports[0]].normal
        n2 = faces[ve.supports[1]].normal
        dihedral = None
        if n1 is not None and n2 is not None:
            dihedral = 180.0 - math.degrees(math.acos(float(np.clip(n1 @ n2, -1, 1))))
        items.append({"face": ve.face, "d1": ds[0], "d2": ds[1], "edge": {"curve": ve.curve, "point": ve.point},
                      "dihedral": dihedral, "surface": faces[ve.face].kind})
    return items


def chamfer_features(items: list[dict], shared) -> list[dict]:
    out = []
    for ch in _chains(items, shared, lambda it: (round(min(it["d1"], it["d2"]), 4), round(max(it["d1"], it["d2"]), 4))):
        it = ch[0]
        equal = abs(it["d1"] - it["d2"]) <= 1e-4 * max(1.0, it["d1"])
        c = {"type": "chamfer", "distance": it["d1"] if equal else None,
             "distances": None if equal else sorted([it["d1"], it["d2"]]),
             "edges": [x["edge"] for x in ch], "surfaces": sorted({x["surface"] for x in ch}),
             "faces": sorted(x["face"] for x in ch),
             "alternatives": sorted({a for x in ch for a in x.get("alternatives", [])})}
        if it["dihedral"] is not None:
            c["edge_angle"] = it["dihedral"]  # angle between the support faces (90 = square edge)
        out.append(c)
    return out


# ------------------------------------------------------------- extrusions --

_CURVE = {"line": Line, "arc": Arc, "circle": Circle}


def _arc_radius(c: dict) -> float:
    a, m, b = (np.array(c[k], float) for k in ("start", "mid", "end"))
    la, lb, lc = np.linalg.norm(m - b), np.linalg.norm(a - b), np.linalg.norm(a - m)
    area2 = abs((m - a)[0] * (b - a)[1] - (m - a)[1] * (b - a)[0])
    return la * lb * lc / (2 * area2) if area2 > 1e-12 else math.inf


def classify_profile(curves: list[dict], tol: float = 1e-6) -> dict:
    """Name a closed sketch loop: circle, rectangle, rounded_rectangle, slot, polygon or general."""
    types = [c["type"] for c in curves]
    if types == ["circle"]:
        c = curves[0]
        return {"shape": "circle", "diameter": 2 * c["radius"], "area": math.pi * c["radius"] ** 2,
                "center_uv": list(c["center"])}
    poly = loop_polygon([_CURVE[c["type"]].model_validate(c) for c in curves])
    area = 0.5 * abs(float(np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], -1))))
    lo, hi = poly.min(0), poly.max(0)
    out = {"shape": "general", "area": area, "size_uv": list(hi - lo), "center_uv": list(0.5 * (lo + hi)),
           "lines": types.count("line"), "arcs": types.count("arc")}
    n = len(curves)
    vec = [np.subtract(c["end"], c["start"]) for c in curves]
    length = [float(np.linalg.norm(v)) for v in vec]
    alternating = all(types[i] != types[(i + 1) % n] for i in range(n))
    radii = [_arc_radius(c) for c in curves if c["type"] == "arc"]
    eq_r = bool(radii) and max(radii) - min(radii) < tol * max(1.0, max(radii))
    if types == ["line"] * 4 and all(abs(vec[i] @ vec[(i + 1) % 4]) < tol * length[i] * length[(i + 1) % 4] + 1e-12
                                    for i in range(4)):
        out.update(shape="rectangle", width=max(length[0], length[1]), height=min(length[0], length[1]))
    elif n == 8 and alternating and eq_r:
        out.update(shape="rounded_rectangle", corner_radius=radii[0])
    elif n == 4 and alternating and eq_r:
        lv = [(v, l) for v, l, t in zip(vec, length, types) if t == "line"]
        if abs(lv[0][1] - lv[1][1]) < tol * max(1.0, lv[0][1]) and abs(abs(lv[0][0] @ lv[1][0]) - lv[0][1] * lv[1][1]) < 1e-6 * lv[0][1] ** 2:
            out.update(shape="slot", width=2 * radii[0], length=lv[0][1] + 2 * radii[0])
    elif set(types) == {"line"}:
        out.update(shape="polygon", sides=n, regular=max(length) - min(length) < tol * max(length))
    return out


def recognize_extrusions(target, used: set, holes: list[dict], n_uniform: int, tol: float, log: list):
    """Extrude / cut features from the geometry-first decomposition of ``recon`` (exact sketch
    profiles in any direction). Each input face not taken by a hole / blend is assigned to the
    latest feature whose tool boundary it lies on. Returns (features, faces, PointSet)."""
    from recon.reconstruct import prismatic_history

    stats: dict = {}
    body, hist, rfaces, ps, tools = prismatic_history(target, log, stats, n_uniform)
    feats, kept_tools = [], []
    for t in tools:
        kind = t.desc.split()[0]
        if kind == "hole":
            continue
        sk, op = t.features[0], t.features[1]
        d = np.array(sk["plane"]["normal"], float)
        x = np.array(sk["plane"]["x_dir"], float)
        y = np.cross(d, x)
        h = float(np.array(sk["plane"]["origin"]) @ d)
        prof = sk["profiles"][0]
        loop = prof["inner"][0] if kind == "trim" else prof["outer"]
        shape = classify_profile(loop)
        if kind == "pocket" and shape["shape"] == "circle":
            cu, cv = shape["center_uv"]
            centre = cu * x + cv * y + h * d
            dup = [hl for hl in holes if abs(abs(hl["axis"] @ d) - 1) < 1e-6
                   and np.linalg.norm(np.cross(centre - hl["position"], d)) < 10 * tol
                   and abs(hl["diameter"] - shape["diameter"]) < 10 * tol]
            if dup:
                continue  # the same cylinder is already reported as a hole
        f = {"type": "extrude" if kind == "pad" else "cut",
             "subtype": "outline_trim" if kind == "trim" else shape["shape"],
             "direction": d if kind == "pad" else -d,
             "sketch_plane": {"origin": d * h, "normal": d, "x_dir": x},
             "profile": shape, "inner_loops": len(prof["inner"]) if kind == "pad" else 0}
        if kind == "pad":
            f.update(start_level=h, end_level=h + op["length"], length=op["length"])
        elif op.get("through_all"):
            f.update(start_level=h, through=True)
        else:
            f.update(start_level=h, end_level=h - op["depth"], depth=op["depth"], through=False)
        feats.append(f)
        kept_tools.append(t)
    # face assignment: a pad face has material inside the tool on its inner side; a cut face has
    # the cut tool on its outer (air) side
    faces_of = [[] for _ in feats]
    for fi in rfaces:
        if fi.index in used or not len(fi.points):
            continue
        pin, pout = fi.points - ps.delta * fi.normals, fi.points + ps.delta * fi.normals
        for k in range(len(kept_tools) - 1, -1, -1):
            t = kept_tools[k]
            ci, co = t.contains(pin), t.contains(pout)
            frac = float(np.mean(ci & ~co)) if t.op == "pad" else float(np.mean(co & ~ci))
            if frac >= 0.6:
                faces_of[k].append(fi.index)
                break
    for f, fl in zip(feats, faces_of):
        d = f["sketch_plane"]["normal"]
        f["faces"] = sorted(fl)
        f["end_faces"] = sorted(i for i in fl if rfaces[i].kind == "plane" and abs(rfaces[i].normal @ d) >= PARALLEL)
    return feats, rfaces, ps


# --------------------------------------------------------------- patterns --

def _clusters(v: np.ndarray, tol: float) -> np.ndarray:
    s = np.sort(v)
    cuts = np.nonzero(np.diff(s) > tol)[0]
    return np.array([g.mean() for g in np.split(s, cuts + 1)])


def _equal_steps(v: np.ndarray, tol: float) -> float | None:
    d = np.diff(np.sort(v))
    return float(d.mean()) if len(d) and np.ptp(d) < tol else None


def _linear(P, tol):
    c = P.mean(0)
    _, S, Vt = np.linalg.svd(P - c)
    if len(P) > 1 and S[1] / math.sqrt(len(P)) < tol:
        t = (P - c) @ Vt[0]
        pitch = _equal_steps(t, tol)
        if pitch:
            first = P[int(np.argmin(t))]
            last = P[int(np.argmax(t))]
            return {"kind": "linear", "direction": (last - first) / np.linalg.norm(last - first), "pitch": pitch}
    return None


def _grid(P, a, tol):
    if len(P) < 4:
        return None
    dist = np.linalg.norm(P[:, None] - P[None], axis=-1) + np.eye(len(P)) * 1e18
    i, j = np.unravel_index(np.argmin(dist), dist.shape)
    e1 = (P[j] - P[i]) - ((P[j] - P[i]) @ a) * a
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    u, v = P @ e1, P @ e2
    cu, cv = _clusters(u, tol), _clusters(v, tol)
    if len(cu) * len(cv) != len(P) or len(cu) < 2 or len(cv) < 2:
        return None
    pu, pv = _equal_steps(cu, tol), _equal_steps(cv, tol)
    if pu is None or pv is None:
        return None
    return {"kind": "grid", "directions": [e1, e2], "counts": [len(cu), len(cv)], "pitches": [pu, pv]}


def _circular(P, a, tol):
    if len(P) < 3:
        return None
    x, y, _ = frame(a)
    uv = np.stack([P @ x, P @ y], 1)
    A = np.c_[2 * uv, np.ones(len(uv))]
    sol, *_ = np.linalg.lstsq(A, (uv ** 2).sum(1), rcond=None)
    c = sol[:2]
    r = math.sqrt(max(sol[2] + c @ c, 0.0))
    if r < tol or np.abs(np.linalg.norm(uv - c, axis=1) - r).max() > tol:
        return None
    ang_tol = 2 * tol / r  # angular equivalent of the position tolerance
    ang = np.sort(np.arctan2(uv[:, 1] - c[1], uv[:, 0] - c[0]))
    steps = np.diff(np.r_[ang, ang[0] + 2 * math.pi])
    full = np.ptp(steps) < ang_tol
    inner = np.diff(ang)
    if not full and np.ptp(inner) >= ang_tol:
        return None
    step = float(steps.mean() if full else inner.mean())
    return {"kind": "circular", "center": c[0] * x + c[1] * y, "axis": a,
            "radius": r, "pcd": 2 * r, "angle_step": math.degrees(step), "full_circle": bool(full)}


def find_patterns(holes: list[dict], tol: float) -> list[dict]:
    """Holes of the same size, type and axis direction; positions are compared in the plane ⊥ axis
    (entry faces may sit at different heights) within the position tolerance ``tol``. Depth /
    through may differ between members (another feature can break into one) — reported as
    ``uniform_depth``."""
    groups = defaultdict(list)
    for h in holes:
        key = (h["subtype"], round(h["diameter"], 4), round(h.get("cb_diameter") or 0, 4),
               round(h.get("cs_diameter") or 0, 4), tuple(np.round(canonical_dir(np.asarray(h["direction"], float)), 6)))
        groups[key].append(h)
    out = []
    for key, hs in groups.items():
        if len(hs) < 2:
            continue
        a = canonical_dir(np.asarray(hs[0]["direction"], float))
        P = np.array([h["position"] for h in hs], float)
        heights = P @ a
        P = P - np.outer(heights, a)
        pat = _linear(P, tol) or _grid(P, a, tol) or _circular(P, a, tol) or {"kind": "identical_set"}
        if pat["kind"] == "circular":
            pat["center"] = pat["center"] + float(heights.mean()) * a
        depths = {(h["through"], round(h.get("depth") or 0, 4)) for h in hs}
        out.append({"type": "hole_pattern", **pat, "count": len(hs), "holes": [h["id"] for h in hs],
                    "hole": {"subtype": key[0], "diameter": hs[0]["diameter"]},
                    "uniform_depth": len(depths) == 1, "coplanar_entries": bool(np.ptp(heights) < tol)})
    return out


def symmetry(target, rfaces, ps) -> list[dict]:
    from recon.reconstruct import symmetry_planes

    return [{"origin": c, "normal": d} for c, d in symmetry_planes(target, rfaces, ps)]
