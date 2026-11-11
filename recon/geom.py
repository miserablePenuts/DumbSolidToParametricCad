"""Exact B-Rep queries: faces with their analytic surfaces, outward normals and samples, inside /
outside classification, planar sections and face loops as schema sketch curves."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cadquery as cq
import numpy as np
from matplotlib.path import Path as MplPath
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepGProp import BRepGProp
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
from OCP.BRepTopAdaptor import BRepTopAdaptor_FClass2d
from OCP.GeomAbs import (GeomAbs_Circle, GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Line, GeomAbs_Plane,
                         GeomAbs_Sphere, GeomAbs_Torus)
from OCP.gp import gp_Pnt, gp_Pnt2d
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_IN, TopAbs_ON, TopAbs_REVERSED, TopAbs_WIRE
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS

from cadseq.schema import Arc, Circle, Line

PAR_TOL = 1e-6  # |cos| tolerance for parallel directions


class Unsupported(ValueError):
    """Geometry that the sketch vocabulary (line / arc / circle) cannot express."""


def vec(p) -> np.ndarray:
    return np.array([p.X(), p.Y(), p.Z()], dtype=float)


def canonical_dir(d: np.ndarray) -> np.ndarray:
    """Unit vector with its first significant component positive (direction up to sign)."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    for c in d:
        if abs(c) > 1e-9:
            return d if c > 0 else -d
    return d


def frame(n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed sketch frame (x, y, n) for normal ``n``; x follows the repo convention
    (+Y for ±X normals, +X for ±Y / ±Z normals) and the least-aligned world axis otherwise."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    axes = np.eye(3)
    if abs(abs(n[0]) - 1) < 1e-12:
        x = axes[1]
    elif abs(abs(n[1]) - 1) < 1e-12 or abs(abs(n[2]) - 1) < 1e-12:
        x = axes[0]
    else:
        x = axes[int(np.argmin(np.abs(n)))]
    x = x - (x @ n) * n
    x /= np.linalg.norm(x)
    return x, np.cross(n, x), n


# ------------------------------------------------------------------ faces --

@dataclass
class FaceInfo:
    index: int
    face: object                 # TopoDS_Face
    kind: str                    # plane | cylinder | cone | torus | sphere | other
    area: float
    points: np.ndarray           # (k, 3) samples on the trimmed face
    normals: np.ndarray          # (k, 3) outward unit normals at the samples
    normal: np.ndarray | None = None   # plane: outward normal
    offset: float | None = None        # plane: normal · p
    axis: np.ndarray | None = None     # cylinder / cone / torus: canonical axis direction
    loc: np.ndarray | None = None      # point on the axis
    radius: float | None = None        # cylinder radius, torus major radius, cone reference radius
    minor: float | None = None         # torus minor radius
    semi_angle: float | None = None    # cone semi-angle (rad)
    concave: bool | None = None        # material outside the surface (hole-like)
    span: float | None = None          # cylinder / cone: angular span of the face (rad)
    s_range: tuple | None = None       # cylinder / cone: extent along ``axis`` (absolute)
    r_range: tuple | None = None       # cylinder / cone: radius at the two ends of ``s_range``


def face_list(shape) -> list:
    s = shape.wrapped if hasattr(shape, "wrapped") else shape
    out, exp = [], TopExp_Explorer(s, TopAbs_FACE)
    while exp.More():
        f = TopoDS.Face_s(exp.Current())
        if not any(f.IsSame(g) for g in out):
            out.append(f)
        exp.Next()
    return out


def _outward(face, adaptor, u, v):
    props = BRepLProp_SLProps(adaptor, u, v, 1, 1e-7)
    if not props.IsNormalDefined():
        return None
    n = vec(props.Normal())
    return -n if face.Orientation() == TopAbs_REVERSED else n


def _sample(face, adaptor, n: int):
    u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
    clf = BRepTopAdaptor_FClass2d(face, 1e-7)
    pts, nrm = [], []
    # cell centres (not the boundary) so offset samples stay off edges
    for u in u0 + (np.arange(n) + 0.5) / n * (u1 - u0):
        for v in v0 + (np.arange(n) + 0.5) / n * (v1 - v0):
            if clf.Perform(gp_Pnt2d(u, v)) != TopAbs_IN:
                continue
            nn = _outward(face, adaptor, u, v)
            if nn is None:
                continue
            pts.append(vec(adaptor.Value(u, v)))
            nrm.append(nn)
    return pts, nrm


def analyse_faces(shape, n: int = 12) -> list[FaceInfo]:
    infos = []
    for i, face in enumerate(face_list(shape)):
        a = BRepAdaptor_Surface(face, True)
        g = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, g)
        pts, nrm = _sample(face, a, n)
        if len(pts) < 4:
            pts, nrm = _sample(face, a, 3 * n)
        info = FaceInfo(i, face, "other", g.Mass(), np.array(pts).reshape(-1, 3), np.array(nrm).reshape(-1, 3))
        t = a.GetType()
        u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
        if t == GeomAbs_Plane:
            info.kind = "plane"
            pl = a.Plane()
            nn = vec(pl.Axis().Direction())
            if face.Orientation() == TopAbs_REVERSED:
                nn = -nn
            info.normal, info.offset = nn, float(nn @ vec(pl.Location()))
        elif t in (GeomAbs_Cylinder, GeomAbs_Cone):
            g_ = a.Cylinder() if t == GeomAbs_Cylinder else a.Cone()
            ax = vec(g_.Axis().Direction())
            loc = vec(g_.Location())
            info.kind = "cylinder" if t == GeomAbs_Cylinder else "cone"
            info.axis, info.loc = canonical_dir(ax), loc
            info.radius = g_.Radius() if t == GeomAbs_Cylinder else g_.RefRadius()
            if t == GeomAbs_Cone:
                info.semi_angle = abs(g_.SemiAngle())
            info.span = u1 - u0
            # axial extent: v is the distance along the (unnormalised-sign) axis for a cylinder;
            # for a cone v runs along the generatrix, so project the iso-curve end points.
            ends = [vec(a.Value(0.5 * (u0 + u1), v)) for v in (v0, v1)]
            s0 = float(loc @ info.axis)
            ends = sorted(((float((p - loc) @ info.axis) + s0, p) for p in ends), key=lambda t: t[0])
            info.s_range = (ends[0][0], ends[1][0])
            rad = lambda p: float(np.linalg.norm((p - loc) - ((p - loc) @ info.axis) * info.axis))  # noqa: E731
            info.r_range = (rad(ends[0][1]), rad(ends[1][1]))
        elif t == GeomAbs_Torus:
            tor = a.Torus()
            info.kind = "torus"
            info.axis, info.loc = canonical_dir(vec(tor.Axis().Direction())), vec(tor.Location())
            info.radius, info.minor = tor.MajorRadius(), tor.MinorRadius()
        elif t == GeomAbs_Sphere:
            info.kind = "sphere"
            info.loc, info.radius = vec(a.Sphere().Location()), a.Sphere().Radius()
        if info.axis is not None and len(info.points):
            d = info.points - info.loc
            radial = d - np.outer(d @ info.axis, info.axis)
            info.concave = bool(np.mean(np.einsum("ij,ij->i", info.normals, radial)) < 0)
        infos.append(info)
    return infos


def neighbours(shape) -> dict[int, set]:
    """Face index → indices of faces sharing an edge."""
    s = shape.wrapped if hasattr(shape, "wrapped") else shape
    faces = face_list(s)
    out = {i: set() for i in range(len(faces))}
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(s, TopAbs_EDGE, TopAbs_FACE, m)
    for k in range(1, m.Extent() + 1):
        idx = sorted({next(i for i, f in enumerate(faces) if f.IsSame(g)) for g in m.FindFromIndex(k)})
        for i in idx:
            out[i].update(j for j in idx if j != i)
    return out


# --------------------------------------------------------- classification --

def classify(shape, pts: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    s = shape.wrapped if hasattr(shape, "wrapped") else shape
    clf = BRepClass3d_SolidClassifier(s)
    out = np.empty(len(pts), dtype=bool)
    for i, (x, y, z) in enumerate(pts):
        clf.Perform(gp_Pnt(float(x), float(y), float(z)), tol)
        out[i] = clf.State() == TopAbs_IN
    return out


# ------------------------------------------------------ sections / loops --

def section_faces(shape: cq.Shape, n: np.ndarray, h: float) -> list[cq.Face]:
    """Faces of the cross-section of ``shape`` with the plane {p : n·p = h}."""
    bb = shape.BoundingBox()
    c = np.array(bb.center.toTuple())
    base = c + n * (h - c @ n)
    size = 3.0 * bb.DiagonalLength + 10.0
    plane = cq.Face.makePlane(size, size, cq.Vector(*base), cq.Vector(*n))
    try:
        return [f for f in shape.intersect(plane).Faces() if f.Area() > 1e-9]
    except Exception:
        return []


def _p2(p, x, y) -> tuple[float, float]:
    return (float(p @ x), float(p @ y))


def wire_curves(wire, face, x: np.ndarray, y: np.ndarray) -> list:
    """Ordered sketch curves (line / arc / circle) of a wire of a planar face, in the 2-D frame
    (x, y). Raises ``Unsupported`` for other curve types."""
    curves = []
    wexp = BRepTools_WireExplorer(TopoDS.Wire_s(wire), face)
    edges = []
    while wexp.More():
        edges.append(wexp.Current())
        wexp.Next()
    for e in edges:
        if BRep_Tool.Degenerated_s(e):
            continue
        c = BRepAdaptor_Curve(e)
        v0, v1 = TopExp.FirstVertex_s(e, True), TopExp.LastVertex_s(e, True)
        p0, p1 = vec(BRep_Tool.Pnt_s(v0)), vec(BRep_Tool.Pnt_s(v1))
        t = c.GetType()
        if t == GeomAbs_Line:
            if np.linalg.norm(p1 - p0) > 1e-9:
                curves.append(Line(type="line", start=_p2(p0, x, y), end=_p2(p1, x, y)))
        elif t == GeomAbs_Circle:
            circ = c.Circle()
            if v0.IsSame(v1) or np.linalg.norm(p1 - p0) < 1e-9:
                if len(edges) != 1:
                    raise Unsupported("closed circle edge inside a multi-edge loop")
                return [Circle(type="circle", center=_p2(vec(circ.Location()), x, y), radius=circ.Radius())]
            pm = vec(c.Value(0.5 * (c.FirstParameter() + c.LastParameter())))
            curves.append(Arc(type="arc", start=_p2(p0, x, y), mid=_p2(pm, x, y), end=_p2(p1, x, y)))
        else:
            raise Unsupported(f"edge curve type {t} in sketch loop")
    if len(curves) < 2 and not (curves and isinstance(curves[0], Circle)):
        raise Unsupported("degenerate loop")
    return curves


def face_loops(face, x: np.ndarray, y: np.ndarray) -> tuple[list, list[list]]:
    """(outer curves, [inner curves]) of a planar face."""
    f = face.wrapped if hasattr(face, "wrapped") else face
    f = TopoDS.Face_s(f)
    outer_w = BRepTools.OuterWire_s(f)
    outer, inner = None, []
    exp = TopExp_Explorer(f, TopAbs_WIRE)
    while exp.More():
        w = exp.Current()
        curves = wire_curves(w, f, x, y)
        if w.IsSame(outer_w):
            outer = curves
        else:
            inner.append(curves)
        exp.Next()
    if outer is None:
        raise Unsupported("face without outer wire")
    return outer, inner


# ------------------------------------------------- 2-D polygons of curves --

def _arc_points(a, m, b, sag: float) -> list:
    a, m, b = map(np.asarray, (a, m, b))
    d = 2 * (a[0] * (m[1] - b[1]) + m[0] * (b[1] - a[1]) + b[0] * (a[1] - m[1]))
    if abs(d) < 1e-12:
        return [tuple(a)]
    sa, sm, sb = a @ a, m @ m, b @ b
    c = np.array([sa * (m[1] - b[1]) + sm * (b[1] - a[1]) + sb * (a[1] - m[1]),
                  sa * (b[0] - m[0]) + sm * (a[0] - b[0]) + sb * (m[0] - a[0])]) / d
    r = np.linalg.norm(a - c)
    ang = lambda p: math.atan2(p[1] - c[1], p[0] - c[0])  # noqa: E731
    t0, tm, t1 = ang(a), ang(m), ang(b)
    sweep = (t1 - t0) % (2 * math.pi)
    if (tm - t0) % (2 * math.pi) > sweep:  # mid not on the ccw path → clockwise
        sweep -= 2 * math.pi
    step = 2 * math.acos(max(-1.0, 1 - sag / r)) if r > sag else math.pi / 8
    k = max(4, int(math.ceil(abs(sweep) / max(step, 1e-3))))
    return [tuple(c + r * np.array([math.cos(t0 + sweep * i / k), math.sin(t0 + sweep * i / k)]))
            for i in range(k)]


def loop_polygon(curves, sag: float = 0.002) -> np.ndarray:
    if isinstance(curves[0], Circle):
        c, r = np.array(curves[0].center), curves[0].radius
        k = max(16, int(math.ceil(2 * math.pi / (2 * math.acos(max(-1.0, 1 - sag / r)))))) if r > sag else 16
        t = np.arange(k) / k * 2 * math.pi
        return c + r * np.stack([np.cos(t), np.sin(t)], 1)
    pts = []
    for cv in curves:
        if isinstance(cv, Line):
            pts.append(cv.start)
        else:
            pts += _arc_points(cv.start, cv.mid, cv.end, sag)
    return np.array(pts, dtype=float)


def in_loop(curves, uv: np.ndarray) -> np.ndarray:
    poly = loop_polygon(curves)
    lo, hi = poly.min(0), poly.max(0)
    sel = np.all((uv >= lo) & (uv <= hi), axis=1)
    out = np.zeros(len(uv), dtype=bool)
    if sel.any():
        out[sel] = MplPath(poly).contains_points(uv[sel])
    return out
