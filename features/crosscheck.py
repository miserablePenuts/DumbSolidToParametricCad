"""Second opinion from the learned face segmenter (``s2pnet.segment``, ``runs/seg``).

The segmenter predicts one of 9 classes per face (extrude side / end, cut side / end, simple /
counterbore / countersink hole, fillet, chamfer). Each rule-based feature lists the classes the
segmenter would have to predict on its faces; the share of faces where it does is the feature's
``seg_agreement``. Faces that no rule explains get the segmenter's class as a hint, and faces where a
rule and a confident segmenter prediction contradict each other are listed as conflicts — both are
what a reviewer should look at first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

EXPECTED = {
    "hole:simple": {"hole_simple"},
    "hole:counterbore": {"hole_counterbore"},
    "hole:countersink": {"hole_countersink"},
    "hole:stepped": {"hole_counterbore", "hole_simple"},
    "fillet": {"fillet"},
    "chamfer": {"chamfer"},
    "extrude": {"extrude_side", "extrude_end"},
    "cut": {"cut_side", "cut_end"},
    # an outline trim is a modelling device: the designer may have padded the outline instead
    "cut:outline_trim": {"cut_side", "cut_end", "extrude_side", "extrude_end"},
}
# coarse family, for "partial" agreement (e.g. counterbore vs simple hole)
FAMILY = {"extrude_side": "extrude", "extrude_end": "extrude", "cut_side": "cut", "cut_end": "cut",
          "hole_simple": "hole", "hole_counterbore": "hole", "hole_countersink": "hole",
          "fillet": "fillet", "chamfer": "chamfer"}


def expected_classes(feature: dict) -> set:
    t, sub = feature["type"], feature.get("subtype")
    return EXPECTED.get(f"{t}:{sub}") or EXPECTED.get(t, set())


def segment_faces(step_path: Path, ckpt: Path, rule_faces) -> dict:
    """Per rule face: (segmenter class, probability). Faces are matched by geometry, so a different
    face order between the two readers cannot misalign them."""
    import torch  # noqa: F401  (fail early with a clear error if torch is missing)

    from cadseq.provenance import FACE_CLASSES
    from identify import identify

    _, raw, pred = identify(step_path, ckpt)
    nx, fa = raw["node_x"], raw["face_attr"]
    n_seg = len(nx)
    kinds = ["plane", "cylinder", "cone", "sphere", "torus", "other"]
    seg_kind = [kinds[int(a[:6].argmax())] for a in fa]
    seg_area = np.expm1(fa[:, 16].astype(float))
    identity = n_seg == len(rule_faces) and all(
        seg_kind[i] == f.kind and abs(seg_area[i] - f.area) <= 1e-3 * max(1.0, f.area) for i, f in enumerate(rule_faces))
    if identity:
        mapping = list(range(n_seg))
    else:  # nearest trimmed-face centroid of the same surface type
        cent = []
        for g in nx:
            m = g[..., 6] > 0.5
            cent.append((g[..., :3][m] if m.any() else g[..., :3].reshape(-1, 3)).mean(0))
        cent = np.array(cent)
        mapping = []
        for f in rule_faces:
            c = f.points.mean(0) if len(f.points) else np.zeros(3)
            d = np.linalg.norm(cent - c, axis=1) + np.array([0 if k == f.kind else 1e9 for k in seg_kind])
            mapping.append(int(np.argmin(d)))
    cls, prob = pred["cls"], pred["prob"]
    return {"matched_by": "order" if identity else "centroid",
            "faces": [(FACE_CLASSES[int(cls[j])], float(prob[j, cls[j]])) for j in mapping]}


def annotate(result: dict, seg: dict, confident: float = 0.9) -> None:
    """Add ``seg`` fields to every feature and face of ``result`` (in place)."""
    labels = seg["faces"]
    owner = {}
    for f in result["features"]:
        exp = expected_classes(f)
        fam = {FAMILY[c] for c in exp}
        got = [labels[i][0] for i in f.get("faces", [])]
        for i in f.get("faces", []):
            owner[i] = f
        if not got:
            f["seg"] = {"status": "no_faces"}
            continue
        agree = sum(c in exp for c in got) / len(got)
        family = sum(FAMILY[c] in fam for c in got) / len(got)
        status = "agree" if agree >= 2 / 3 else "partial" if family >= 2 / 3 else "disagree"
        counts: dict = {}
        for c in got:
            counts[c] = counts.get(c, 0) + 1
        f["seg"] = {"status": status, "agreement": agree, "classes": counts}
    conflicts = []
    for face in result["faces"]:
        c, p = labels[face["index"]]
        face["seg_class"], face["seg_prob"] = c, p
        f = owner.get(face["index"])
        if f is not None and p >= confident and FAMILY[c] not in {FAMILY[x] for x in expected_classes(f)}:
            conflicts.append({"face": face["index"], "rule_feature": f["id"], "seg_class": c, "seg_prob": p})
    result["cross_check"] = {
        "model": seg["model"], "matched_by": seg["matched_by"],
        "features": {s: sum(1 for f in result["features"] if f.get("seg", {}).get("status") == s)
                     for s in ("agree", "partial", "disagree", "no_faces")},
        "conflicts": conflicts,
        "unrecognised_hints": [{"face": u["face"], "seg_class": labels[u["face"]][0], "seg_prob": labels[u["face"]][1]}
                               for u in result["unrecognised_faces"]],
    }
