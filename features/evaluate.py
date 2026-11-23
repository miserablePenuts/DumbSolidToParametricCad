"""Score the feature extractor against the known histories of synthetic parts.

    .venv\\Scripts\\python.exe -m features.evaluate --data data\\synth_v2 --split test --n 200 --workers 8

Per part, the extracted features are compared with the ground-truth JSON history:

* holes — a predicted hole matches a ground-truth hole with the same diameter whose axis line
  coincides (parallel, < 0.05 mm apart); precision / recall over all holes, and the share of
  matched holes with the right type (simple / counterbore / countersink). A final mirror in the
  history is applied to the ground-truth holes (the mirrored copies exist in the solid).
* fillet radii / chamfer sizes — the set of distinct values in the history vs the extracted set.
* faces explained, segmenter agreement (features whose faces the segmenter labels as the rule says).

The history is one way to build the part, so a few disagreements are modelling choices (a
circular pocket vs a hole, a rounded sketch corner vs a fillet) rather than errors.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from cadseq.workers import SupervisedPool

AXIS_TOL = 0.05


def _gt_holes(seq: dict) -> list[dict]:
    holes = [{"kind": f["kind"], "d": f["diameter"], "p": np.array(f["position"], float),
              "a": np.array(f["direction"], float) / np.linalg.norm(f["direction"])}
             for f in seq["features"] if f["type"] == "hole"]
    for m in (f for f in seq["features"] if f["type"] == "mirror"):
        o, n = np.array(m["plane"]["origin"], float), np.array(m["plane"]["normal"], float)
        n /= np.linalg.norm(n)
        holes += [{**h, "p": h["p"] - 2 * ((h["p"] - o) @ n) * n, "a": h["a"] - 2 * (h["a"] @ n) * n} for h in list(holes)]
    return holes


def _same_hole(g: dict, h: dict) -> bool:
    a = np.array(h["axis"], float)
    if abs(g["d"] - h["diameter"]) > 1e-3 or abs(abs(a @ g["a"]) - 1) > 1e-6:
        return False
    v = np.array(h["position"], float) - g["p"]
    return float(np.linalg.norm(v - (v @ a) * a)) < AXIS_TOL


def _value_sets(seq: dict, kind: str, key: str) -> set:
    return {round(f[key], 3) for f in seq["features"] if f["type"] == kind}


def run_one(task: dict) -> dict:
    from .crosscheck import annotate
    from .extract import clean, extract

    t0 = time.time()
    res, _, seg = extract(Path(task["step"]), ckpt=Path(task["ckpt"]) if task.get("ckpt") else None)
    if seg is not None:
        annotate(res, seg)
    res = clean(res)
    gt = json.loads(Path(task["json"]).read_text(encoding="utf-8"))
    gh = _gt_holes(gt)
    ph = [f for f in res["features"] if f["type"] == "hole"]
    match = {}
    for i, h in enumerate(ph):
        for j, g in enumerate(gh):
            if j not in match.values() and _same_hole(g, h):
                match[i] = j
                break
    kind_ok = sum(ph[i]["subtype"] == gh[j]["kind"] for i, j in match.items())
    fr = {round(f["radius"], 3) for f in res["features"] if f["type"] == "fillet" and f["radius"]}
    cr = {round(f["distance"], 3) for f in res["features"] if f["type"] == "chamfer" and f["distance"]}
    cc = res.get("cross_check", {})
    seg_feats = cc.get("features", {})
    return {"id": task["id"], "family": task["family"],
            "holes_gt": len(gh), "holes_pred": len(ph), "holes_matched": len(match), "holes_kind_ok": kind_ok,
            "fillet_gt": sorted(_value_sets(gt, "fillet", "radius")), "fillet_pred": sorted(fr),
            "chamfer_gt": sorted(_value_sets(gt, "chamfer", "distance")), "chamfer_pred": sorted(cr),
            "faces": res["summary"]["faces"], "faces_explained": res["summary"]["faces_explained"],
            "features": len(res["features"]), "patterns": len(res["patterns"]),
            "seg_agree": seg_feats.get("agree", 0), "seg_checked": sum(seg_feats.get(k, 0) for k in ("agree", "partial", "disagree")),
            "seg_conflicts": len(cc.get("conflicts", [])), "seconds": time.time() - t0}


def _set_scores(rows, gt_key, pred_key):
    tp = sum(len(set(r[gt_key]) & set(r[pred_key])) for r in rows)
    n_gt = sum(len(r[gt_key]) for r in rows)
    n_pr = sum(len(r[pred_key]) for r in rows)
    return {"recall": round(tp / max(1, n_gt), 3), "precision": round(tp / max(1, n_pr), 3), "values_gt": n_gt}


def summarize(rows: list[dict]) -> dict:
    s = {"parts": len(rows)}
    m = sum(r["holes_matched"] for r in rows)
    s["holes"] = {"gt": sum(r["holes_gt"] for r in rows), "pred": sum(r["holes_pred"] for r in rows),
                  "recall": round(m / max(1, sum(r["holes_gt"] for r in rows)), 3),
                  "precision": round(m / max(1, sum(r["holes_pred"] for r in rows)), 3),
                  "type_accuracy": round(sum(r["holes_kind_ok"] for r in rows) / max(1, m), 3)}
    s["fillet_radii"] = _set_scores(rows, "fillet_gt", "fillet_pred")
    s["chamfer_sizes"] = _set_scores(rows, "chamfer_gt", "chamfer_pred")
    s["faces_explained"] = round(sum(r["faces_explained"] for r in rows) / max(1, sum(r["faces"] for r in rows)), 4)
    s["parts_all_faces_explained"] = round(np.mean([r["faces_explained"] == r["faces"] for r in rows]), 3)
    s["seg_feature_agreement"] = round(sum(r["seg_agree"] for r in rows) / max(1, sum(r["seg_checked"] for r in rows)), 3)
    s["seg_conflicts_per_part"] = round(float(np.mean([r["seg_conflicts"] for r in rows])), 2)
    s["seconds_median"] = round(float(np.median([r["seconds"] for r in rows])), 1)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/synth_v2"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ckpt", type=Path, default=Path("runs/seg/last.pt"))
    ap.add_argument("--no-seg", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()

    ids = json.loads((args.data / "splits.json").read_text())[args.split][: args.n]
    family = {}
    with open(args.data / "index.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            family[r["id"]] = r["family"]
    tasks = [{"id": i, "family": family.get(i, "?"), "step": str(args.data / "step" / i[:-3] / f"{i}.step"),
              "json": str(args.data / "json" / i[:-3] / f"{i}.json"),
              "ckpt": None if args.no_seg else str(args.ckpt)} for i in ids]
    out = Path("runs/features") / f"eval_{args.split}{args.n}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, failed, t0 = [], [], time.time()
    pool = SupervisedPool(run_one, args.workers, args.timeout, tasks_per_worker=25)
    with open(out, "w", encoding="utf-8") as fh:
        for k, (i, status, result) in enumerate(pool.run(tasks), 1):
            if status == "ok":
                rows.append(result)
                fh.write(json.dumps(result) + "\n")
            else:
                failed.append({"id": tasks[i]["id"], "status": status, "error": str(result)[:200]})
                fh.write(json.dumps(failed[-1]) + "\n")
            if k % 25 == 0 or k == len(tasks):
                print(f"[{k}/{len(tasks)}] {time.time() - t0:.0f}s failed {len(failed)}", flush=True)
    pool.close()
    summary = {"overall": summarize(rows), "failed": failed}
    fams = defaultdict(list)
    for r in rows:
        fams[r["family"]].append(r)
    summary["by_family"] = {f: summarize(rs) for f, rs in sorted(fams.items())}
    out.with_name(out.stem + "_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=1))
    print("failed:", failed)


if __name__ == "__main__":
    main()
