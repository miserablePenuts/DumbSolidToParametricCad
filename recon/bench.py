"""Benchmark the geometry-first reconstruction on parts with known histories (synthetic test split)
and on real STEP files without histories.

    .venv\\Scripts\\python.exe -m recon.bench --data data/synth_v2 --split test --n 200 --workers 8
    .venv\\Scripts\\python.exe -m recon.bench --steps C:/parts/*.stp --workers 4

Per part: IoU vs the input, strict rebuild of the saved JSON, number of features (vs the ground
truth where known), unexplained target faces and runtime. Writes runs/recon/bench_<name>.jsonl and
a summary (overall and per family).
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from cadseq.workers import SupervisedPool


def run_one(task: dict) -> dict:
    from cadseq.build import BuildError, build
    from cadseq.metrics import mc_iou

    from .reconstruct import EXPLAINED, load_target, reconstruct

    t0 = time.time()
    target = load_target(Path(task["step"]))
    res = reconstruct(target, task.get("uniform", 16384), task.get("mirror", True))
    seq = res["history"].sequence()
    try:
        strict = build(seq)
        strict_ok = abs(strict.Volume() - res["body"].Volume()) <= 1e-6 * max(1.0, abs(res["body"].Volume()))
    except BuildError:
        strict_ok = False
    ops = defaultdict(int)
    for f in seq.features:
        if f.type != "sketch":
            ops[f.type] += 1
    acc = res["face_acc"]
    out = {**{k: v for k, v in task.items() if k not in ("uniform", "mirror")},
           "iou": mc_iou(res["body"], target), "strict": strict_ok, "features": res["history"].n_features(),
           "ops": dict(ops), "faces": len(acc), "unexplained": int((acc < EXPLAINED).sum()),
           "unexplained_kinds": sorted({res["faces"][i].kind for i in np.nonzero(acc < EXPLAINED)[0]}),
           "mirror": res["mirror"], "seconds": time.time() - t0}
    if task.get("json"):
        gt = json.loads(Path(task["json"]).read_text(encoding="utf-8"))
        gt_ops = defaultdict(int)
        for f in gt["features"]:
            if f["type"] != "sketch":
                gt_ops[f["type"]] += 1
        out["gt_ops"], out["gt_features"] = dict(gt_ops), sum(gt_ops.values())
    return out


def summarize(rows: list[dict], failed: int) -> dict:
    n = len(rows) + failed
    iou = np.array([r["iou"] for r in rows] + [0.0] * failed)
    s = {"parts": n, "failed": failed,
         "iou_mean": round(float(iou.mean()), 4), "iou_median": round(float(np.median(iou)), 4),
         "iou>0.99": round(float((iou > 0.99).mean()), 3), "iou>0.999": round(float((iou > 0.999).mean()), 3),
         "strict": round(sum(r["strict"] for r in rows) / max(1, n), 3),
         "all_faces_explained": round(sum(r["unexplained"] == 0 for r in rows) / max(1, n), 3),
         "faces_explained": round(1 - sum(r["unexplained"] for r in rows) / max(1, sum(r["faces"] for r in rows)), 4),
         "seconds_median": round(float(np.median([r["seconds"] for r in rows])), 1) if rows else None}
    gt = [r for r in rows if "gt_features" in r]
    if gt:
        s["features_vs_gt"] = round(float(np.median([r["features"] / max(1, r["gt_features"]) for r in gt])), 2)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, help="synthetic dataset (step/ + json/ + splits.json)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--steps", nargs="*", default=[], help="STEP files / globs without histories")
    ap.add_argument("--name", default=None, help="output name (default: split or 'real')")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()

    tasks, family = [], {}
    if args.data:
        ids = json.loads((args.data / "splits.json").read_text())[args.split][: args.n]
        with open(args.data / "index.jsonl", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                family[r["id"]] = r["family"]
        tasks += [{"id": i, "family": family.get(i, "?"), "step": str(args.data / "step" / i[:-3] / f"{i}.step"),
                   "json": str(args.data / "json" / i[:-3] / f"{i}.json")} for i in ids]
    for pat in args.steps:
        for p in sorted(glob.glob(pat)) or [pat]:
            tasks.append({"id": Path(p).stem, "family": "real", "step": p})
    for t in tasks:
        t["mirror"] = not args.no_mirror
    name = args.name or (args.split if args.data else "real")
    out = Path("runs/recon") / f"bench_{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, failed, t0 = [], [], time.time()
    pool = SupervisedPool(run_one, args.workers, args.timeout, tasks_per_worker=20)
    with open(out, "w", encoding="utf-8") as fh:
        for k, (i, status, result) in enumerate(pool.run(tasks), 1):
            if status == "ok":
                rows.append(result)
                fh.write(json.dumps(result) + "\n")
                fh.flush()
            else:
                failed.append({**tasks[i], "status": status, "error": result})
                fh.write(json.dumps(failed[-1]) + "\n")
            if k % 20 == 0 or k == len(tasks):
                ok = summarize(rows, len(failed))
                print(f"[{k}/{len(tasks)}] {time.time() - t0:.0f}s  IoU mean {ok['iou_mean']} median {ok['iou_median']} "
                      f">0.99 {ok['iou>0.99']}  strict {ok['strict']}  all faces explained {ok['all_faces_explained']}  "
                      f"failed {len(failed)}", flush=True)
    pool.close()

    summary = {"overall": summarize(rows, len(failed))}
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    fam_failed = defaultdict(int)
    for f in failed:
        fam_failed[f["family"]] += 1
    for fam in sorted(set(by_fam) | set(fam_failed)):
        summary[fam] = summarize(by_fam[fam], fam_failed[fam])
    kinds = defaultdict(int)
    for r in rows:
        for k in r["unexplained_kinds"]:
            kinds[k] += 1
    summary["parts_with_unexplained_kind"] = dict(kinds)
    summary["failures"] = [{"id": f["id"], "status": f["status"], "error": str(f["error"])[:200]} for f in failed[:20]]
    out.with_name(out.stem + "_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
