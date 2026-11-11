# solid2param — Dumb solid → parametric design history

Goal: given a "dumb" B-Rep solid (STEP, no history), predict **a** valid parametric
feature history (JSON) that rebuilds it. Any history that reproduces the geometry is
acceptable — it does not need to match the original designer's history.

```
STEP → UV-Net B-Rep graph → encoder → Transformer decoder → token sequence → JSON → rebuild (CadQuery / CATIA)
```

New to the underlying theory (neural networks, GNNs, Transformers, B-Rep, NURBS, UV space,
continuity, kernel internals)? See the study guide: [THEORY.md](THEORY.md).

## Status (v1 — complete)

| Step | Status | What |
|------|--------|------|
| 1 | ✅ | Operation vocabulary + JSON schema + reference rebuilder (`cadseq/`) |
| 2 | ✅ | Procedural generator → 20K (STEP, JSON, graph) triplets with validity checks (`datagen/`) |
| 4 | ✅ | Tokenizer (quantized, lossless JSON ↔ tokens) + dataset loader (`cadseq/tokenizer.py`, `prepare.py`, `dataset.py`) |
| 5 | ✅ | Encoder–decoder model, grammar-constrained decoding, training & CAD-level evaluation (`s2pnet/`) — test mean IoU 0.958 / median 0.978, 99.3 % rebuilt (with generate-and-verify) |

## Roadmap v2 — hybrid, geometry-informed architecture

v1 predicts everything — structure **and** every number — as tokens. Its measured weaknesses are
numeric imprecision (IoU capped at ~0.97, 0 % exact) and fillet/chamfer edge references 1–4 mm off
(the cause of 85 % of strict-rebuild failures). v2 keeps what tokens are good at and moves geometry
into equations:

```
                         ┌─ discrete structure tokens (op type, axis, loops) ── grammar-constrained
STEP → B-Rep graph → encoder ─┼─ numbers: coarse token + continuous regression offset
        (face/edge tokens)    ├─ pointer heads: feature → input faces, fillet/chamfer → input edges
                         └─ per-face feature labels (feature identification, auxiliary task)
                                        │
   training:  cross-entropy + regression + pointer loss
              + equation-residual loss (predicted feature surfaces evaluated at input UV-grid points)
   inference: decode → least-squares refinement against input faces → generate-and-verify rerank
              → rebuild (CadQuery / CATIA CAA R29)
```

| Decision | Choice | Reason |
|---|---|---|
| Replace tokens entirely with surface equations | ❌ | Loses the parametric feature tree; trimming/sewing infinite surfaces becomes the (hard) problem |
| Tokens for structure + continuous parameters | ✅ | Classification handles multi-modal choices; regression removes the 0.05 mm precision floor |
| Equation-residual losses on input UV samples | ✅ | The differentiable "geometry-informed" term — e.g. a hole's cylinder `‖(x−p)×d‖ − r` must vanish on the input cylinder face's samples; no CAD kernel in the loop |
| Pointer heads to input faces / edges | ✅ | Edge references valid by construction (fixes the persistent-naming failure); gives the equation loss its correspondences |
| Least-squares refinement at inference | ✅ | Snaps predicted parameters to the exact input faces — cheap, large precision gain |
| NURBS head for faces no feature explains | ✅ | Home for freeform / A-surface faces and for G1/G2 continuity losses |
| CAD-kernel rewards (build, IoU, χ/genus, edge-continuity match) | ✅ | Used for reranking and RL fine-tuning (GRPO), where topology/continuity checks belong — they are not differentiable |
| CATIA rebuild via CAA R29 | ✅ | PartDesign features from JSON; B-Rep cell access for edge selection; CGM NURBS for A-surfaces |

### Phases

| Phase | Status | What | Output |
|---|---|---|---|
| 0 | ⏳ | **Fix known v1 issues** (see below) | v1.1 baseline |
| 1 | ✅ | **Provenance labels**: face → creating feature (class, role, instance, mirror) by replaying each history with OCC operation history (`cadseq/provenance.py`, `cadseq/labels.py`) — added to the existing 20K dataset without regeneration; plus exact analytic face / edge attributes (surface type, axis, radius, convexity) | 925K labelled faces |
| 2 | ✅ | **Feature identification** (UV-Net / AAGNet-style face segmentation): `identify.py` colours each face of a STEP by feature class, groups faces into feature regions and reads parameters from the exact B-Rep — test: 99.8 % face accuracy, mIoU 0.995, instance F1 0.984, placement-invariant. (MFCAD++ benchmark still to do) | standalone tool + auxiliary task for phase 3 |
| 3 | ✅ | **Hybrid decoder**: copy head over exact input B-Rep values (pointer-generator), grammar value ranges, pose augmentation, aux segmentation, edge pointer to recovered sharp edges — test (verified): strict build 15.8 % → 42.4 %, IoU > 0.99 23 % → 60 %, median 0.978 → 0.994. Equation-residual losses deferred to phase 4 | v2 / v3 model |
| 4 | ⏳ | **Inference**: least-squares refinement, N-sample generate-and-verify reranking; then RL fine-tuning with CAD-kernel rewards | precision + build-rate gains |
| 5 | ⏳ | **A-surface solids**: NURBS head, G1/G2 continuity losses, freeform synthetic data | freeform reconstruction |
| 6 | ⏳ | **CATIA rebuild** (was step 3): JSON → CATIA V5 R29 feature tree via CAA (COM automation prototype first) | parametric CATIA parts |

Evaluation for every phase: the same 1,000-part test split (strict / lenient build rate, IoU, Chamfer
distance, plus topology match χ/genus and per-edge continuity-class match), ablating each component
against the v1 baseline.

### Known issues to fix first (phase 0)

| # | Issue | Evidence | Fix |
|---|---|---|---|
| 0.1 | **No placement canonicalization** — real STEP files fail | `examples/SimpleRectangle.stp` (centred on Y/Z) → IoU 0.21; all training parts start at z = 0 | Canonical pose at inference (axis-aligned bbox → canonical origin/orientation, map the predicted history back) + translation / axis-permutation augmentation in training |
| 0.2 | ✅ **Checkpoint selection on the wrong metric** | `best.pt` (lowest val loss, epoch 34) is clearly worse than `last.pt` (IoU > 0.95: 54 % vs 71 %) | `best.pt` now selected on the CAD score (mean IoU on 64 val parts, every `--gen-every` epochs) — applies from the next training run |
| 0.3 | **Overfitting** | val loss rises after epoch ~30; train 0.15 vs val 0.49 | More synthetic data (50–100K, ~45 min per 20K), augmentation from 0.1, tuned dropout / label smoothing |
| 0.4 | ✅ **Weak in-training signal** | `gen_exact` is always 0, so it says nothing | Added `gen_tok_err` (normalized token edit distance) and the periodic CAD check (`cad_built*`, `cad_iou_*`) |
| 0.5 | **Long, repetitive sequences** | holes ≈ 30 % of features (each pattern member is a separate feature) | Linear / circular pattern operations in schema, tokenizer, grammar and generator |
| 0.6 | **Dataset coverage gaps** | only axis-aligned planar host faces; no revolve / groove; no radial holes into shafts | Revolve + groove ops; angled sketch planes; radial holes |
| 0.7 | **Small faces invisible to the 10×10 grid** | 619 faces (thin chamfer/fillet strips) with an all-zero trim mask | Adaptive sampling inside the trimmed region (fallback to points from the face triangulation) |
| 0.8 | ✅ **Two Python environments** | datagen ran in the DGL venv only for historical reasons | `.venv` is now the single environment (datagen output bit-identical); DGL optional |
| 0.9 | ✅ **Stale data on disk** | `data/synth_v1`, `data/pilot` | Deleted |
| 0.10 | ✅ **Single greedy prediction** (quick win) | one decode per part | Generate-and-verify (8 samples, T = 0.5): test mean IoU 0.897 → **0.958**, built 94.7 % → **99.3 %**, IoU > 0.95 70.6 % → **83.2 %** — no retraining |
| 0.11 | ✅ (generator) **Degenerate counterbores** | ~1 % of counterbores have a head deeper than the wall, leaving a plain Ø(cb) hole — ambiguous labels / targets | Generator now requires wall thickness > head + 1 mm (takes effect at the next regeneration); labels relabel existing cases as plain holes |

### Related work — comparison and what we adopt

Survey of 20+ papers / repos (licenses checked via the GitHub API, 2026-11-02). **Bold** licenses
allow commercial reuse; others are research / non-commercial only — use their *ideas*, not code or data.

| Work | Input → output | Key result | License | Relevance to us |
|---|---|---|---|---|
| [Brep2Seq](https://github.com/zhangshuming0668/Brep2Seq) | B-Rep → primitive + 24 feature sequence (fillets by 2 points, 256-level quantization) | synthetic: 97.5 % valid, IoU 92.9; real (DeepCAD / Fusion): 87–88 % valid, IoU 70–78 | **MIT** | Closest to us — same point-based edge references (our main failure), 1M training parts; shows the expected synthetic → real gap |
| [UV-Net](https://github.com/AutodeskAILab/UV-Net) | B-Rep → class / per-face segmentation | MFCAD++ 99.0 % acc | **MIT** | Our encoder already follows it; its segmentation loop is the baseline for phase 2 |
| [AAGNet](https://github.com/whjdark/AAGNet) | B-Rep → semantic + **instance** + bottom-face segmentation | MFCAD++ 99.3 % acc / 98.6 IoU (best) | **MIT** | Instance segmentation = our feature → face correspondence; bottom-face head = sketch-plane face; feature-level F1 metric |
| [BRepNet](https://github.com/AutodeskAILab/BRepNet) | coedge convolution → segmentation | Fusion seg 90.2 % acc | CC BY-NC-SA | Coedge (next / prev / mate) topology idea for edge pointers — reimplement from the paper |
| [BRT](https://github.com/Qiang-Zou/BRT) | Bézier-patch tokens → Transformer | best on curved geometry | academic only | Ideas: 25–50 % face-token masking augmentation; Bézier encoding for A-surfaces |
| [Fusion 360 Gallery](https://github.com/AutodeskAILab/Fusion360GalleryDataset) | face-extrusion agent **points at target B-Rep faces** | synthetic-only training IoU 0.72 vs real 0.90 | data non-commercial | Pointer-to-input-entity design; face → feature labels; quantifies the synthetic-data gap |
| [JoinABLe](https://github.com/AutodeskAILab/JoinABLe), SolidGen | softmax / pointer over B-Rep faces & edges | — | **MIT** / no code | Working pattern for "select an input edge" pointer heads |
| [DeepCAD](https://github.com/rundiwu/DeepCAD) | sketch-extrude command sequence, 8-bit quantization | quantized CD 0.79 vs **continuous regression 2.14** | **MIT** | Evidence that pure regression is worse → keep tokens, add an offset + refinement; ACC_cmd / ACC_param metrics |
| [CAD-Recode](https://github.com/filaPro/cad-recode) | point cloud → CadQuery code (LLM), 1M procedural programs | own procedural data beats DeepCAD data (CD 0.54 vs 1.98) | data / weights NC | **Test-time sampling: 10 candidates, keep the best**; generator design (validity filter, scale to ~1M) |
| [cadrille](https://github.com/col14m/cadrille) | multimodal → CadQuery; SFT + **online RL** (reward 10·IoU − 10·invalid) | real scans: invalid < 0.2 %; online RL > DPO | code **Apache-2.0**, weights NC | RL recipe on **unlabeled** real STEP files (hard-example mining); IoU / CD / invalid-ratio evaluation |
| [ParSeNet](https://github.com/Hippogriff/parsenet-codebase), [Point2Cyl](https://github.com/mikacuy/point2cyl), [ComplexGen](https://github.com/guohaoxiang/ComplexGen) | primitive / extrusion / B-spline fitting | — | none / **MIT** / **MIT** | Residual distances for plane / cylinder / cone / sphere / torus / B-spline and differentiable least squares (= our equation losses + refinement; reimplement ParSeNet's math); Hungarian-matching losses; 20×20 B-spline control nets for the NURBS head |

No published work solves persistent naming for *learned* fillet references; pointing at entities of
the **final input solid** (supervised via OCC `BRepTools_History` in our generator) is our contribution.

**Adopted into the plan (ranked):**

| # | Adopt | Fixes | Phase |
|---|---|---|---|
| 1 | Pointer heads to input edges / faces | edge-reference failures (85 % of strict failures) | 1 + 3 |
| 2 | Equation residuals + least-squares refinement (ParSeNet / Point2Cyl math) | numeric precision | 3 + 4 |
| 3 | Test-time N-sample generate-and-verify (CAD-Recode) | build rate, IoU — cheap, can run on the v1 model now | 0 / 4 |
| 4 | Face semantic + instance + bottom-face segmentation (AAGNet), pre-trained on MFCAD++ | feature identification; prior for pointers | 2 |
| 5 | Pose canonicalization + rigid augmentation + face-token masking (DeepCAD / Brep2Seq / BRT) | real-part placement, overfitting | 0 |
| 6 | Scale procedural data to 100K–1M (CAD-Recode, Brep2Seq) | overfitting, coverage | 0 |
| 7 | Online RL with IoU / validity reward on unlabeled real STEP (cadrille) | synthetic → real gap | 4 |
| 8 | B-spline control-net head (ParSeNet), ComplexGen-style constrained refinement | A-surfaces | 5 |

Public data: **MFCAD++** (59,665 STEP files with per-face labels, **CC BY**) is the clean choice for
segmentation training and evaluation; MFInstSeg (60K+, instance labels, terms unverified); Fusion 360
Gallery, DeepCAD / ABC and CAD-Recode data are for research evaluation only (non-commercial or
Onshape terms).

## Layout

```
├── step_to_graph.py  # STEP → UV-Net face-adjacency graph (script version of the exploration notebook);
│                     #   the single implementation of UV-grid sampling used by the whole pipeline,
│                     #   plus optional exact analytic face / edge attributes
├── identify.py       # feature identification for a STEP: per-face classes + instances (JSON / PNG / STEP)
├── recon/            # geometry-first reconstruction (no learning): exact feature candidates from the
│                     #   STEP B-Rep + search + kernel verification (reconstruct.py, bench.py)
├── cadseq/
│   ├── schema.py     # pydantic models = the JSON format (v1.0)
│   ├── build.py      # JSON → CadQuery solid → STEP (Builder applies features incrementally)
│   ├── graph.py      # pipeline access to step_to_graph (numpy arrays) + to_dgl()
│   ├── tokenizer.py  # JSON ↔ token ids (quantized 0.05 mm, canonical)
│   ├── grammar.py    # incremental token grammar → allowed-next-token sets (constrained decoding)
│   ├── prepare.py    # pack a dataset: tokens.npz, graphs/*.npy, splits.json, vocab.json
│   ├── dataset.py    # torch Dataset + collate (pure torch, memory-mapped; labels/attributes if present)
│   ├── provenance.py # history-tracked replay: which feature created each face
│   ├── labels.py     # per-face labels + analytic attributes for a dataset (+ packing)
│   ├── augment.py    # rotation / translation / scale augmentation of graph features
│   ├── metrics.py    # Monte Carlo IoU + Chamfer distance between solids
│   └── workers.py    # crash/hang-tolerant supervised process pool for CAD-kernel work
├── s2pnet/           # model (encoder.py, decoder.py, model.py), train.py, evaluate.py, predict.py,
│                     #   verify.py (generate-and-verify: sample, rebuild, keep the best),
│                     #   segment.py (feature identification model)
├── datagen/          # synthetic dataset generator (see below)
├── examples/         # SimpleRectangle.stp, simple_rectangle / bracket / flange JSON histories
├── THEORY.md         # study guide for the underlying theory
├── requirements.txt / requirements-dgl.txt (optional, DGL only)
├── runs/             # (git-ignored) training runs: checkpoints, logs, eval results
└── data/             # (git-ignored) generated datasets (synth_v2/ = current)
```

`data/` and `runs/` are not in the repository (several GB; checkpoints exceed GitHub's 100 MB
file limit). Regenerate them with the commands below.

## STEP → graph (`step_to_graph.py`)

Script version of the `STEP_to_UVNet_Graph.ipynb` exploration notebook, with the same functions:
`load_step`, `tessellate_shape`, `render_shape`, `count_topo`, `check_brep`,
`sample_face_uv_grid`, `sample_edge_uv_grid`, `brep_to_graph_arrays`, `brep_to_dgl_graph`,
`graph_checks` and `plot_graph`. `cadseq.graph` re-exports the sampling functions, so dataset
generation, training data and prediction all use exactly this code.

```powershell
# topology, validity, graph sanity checks (+ optional plots / arrays / DGLGraph)
python step_to_graph.py examples\SimpleRectangle.stp --plots out\ --save-npz out\ --dgl
python step_to_graph.py path\to\folder_of_step_files
```

```python
from step_to_graph import load_step, brep_to_dgl_graph   # DGL optional; brep_to_graph_arrays needs only numpy
shape = load_step("examples/SimpleRectangle.stp")
graph, centroids = brep_to_dgl_graph(shape, N=10)        # ndata["x"] (6, 700), edata["x"] (24, 60)
```

## Usage

One virtual environment, `.venv`, runs everything: dataset generation, `cadseq.prepare`, training,
evaluation and prediction (`requirements.txt`). torch ≥ 2.7 with CUDA 12.8 is required for
Blackwell GPUs (sm_120). Dataset generation gives bit-identical samples in this environment
(checked against `synth_v2`).

```powershell
# one-off setup
py -3.11 -m venv .venv
$th = "--trusted-host","pypi.org","--trusted-host","files.pythonhosted.org","--trusted-host","download.pytorch.org"
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128 @th
.venv\Scripts\python.exe -m pip install -r requirements.txt @th
```

Run everything from the repository root:

```powershell
.venv\Scripts\python.exe -m cadseq.build examples\bracket.json --step out\bracket.step
```

DGL is **optional** and only used for `DGLGraph` output (`step_to_graph.py --dgl`,
`brep_to_dgl_graph`, `cadseq.graph.to_dgl`). DGL 2.2 needs torch 2.3 + numpy 1.x and has no newer
Windows builds, so it needs its own environment — see `requirements-dgl.txt`. The model itself is
pure PyTorch.

## JSON format (v1.0)

Units are millimetres and degrees. A file is `{"version": "1.0", "units": "mm", "features": [...]}`;
every feature has a unique `id` and a `type`.

### Sketch

```json
{"id": "S1", "type": "sketch",
 "plane": {"origin": [0,0,0], "normal": [0,0,1], "x_dir": [1,0,0]},
 "profiles": [{"outer": [<curves>], "inner": [[<curves>], ...]}]}
```

- 2-D coordinates are in the plane's local frame: local X = `x_dir`, local Y = `normal × x_dir`.
- Curves: `{"type":"line","start","end"}`, `{"type":"arc","start","mid","end"}` (3-point),
  `{"type":"circle","center","radius"}`. A circle must be alone in its loop; other loops must be closed and
  continuous (each curve's `end` = the next curve's `start`).
- `inner` loops are islands removed from the profile.

### Sketch-based features

| type | fields | notes |
|------|--------|-------|
| `pad` | `sketch`, `length`, `direction` (±1, default 1), `symmetric` | adds material; direction is relative to sketch normal |
| `pocket` | `sketch`, `depth` **or** `through_all`, `direction` (±1, default −1) | removes material |

### Hole

```json
{"id": "H1", "type": "hole", "kind": "simple|counterbore|countersink",
 "position": [x,y,z], "direction": [dx,dy,dz], "diameter": d,
 "depth": h | "through_all": true,
 "cb_diameter": D, "cb_depth": t,          // counterbore only
 "cs_diameter": D, "cs_angle": 90}         // countersink only
```

`position` lies on the entry face; `direction` points into the material. Flat-bottomed.

### Dress-up: fillet / chamfer

```json
{"id": "F1", "type": "fillet",  "radius": 5,   "edges": [{"point": [10,30,10], "curve": "line"}]}
{"id": "C1", "type": "chamfer", "distance": 1, "edges": [{"point": [50,0,0],   "curve": "line"}]}
```

**Edge references** are topology-independent: a point lying on the edge plus the edge's
curve type (`line|circle|ellipse|bspline|other`). A rebuilder picks the nearest edge of matching
type in the *current* body (tolerance 1e-3 mm for exact JSON; `QUANTIZED_EDGE_TOL` = 0.3 mm for
tokenized / predicted sequences, e.g. `build(seq, edge_tol=QUANTIZED_EDGE_TOL)`). This is what makes
the history replayable in a different kernel (CATIA) where edge IDs differ.

### Mirror

```json
{"id": "M1", "type": "mirror", "plane": {"origin": [0,0,0], "normal": [0,0,1]}}
```

Mirrors the whole current body and fuses it with the original (CATIA: *Mirror* of PartBody).

## Canonical ordering (generator rules)

Several histories can produce the same solid. To reduce that ambiguity for training, generated
sequences follow one fixed order:

1. Sketch + pad features (the first pad creates the body)
2. Sketch + pocket features
3. Holes
4. Fillets, then chamfers
5. Mirror (last — so dress-ups are mirrored for free)

Each sketch is immediately followed by the pad/pocket that consumes it.

## Synthetic dataset generator (`datagen/`)

```powershell
# full run (resumable: re-running skips finished ids)
.venv\Scripts\python.exe -m datagen.run --out data\synth_v2 --count 20000
# contact sheet of random samples
.venv\Scripts\python.exe -m datagen.preview data\synth_v2 --n 24 --out preview.png
```

Output per sample (`<shard>` = id without its last 3 digits, 1000 files per folder):

| Path | Content |
|------|---------|
| `step/<shard>/<id>.step` | dumb solid — model **input** |
| `json/<shard>/<id>.json` | design history — model **target** |
| `graph/<shard>/<id>.npz` | UV-Net arrays `node_x (F,10,10,7)`, `edge_index (2,E)`, `edge_x (E,10,6)`; load with `cadseq.graph.to_dgl` |
| `index.jsonl` / `summary.json` | per-sample metadata (family, op counts, faces, seed…) / dataset statistics |

### How a sample is generated

1. **Base family** (`families.py`), one of: prismatic — `plate` (rect / rounded / chamfered / polygon),
   `block`, `l_bracket` (optionally rounded top), `gusset_bracket`, `u_channel`, `t_block`;
   turned — `stepped_shaft`, `flange` (hub, optional spigot, bore), `hub_disc`.
2. **Decoration stages** (`features.py`) in canonical order, each with a random count:
   bosses → pockets (rect / rounded rect / circle / slot / polygon; blind or through) →
   hole groups (single / linear / grid / circular pattern; simple, counterbore or countersink with
   metric M3–M16 sizes; blind or through) → fillets → chamfers → optional mirror.
3. Features are only placed on **real axis-aligned planar faces** of the current body; a profile
   must fit inside the face with a margin, pocket/hole depths come from the measured wall thickness,
   and fillet/chamfer edges are chosen from actual edges (sizes ≤ 30% of the shortest selected edge).
4. Each feature group is applied **atomically** — kernel errors, invalid solids, no-op features,
   sliver geometry (any edge < 0.2 mm) or > 200 faces roll it back.
5. All geometric values are **snapped to the tokenizer grid (0.05 mm)**, so the stored history is
   exactly representable as tokens.

### Validation (a sample is kept only if all pass)

- The history is canonicalized through the tokenizer (`decode(encode(seq))`); the canonical JSON is
  what gets saved, and replaying it from scratch must reproduce the incremental body (volume, faces).
- STEP export → re-import gives one valid solid with identical faces/volume.
- UV-Net graph has no NaN/inf and the face graph is connected.

`datagen.run` supervises its own worker processes: a worker that crashes inside OpenCascade or
exceeds `--timeout` is killed and restarted, and the sample is retried with a fresh seed range.

Sequences are deterministic per `(--seed, id)`; use a different `--seed` for a disjoint split.

## Tokenizer (`cadseq/tokenizer.py`)

`encode(seq) -> list[int]`, `decode(ids) -> CADSequence`, `to_text(ids)` for debugging.
Vocabulary: 335 tokens = 3 special + 30 structural + 102 coarse + 200 fine number tokens.

**Numbers** are quantized to 0.05 mm (angles to 0.05°) in the range [-510, 510) and written as
two tokens: a 10 mm bucket `#k` and a 0.05 mm offset `.k` (value = (k_coarse·200 + k_fine − 10200) / 20).

**Grammar** (each feature starts with a feature token; no ids or sketch references are emitted):

```
SKETCH axis n_offset ( PROFILE LOOP curve+ (LOOP curve+)* )+       curve: LINE x y | ARC x y mx my | CIRCLE cx cy r
PAD    (EXT+|EXT-|EXT_SYM) n_length                                 # consumes the preceding SKETCH
POCKET (EXT+|EXT-) (THROUGH | n_depth)                              # consumes the preceding SKETCH
HOLE   (H_SIMPLE | H_CBORE D t | H_CSINK D angle) axis x y z d (THROUGH | n_depth)
FILLET r  (E_LINE|E_CIRCLE|...) x y z ...       CHAMFER d (E_LINE|E_CIRCLE|...) x y z ...
MIRROR (+X|+Y|+Z) n_offset
```

Example (`examples/bracket.json`, 184 tokens; `to_text` output):

```
SKETCH -Y 60 PROFILE LOOP LINE 0 0 LINE 100 0 LINE 100 10 LINE 10 10 ARC 10 70 5 75 LINE 0 70
PAD EXT+ 60
SKETCH +Z 10 PROFILE LOOP LINE 55 24 ARC 85 24 91 30 LINE 85 36 ARC 55 36 49 30
POCKET EXT- THROUGH
HOLE H_CBORE 11 4 +X 0 15 45 6.6 THROUGH
HOLE H_CSINK 10 90 -Z 25 15 10 5.5 THROUGH
FILLET 8 E_LINE 100 0 5 E_LINE 100 60 5
CHAMFER 1 E_LINE 50 0 0
```

**Canonicalization** (removes ambiguity the model would otherwise have to guess):
sketch planes must be axis-aligned with the origin on the axis and the canonical `x_dir`
(+Y for ±X normals, +X otherwise); lines/arcs store only their start point (the end is the next
curve's start); loops are re-oriented counter-clockwise and start at the lowest (x, y) vertex;
inner loops, profiles and fillet/chamfer edges are sorted; ids are regenerated (S1, P1, K1, H1…).
Inputs that cannot be represented raise `TokenizeError`; ungrammatical / invalid token sequences
raise `DecodeError` (use this to reject bad model outputs).

## Training data (`cadseq/prepare.py`, `cadseq/dataset.py`)

```powershell
# once per dataset: tokens.npz, graphs/ (packed, memory-mapped), splits.json (90/5/5 by family), vocab.json
.venv\Scripts\python.exe -m cadseq.prepare data\synth_v2
```

```python
from torch.utils.data import DataLoader
from cadseq.dataset import CADGraphSeqDataset, LengthBucketSampler, collate

ds = CADGraphSeqDataset("data/synth_v2", split="train")          # also "val" / "test"
dl = DataLoader(ds, batch_sampler=LengthBucketSampler(ds, 32), collate_fn=collate)
batch = next(iter(dl))
batch["node_x"]      # (ΣF,10,10,7)  face UV-grids
batch["edge_x"]      # (ΣE,10,6)     edge U-grids (directed)
batch["edge_index"]  # (2,ΣE)        global face ids;  batch["node_batch"] (ΣF,) sample per face
batch["tokens"]      # (B, L) int64, <bos> … <eos>, padded with PAD_ID = 0
```

Graphs are memory-mapped from `graphs/*.npy`, so `num_workers=0` is fast (no Windows worker
start-up). For a DGLGraph of one sample use `cadseq.graph.to_dgl` on its `.npz` file.

Point coordinates in the grids are scaled by 0.01 (1 unit = 100 mm) but **not centred**: the
target tokens are absolute mm in the world frame, so the encoder sees the same frame. (Real STEP
files will need a placement canonicalization step before inference — planned for step 6.)

## Model (`s2pnet/`)

```
face grids (ΣF,10,10,7) ─ SurfaceCNN (UV-Net 2-D conv) + geometric summary MLP ─┐
edge grids (ΣE,10,6)    ─ CurveCNN   (UV-Net 1-D conv) + endpoint/tangent MLP ──┤
                                                                                ▼
                       4 × edge-aware message passing (faces ↔ shared edges, edges updated too)
                                                                                ▼
                       [GLOBAL] + face tokens → 3-layer Transformer encoder  ──► memory (B, 1+Fmax, 384)
                                                                                ▼ cross-attention
<bos> … ─► 8-layer pre-norm Transformer decoder (d=384, 8 heads, tied embeddings, KV cache) ─► next token
```

- **Encoder** (`encoder.py`): the geometric summary per face (masked centroid, bbox, mean normal,
  trimmed fraction) gives the network direct access to absolute positions, which the tokens need.
- **Decoder** (`decoder.py`): teacher-forced cross-entropy for training; `generate()` does greedy
  or sampled decoding with a KV cache.
- **Grammar-constrained decoding** (`cadseq/grammar.py`): at every step only tokens allowed by the
  format are kept (feature order, number = coarse+fine pair, positive lengths/radii, ≥ 2 curves per
  loop, …). Every one of the 20K dataset sequences is accepted by the grammar.

## Training & evaluation

```powershell
# train (resumes automatically from runs\base\last.pt; logs runs\base\log.jsonl;
#        best.pt = highest CAD score on a val subset, see below)
.venv\Scripts\python.exe -m s2pnet.train --data data\synth_v2 --out runs\base --epochs 60
# CAD-level evaluation: generate → rebuild with CadQuery → compare with the input STEP
.venv\Scripts\python.exe -m s2pnet.evaluate --ckpt runs\base\last.pt --split test --n 1000
# generate-and-verify: greedy + 7 sampled candidates per part, keep the best rebuild
.venv\Scripts\python.exe -m s2pnet.evaluate --ckpt runs\base\last.pt --split test --n 1000 --samples 8 --temperature 0.8
```

Defaults: AdamW (lr 3e-4, wd 0.05, β = 0.9/0.98), 1000 warm-up steps, cosine decay, bf16 autocast,
batch 32 with length bucketing, grad-clip 1.0. ~37.6M parameters, ~24K tokens/s, ~3.5 min/epoch on
the RTX PRO 2000 (8 GB; peak ≈ 7.8 GB — use `--bs 24` if you hit out-of-memory).

During training (`log.jsonl`): `val_loss`, `val_tok_acc` (teacher forcing) every epoch, and every
`--gen-every` epochs:
- greedy generation on `--gen-samples` val parts: `gen_valid` (decodes), `gen_feature_types_match`,
  `gen_tok_err` (token edit distance / ground-truth length — moves as the model improves, unlike
  `gen_exact`, which stays 0);
- a CAD check on the first `--cad-samples` of them (default 64, in `--cad-workers` processes, ~40 s):
  `cad_built_strict`, `cad_built`, `cad_iou_mean`, `cad_iou_median`.

`best.pt` is the checkpoint with the highest `cad_iou_mean` (0 for parts that do not rebuild).
Validation loss is a poor proxy — in `runs/base` it rose after epoch ~30 while generation kept
improving, and the lowest-val-loss checkpoint was clearly worse (see Results). `--cad-samples 0`
restores selection by lowest val loss.

`evaluate.py` metrics (CAD kernel in parallel worker processes):

| metric | meaning |
|--------|---------|
| `exact` | chosen tokens identical to the ground truth |
| `valid` | at least one candidate decodes into a valid `CADSequence` |
| `built_strict` | the chosen history rebuilds exactly as predicted (edge refs within 0.3 mm) |
| `built` | rebuilds with `build_lenient` (see below) |
| `iou_*` | Monte Carlo volume IoU of the rebuild vs the input STEP (8192 points, ≈ 0.5 % std. error; 0 if not built) |
| `iou_strict_mean` | IoU counting only strict rebuilds |
| `chamfer_median_built` | surface Chamfer distance / bbox diagonal; ≈ 0.009 is the sampling floor for identical shapes |
| `mean_skipped_features` | features dropped by the lenient rebuild |
| `chosen_greedy` | share of parts where the greedy candidate was the best (generate-and-verify only) |

**Generate-and-verify** (`s2pnet/verify.py`, `--samples N` in `evaluate.py` / `predict.py`): the
greedy prediction plus N−1 temperature samples are rebuilt and the one with the highest IoU against
the **input** solid is kept — legitimate at inference, since the input is all it uses. Selection uses
its own 2048 random points (seed 1) so the reported IoU (seed 0) is not biased by the selection.
Temperature 0.5 works best (val, 200 parts, N = 8: mean IoU 0.958 at T = 0.5 vs 0.952 at T = 0.8,
which also took 5× longer to rebuild).

CAD work runs in a **supervised worker pool** (`cadseq/workers.py`, used by `evaluate.py` and the
training-time CAD check). With `multiprocessing.Pool`, an OpenCASCADE access violation silently
loses the task and the caller waits out the full timeout — this stalled a 1,000-part evaluation for
over an hour. The supervised pool watches every worker, reports a crash immediately, kills and
replaces hung workers at `--timeout` seconds per candidate, and counts both in `worker_failures`.

**Lenient rebuild** (`cadseq.build.build_lenient`, used when the strict rebuild fails): predicted
fillet/chamfer edge points are typically 1–4 mm off the true edge, which fails the strict 0.3 mm
match and — because one failed feature kills the whole part — made most predictions unbuildable.
The lenient builder snaps each edge ref to the nearest edge of the same curve type within 3 % of the
body's bbox diagonal (dropping duplicates/unresolvable refs) and skips any feature that still fails.
It also returns the **repaired history** (snapped edge refs written back, skipped features removed),
which `predict.py` saves — so the output JSON always strictly rebuilds the output STEP (verified on
44/44 lenient val rebuilds).

IoU is computed by point classification (`cadseq/metrics.py`) rather than OCC booleans, which fail
on near-coincident solids (prediction vs STEP round-trip) and returned IoU = 0 for exact matches.

Sanity check that was run: the model memorises 64 training parts to 100 % exact generation and
IoU 1.0 (`--limit 64 --val-split train --dropout 0`).

## Results — `runs/base` (60 epochs, synth_v2)

Held-out **test split, 1000 parts**, grammar-constrained decoding, lenient rebuild:

| checkpoint / inference | built (strict / lenient) | IoU mean | IoU median | IoU > 0.9 | > 0.95 | > 0.99 | Chamfer median |
|---|---|---|---|---|---|---|---|
| `last.pt` + **generate-and-verify** (8 samples, T = 0.5) | 15.8 % / **99.3 %** | **0.958** | **0.978** | **94.4 %** | **83.2 %** | **23.1 %** | 0.0091 |
| `last.pt` (epoch 59), greedy | 15.1 % / 94.7 % | 0.897 | 0.970 | 84.6 % | 70.6 % | 13.9 % | 0.0093 |
| `best.pt` (lowest val loss, epoch 34), greedy | 14.3 % / 93.8 % | 0.873 | 0.954 | 81.5 % | 54.4 % | 4.3 % | 0.0094 |

100 % of predictions are grammatical and decode; 0 % are token-for-token identical (sequences
are ~260 tokens of 0.05 mm numbers). Use **`last.pt`**: validation *loss* rises after epoch ~30
(overfitting / over-confidence; train loss 0.15 vs val 0.49) but generation quality keeps improving.

**Generate-and-verify** (phase-0 quick win, no retraining) lifts every family; the greedy candidate
is the best one in only 17 % of parts. Cost: 8× generation (11 min per 1000 parts) and ~18 min of
CAD rebuilds on 16 workers.

| family | median IoU greedy → verified | IoU > 0.95 | IoU > 0.99 |
|---|---|---|---|
| flange | 0.984 → 0.990 | 84 % → 94 % | 30 % → 49 % |
| hub_disc | 0.984 → 0.990 | 88 % → 95 % | 35 % → 50 % |
| l_bracket | 0.974 → 0.979 | 81 % → 88 % | 11 % → 17 % |
| t_block | 0.972 → 0.980 | 76 % → 84 % | 15 % → 19 % |
| gusset_bracket | 0.973 → 0.975 | 73 % → 83 % | 5 % → 13 % |
| plate | 0.959 → 0.974 | 61 % → 77 % | 7 % → 10 % |
| block | 0.961 → 0.970 | 63 % → 78 % | 1 % → 5 % |
| stepped_shaft | 0.953 → 0.969 | 53 % → 72 % | 5 % → 17 % |
| u_channel | 0.951 → 0.969 | 56 % → 75 % | 6 % → 11 % |

Remaining errors: strict rebuilds still fail on fillet/chamfer **edge references** (predicted points
1–4 mm off; sampling does not fix this — strict build stays ~15 %), and IoU is capped by slightly
wrong **dimensions/positions** of pockets and holes. Both are what the v2 pointer heads, equation
losses and least-squares refinement target. With generate-and-verify only 0.7 % of parts produce no
valid body.

Out-of-distribution check — `examples/SimpleRectangle.stp` (20 × 122.1 × 88.8 mm,
centred on Y/Z): the model predicts a ~40 × 122.5 × 90 box built up from z = 0 (IoU 0.21). Sizes are
roughly recognised but the **placement convention differs** from the synthetic data, which always
starts at z = 0 with sketch planes through the origin → placement canonicalization is needed (issue 0.1).

```powershell
.venv\Scripts\python.exe -m s2pnet.predict --ckpt runs\base\last.pt --step part.stp --out out\part --samples 8 --temperature 0.5
```

Training notes: the RTX PRO 2000 laptop GPU throttles thermally (~89 °C, 915 of 3090 MHz) which halves
throughput (≈ 48K → 24K tokens/s, 110 → 225 s/epoch). `train.py` blocks Windows idle sleep / Modern
Standby (which once paused a run for 75 min); closing the lid still suspends it.

## Face provenance labels (phase 1, `cadseq/provenance.py`, `cadseq/labels.py`)

Every face of every dataset solid is labelled with the feature that created it — without regenerating
the dataset. `TrackedBuilder` replays a history with the **same OpenCascade operations and settings**
as `cadseq.build.Builder` (identical solid, verified), but keeps each operation's history
(`BRepAlgoAPI_*`, `BRepFilletAPI_*`, `ShapeUpgrade_UnifySameDomain`, `BRepBuilderAPI_Transform`):
faces that survive (possibly split or trimmed) keep their label, faces of a pad / pocket / hole tool get
the new feature's label, faces created by a fillet / chamfer get theirs, mirror images inherit their
source label with `mirrored = 1`, and merged faces take the label of the largest source face.

```powershell
.venv\Scripts\python.exe -m cadseq.labels data\synth_v2 --workers 16   # ~23 min for 20K parts
```

| field (`graphs/*.npy`) | meaning |
|---|---|
| `face_cls` | `extrude_side`, `extrude_end`, `cut_side`, `cut_end`, `hole_simple`, `hole_counterbore`, `hole_countersink`, `fillet`, `chamfer` (Fusion 360 Gallery-style taxonomy + hole kinds) |
| `face_role` | finer role: side, end, wall, bottom, cb_wall, cb_floor, cs_cone, blend |
| `face_instance` / `face_feature` / `face_mirror` | feature instance in the part / index of the creating feature in the history (pointer target for phase 3) / mirror image flag |
| `face_attr` (17) | exact surface: type (plane, cylinder, cone, sphere, torus, other), axis or outward normal, location, radii, cone angle, convexity (+1 boss / −1 hole), log area |
| `edge_attr` (9) | curve type (line, circle, ellipse, other), convex / concave / smooth, length, radius |

STEP export / import can reorder faces; labels are mapped onto the STEP solid's faces geometrically
(sample points of each stored face must lie on the replayed face; 50 of 925K faces had moved).
Counterbores whose head is deeper than the wall (no hole wall left) are labelled as plain holes, and
the generator now rejects them (issue 0.11 below). Analytic attributes are an optional output of
`step_to_graph.brep_to_graph_arrays(..., with_attributes=True)`, so any STEP file gets them at inference.

## Feature identification (phase 2, `s2pnet/segment.py`, `identify.py`)

Per-face classification into the 9 classes above plus feature-instance grouping, as in UV-Net /
AAGNet. The model is the B-Rep encoder (UV-grid CNNs **+ exact analytic attributes**, edge-aware message
passing, face Transformer; 7.9M parameters) with a class head and an instance-embedding head
(pairwise same-feature affinity, balanced BCE). Instances are evaluated as **connected feature
regions**: one fillet feature may round several separate edges, and how a history splits equal-radius
fillets into features is arbitrary — `identify.py` regroups regions of equal radius into one feature.
Training applies random rotation / translation / scale (exactly consistent with an OCC transform of
the solid, verified to 3e-5), so identification does not depend on the STEP's placement.

```powershell
.venv\Scripts\python.exe -m s2pnet.segment train --data data\synth_v2 --out runs\seg      # ~40 min
.venv\Scripts\python.exe -m s2pnet.segment eval  --ckpt runs\seg\last.pt --split test
.venv\Scripts\python.exe identify.py part.stp --ckpt runs\seg\last.pt --out out\part
#   → out\part_features.json  (per-face class / region, feature list with parameters)
#     out\part_features.png   (VTK render, faces coloured by class, visible regions numbered)
#     out\part_features.step  (input solid with per-face colours — open in CATIA)
```

Feature parameters are read from the exact B-Rep, not estimated: hole diameter / axis / depth /
through, counterbore diameter, countersink angle, fillet radius and convex / concave edge, extrude /
pocket direction and height. Example (`examples/bracket.json` rebuilt as a STEP) — every feature of the
history is recovered: 2 counterbores Ø6.6 / Ø11 through, 2 countersinks Ø5.5 / 90° through, the R8
fillet on two edges (one feature), the concave R5 fillet, the slot pocket and the chamfer.

![bracket](docs/identify_bracket.png)

Test split (1000 parts), `runs/seg/last.pt`:

| placement | face accuracy | mIoU (9 classes) | instance precision / recall / F1 |
|---|---|---|---|
| random rotation + translation + scale | **99.81 %** | **0.995** | 0.995 / 0.973 / **0.984** |
| as generated | 99.83 % | 0.995 | 0.996 / 0.974 / 0.985 |
| ablation: UV-grids only (no analytic attributes), random placement | 99.55 % | 0.988 | 0.989 / 0.960 / 0.974 |

The exact analytic attributes cut face errors 2.4× (0.45 % → 0.19 %); the largest gains are on countersinks
(IoU 0.962 → 0.985), plain holes and pocket floors — direct evidence for feeding exact B-Rep geometry to the
phase-3 decoder as well.

Per-class IoU: extrude side / end 0.9995 / 0.9987, cut side / end 0.9978 / 1.0, holes simple /
counterbore / countersink 0.988 / 0.989 / 0.985, fillet 0.9999, chamfer 0.995. Instance recall per type:
extrude 0.97, cut 0.99, holes 0.98–0.99, fillet 0.97, chamfer 0.94. For reference, published MFCAD++
results are 99.0 % (UV-Net) and 99.3 % (AAGNet) face accuracy — a different (machining-feature) dataset,
so not directly comparable; these numbers are in-distribution on our synthetic parts.

## Hybrid copy decoder (phase 3, `runs/v2`)

v1 generates every number digit by digit (0.05 mm grid), so its IoU is capped by small dimension and
position errors. v2 keeps the token decoder for structure but **copies numbers from the input B-Rep**:

- **Candidates** (`cadseq/candidates.py`): every exact value of the input solid that a history could
  use — vertex / circle-centre / arc / edge-midpoint coordinates (±), plane offsets (±), bounding box,
  radii, diameters, cone angles, and per-axis differences of positions — deduplicated to 1e-4 mm and
  tagged with their origin (18 multi-hot tags). Median 446 per part; **98.9 % of all ground-truth numbers
  occur exactly** in their part's candidate set.
- **Copy head** (`s2pnet/model.py`, pointer-generator): candidate embeddings (Fourier value features +
  tags, cross-attending to the face memory) are scored against the decoder state. A gate mixes
  "copy candidate k" with "generate the digits"; the loss marginalises over every candidate holding the
  target value: `log p(num) = logaddexp(log P(start)·σ(g)·Σ_match π_k, σ(−g)·p_gen)`.
- **Grammar value ranges** (`cadseq/grammar.py`): each number slot has a quantized range (e.g. a
  counterbore hole diameter must be below the head diameter); copying only considers candidates in range.
- **Pose augmentation** (`cadseq/pose.py`, `--pose-aug 0.75`): one of the 48 signed axis permutations +
  a grid-aligned translation up to ±100 mm, applied consistently to the history (re-expressed in
  canonical sketch frames and re-tokenized), the graph features and the candidates. Verified: rebuilt
  solid IoU 1.0 against the OCC-transformed reference, and the inverse transform round-trips exactly.
- Inputs also include the exact analytic face / edge attributes (`--attrs`) and an auxiliary face-class
  loss (`--aux-seg 0.2`). 39.7M parameters.

```powershell
.venv\Scripts\python.exe -m cadseq.candidates data\synth_v2 --workers 16     # once: pack candidate primitives
.venv\Scripts\python.exe -m s2pnet.train --data data\synth_v2 --out runs\v2 --epochs 60 --bs 24 --attrs --copy --aux-seg 0.2 --pose-aug 0.75 --loader-workers 4
.venv\Scripts\python.exe -m s2pnet.evaluate --ckpt runs\v2\best.pt --split test --n 1000 --samples 8 --temperature 0.5 [--pose]
.venv\Scripts\python.exe -m s2pnet.predict --ckpt runs\v2\best.pt --step part.stp --out out\part --samples 8 --temperature 0.5
```

`predict.py` builds the same inputs from the STEP and, for copy models, moves the bounding-box centre
(rounded to whole mm, so every input value stays on the grid) to the origin and maps the predicted
history back (`--no-center` to disable). Arbitrarily rotated parts still need axis alignment.

Training: batch 24 (peak 6.6 GB; validation uses the training batch size — at 2× it spilled into
shared memory), ~280 s/epoch on the throttled laptop GPU, 60 epochs ≈ 4.7 h.

### Results — test split, 1000 parts

| model / inference | exact | built (strict / lenient) | IoU mean | IoU median | > 0.9 | > 0.95 | > 0.99 |
|---|---|---|---|---|---|---|---|
| v1 greedy | 0 % | 15.1 % / 94.7 % | 0.897 | 0.970 | 84.6 % | 70.6 % | 13.9 % |
| **v2 greedy** | 2.8 % | 15.4 % / 95.6 % | 0.909 | 0.987 | 85.5 % | 74.8 % | **44.3 %** |
| v1 + generate-and-verify (8, T = 0.5) | 0 % | 15.8 % / 99.3 % | 0.958 | 0.978 | 94.4 % | 83.2 % | 23.1 % |
| **v2 + generate-and-verify** (8, T = 0.5) | 3.1 % | 19.5 % / 98.7 % | **0.960** | **0.994** | 93.8 % | **87.1 %** | **58.0 %** |
| v2 + generate-and-verify, **random pose** (signed axis permutation + ±100 mm) | 1.8 % | 15.2 % / 98.7 % | 0.932 | 0.982 | 85.7 % | 71.2 % | 40.4 % |

Copying **triples the share of near-perfect parts** (IoU > 0.99: 13.9 % → 44.3 % greedy, 23.1 % → 58.0 %
with verification) — numbers are now exact rather than within ~0.5 mm. In random poses the model is
weaker (0.932 mean, 40 % > 0.99; 25 % of training items were unposed) but still ahead of v1 in its home
placement. Per family (verified, median IoU / > 0.95 / > 0.99, v1 → v2):

| family | median IoU | IoU > 0.95 | IoU > 0.99 |
|---|---|---|---|
| flange | 0.990 → 0.997 | 94 % → 93 % | 49 % → 74 % |
| t_block | 0.979 → 0.997 | 84 % → 85 % | 19 % → 68 % |
| l_bracket | 0.979 → 0.995 | 88 % → 89 % | 17 % → 62 % |
| gusset_bracket | 0.975 → 0.995 | 82 % → 95 % | 12 % → 64 % |
| hub_disc | 0.990 → 0.993 | 95 % → 93 % | 50 % → 55 % |
| u_channel | 0.969 → 0.992 | 75 % → 89 % | 11 % → 52 % |
| block | 0.969 → 0.991 | 78 % → 85 % | 5 % → 52 % |
| stepped_shaft | 0.969 → 0.990 | 72 % → 76 % | 17 % → 49 % |
| plate | 0.974 → 0.987 | 77 % → 81 % | 10 % → 47 % |

`examples/SimpleRectangle.stp` — v1: IoU 0.21 (wrong placement convention); **v2: IoU 1.0, strict
rebuild**, the 122.1 × 88.8 × 20 mm dimensions copied from the input:

```
SKETCH +X -10 PROFILE LOOP LINE -61.05 -44.4 LINE 61.05 -44.4 LINE 61.05 44.4 LINE -61.05 44.4
PAD EXT+ 20
```

### Remaining bottleneck: fillet / chamfer edge references

Strict rebuilds stay at ~15–20 %. Of the 846 greedy strict failures, **644 are edge references** (fillet
543, chamfer 101; median miss 2 mm). Teacher-forced copy accuracy per number role (400 test parts):

| number role | share | copyable | copy pick accuracy |
|---|---|---|---|
| sketch curve coordinates | 39 % | 99.1 % | 94.2 % |
| **edge-reference point coordinates** | **25 %** | 97.4 % | **64.1 %** |
| sketch plane offsets | 23 % | 99.7 % | 86–93 % |
| fillet / chamfer size | 3 % | 99.6 % | 80.0 % |
| extrude lengths, hole parameters | 10 % | 97–100 % | 80–98 % |

An edge ref is the midpoint of an edge of the *intermediate* body — an edge the fillet itself removed,
so it does not exist in the input; the model must assemble x, y, z independently from candidates.
Two findings drive phase 3b:

1. **21 % of all edge refs in synth_v2 are no-ops**: the generator picked smooth edges (tangent joins
   of earlier fillets, arc–line seams of slots, periodic seams). Filleting them changes nothing, so they
   cannot be inferred from the solid — pure label noise the model learned to imitate.
   `cadseq.build.prune_smooth_edges` removes them; the pruned histories rebuild the same solids
   (IoU ≥ 0.9997 on 110 test parts, no features lost).
2. **The removed sharp edges can be recovered analytically** (`cadseq/blends.py`): intersect the
   blend face's two support surfaces (`GeomAPI_IntSS`) and project the blend face onto the curve.
   ~0.1 s per part, label-free; **94–96 % of the remaining refs lie on a recovered edge**, and
   histories re-targeted to recovered points rebuild strictly with IoU 1.0. (OCC's
   `BRepAlgoAPI_Defeaturing` was tried first: up to 100 s per face and no usable history.)

### Phase 3b — edge pointer (done)

- `cadseq/edgefix.py`: prunes smooth refs, computes virtual edges for every face, links each ref to
  the virtual edges lying on its exact intermediate-body edge (a pointer *group*), moves the ref point
  onto the grid-snapped virtual point and verifies the rebuild (volume) → `tokens_ef.npz`,
  `graphs/vedge_*.npy`. synth_v2 (20K, 8 workers, ~45 min): 153,210 refs, **32,594 smooth (21.3 %)
  pruned, 113,277 of the remaining 120,616 (93.9 %) linked** to a virtual edge; 189 samples (0.9 %) keep
  their original tokens (replay or volume check failed); ~200 virtual edges per part.
- `--edge-ptr`: an `EdgeHead` scores virtual edges (point Fourier features + curve type + embeddings of
  the blend face and its support faces). After an `E_LINE` / `E_CIRCLE` token a gate chooses "point at
  virtual edge k" (its x, y, z tokens are forced) or falls back to copying the coordinates; already
  used edges are masked. Loss: `−log σ(g) − log Σ_group π_k` for refs with targets, `−log σ(−g)` otherwise.
- Sanity check: 64 training parts memorised → pointer used for 91 % of generated edge refs, 98.4 %
  strict rebuilds, IoU 0.99996 (token-exact 91 %: the pointer may pick another point on the same edge).

**Results — `runs/v3` (60 epochs), test split, 1000 parts:**

| model / inference | exact | built (strict / lenient) | IoU mean | IoU median | > 0.9 | > 0.95 | > 0.99 |
|---|---|---|---|---|---|---|---|
| v2 greedy | 2.8 % | 15.4 % / 95.6 % | 0.909 | 0.987 | 85.5 % | 74.8 % | 44.3 % |
| **v3 greedy** | 4.6 % | **32.7 %** / 95.8 % | 0.911 | 0.989 | 85.0 % | 74.7 % | 47.8 % |
| v2 + generate-and-verify (8, T = 0.5) | 3.1 % | 19.5 % / 98.7 % | 0.960 | 0.994 | 93.8 % | 87.1 % | 58.0 % |
| **v3 + generate-and-verify** (8, T = 0.5) | 5.8 % | **42.4 % / 99.5 %** | **0.966** | **0.994** | **94.2 %** | **88.1 %** | **60.0 %** |

The edge pointer **more than doubles strict rebuilds** (the predicted history rebuilds exactly as written,
no edge snapping or skipped features) and gives the best result on every metric. Teacher-forced edge-pick
accuracy 84 % (v2: 64 %). `runs/v3/best.pt` is the current recommended checkpoint. Validation loss was still
falling at epoch 60 — longer training is a cheap later gain.

```powershell
.venv\Scripts\python.exe -m cadseq.edgefix data\synth_v2 --workers 8     # once
.venv\Scripts\python.exe -m s2pnet.train --data data\synth_v2 --out runs\v3 --epochs 60 --bs 24 --attrs --copy --aux-seg 0.2 --edge-ptr --pose-aug 0.75 --loader-workers 4
```

## Geometry-first reconstruction (`recon/`, no learning)

On real STEP files the v3 model failed badly (`WeirdPart.stp` IoU 0.41, a 54-face production part
0.14): the structure it generates is wrong whenever the part leaves the synthetic distribution (tilted
planes, off-grid dimensions, lengths beyond the token range). `recon/` turns the problem around:
every feature is **read from the exact B-Rep of the input** and a search only *chooses* among them,
with the CAD kernel verifying every choice. No training data, no tokens, no grid — values are the
STEP's own numbers (e.g. 59.8134 mm, not 59.8).

```
STEP → exact faces (plane / cylinder / cone / torus …), sample points (uniform + ±δ off every face)
  ├─ pad / pocket / trim candidates: for every extrusion direction (plane normals, cylinder axes —
  │    tilted ones included) the planes ⊥ d give levels; each slab between levels is sectioned
  │    (OCC) at several heights and every representable outline (lines / arcs / circles) becomes a
  │    sketch, extruded over any span of levels. Trim = cut everything outside an outline, which
  │    intersects prisms of different directions (side profile × front outline).
  ├─ hole candidates: coaxial concave cylinders / cones → simple / counterbore / countersink,
  │    blind or through, from either end (countersinks extended through a chamfer on their edge)
  ├─ beam search: minimise the weighted point error (each face weighs the same as the bulk) +
  │    a small cost per feature; every candidate has an analytic inside-test, so scoring needs no
  │    booleans
  ├─ kernel rebuild (cadseq.build.Builder); rejected features are banned and the search re-planned
  ├─ blends: unexplained fillet faces (tangent cylinder / torus) and chamfer faces (planar strip,
  │    cone on a circular edge) → recovered sharp edge (cadseq.blends) + exact radius / distance,
  │    kept only if they reduce the error
  └─ mirror: if the part is symmetric, the half part is reconstructed and mirrored; the shorter
       history of equal quality wins
→ <out>.json (strictly rebuildable), <out>_rebuilt.step, <out>_report.json,
  <out>_unexplained.step (input faces the history does not explain, in red — what is left for the designer)
```

```powershell
.venv\Scripts\python.exe -m recon.reconstruct part.stp --out runs\recon\part
.venv\Scripts\python.exe -m recon.bench --data data\synth_v2 --split test --n 200 --workers 12 --timeout 300 --name test200
.venv\Scripts\python.exe -m recon.bench --steps C:\parts\*.stp --workers 4 --name real
```

OpenCASCADE sometimes kills the process (access violation) inside fillet / chamfer calls, which no
`try` can catch. Blend calls are journaled; `recon.reconstruct` runs in a child process and on a crash
restarts it, skipping the blend group that crashed (`recon.bench` does the same per part).
Point classification dominated the runtime, so surface samples are labelled by construction (agrees
with the kernel classifier on 99.97 % of 255K samples) and the rebuilt body is predicted from the
candidates' analytic inside-tests.

**Results — first 200 parts of the test split** (same parts for every row; failures count as IoU 0):

| method | IoU mean | IoU median | > 0.95 | > 0.99 | > 0.999 | strict rebuild |
|---|---|---|---|---|---|---|
| v3 greedy | 0.894 | 0.987 | 76.5 % | 44.0 % | 14.5 % | 31.0 % |
| v3 + generate-and-verify (8, T = 0.5) | 0.957 | 0.993 | 87.0 % | 57.0 % | 24.0 % | 39.0 % |
| **recon** | **0.979** | **0.9995** | **92.0 %** | **81.0 %** | **60.5 %** | **99.0 %** |

Median 9.5 s per part (90th percentile 38 s, 12 workers); 1 of 200 timed out (123 faces, > 300 s).
Histories have a median 0.88× the ground-truth feature count. Weakest family: `stepped_shaft`
(53 % > 0.99) — turned parts are built from stacked pads, revolve is not in the vocabulary yet.
17 % of input faces stay unexplained, mostly fillets / chamfers whose sharp edge does not exist in the
rebuilt body and small cone / torus faces.

Real parts (no history known):

| part | v3 + G&V IoU | recon IoU | features | unexplained faces | time |
|---|---|---|---|---|---|
| `examples/SimpleRectangle.stp` | 1.0 (v2) | **1.0**, strict | 1 pad | 0 / 6 | 2 s |
| `WeirdPart.stp` (13 faces, off-grid) | 0.41 | **1.0**, strict | pad + 2 trims | 0 / 13 | 1 s |
| `6439884_PIA_WEBBING.stp` (54 faces, 600 mm, tilted bend) | 0.14 | **0.979**, strict | pad + 2 trims | 5 / 54 (top end: 2 small R10 corner rounds, 2 angled end faces, 1 step face) | 12 s |

Next: revolve candidates (turned parts), sharp-edge reconstruction for fillets whose edge is missing
in the rebuilt body, and a learned candidate scorer trained by self-training on real STEP files
(search-verified histories as targets).
