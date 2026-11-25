# features — STEP solid → structured feature list (JSON)

A mini side project inside `solid2param`. Input: a "dumb" STEP solid (no history). Output: a
structured JSON describing **which features the part contains** — holes (with type, diameter,
depth, axis), hole patterns, fillets, chamfers, extrusions / cuts with their profile (rectangle,
circle, slot, …), symmetry — each with **exact values read from the B-Rep** and the **face ids** that
form it. It is meant as the "understand the CAD" front end of an automated **design review**: a rule
engine can check a guideline document (min hole Ø, standard sizes, fillet radii, edge distances …)
against this JSON, and every finding can point at concrete faces.

```powershell
# from solid2param/
.venv\Scripts\python.exe -m features.extract part.stp                         # → runs\features\part_features.json / .step
.venv\Scripts\python.exe -m features.extract part.stp --out out\part --no-seg  # rules only, no segmenter
.venv\Scripts\python.exe -m features.render runs\features\part_features.json   # → runs\features\part_render.png
.venv\Scripts\python.exe -m features.evaluate --data data\synth_v2 --split test --n 200 --workers 8
```

| file | content |
|---|---|
| `extract.py` | CLI + orchestration: recognizers → tie-breaks → ids → JSON / coloured STEP |
| `recognize.py` | rule-based recognizers (holes, fillets, chamfers, extrusions, patterns, symmetry) |
| `crosscheck.py` | segmenter second opinion: per-feature agreement, conflicts, hints for unrecognised faces |
| `render.py` | PNG of the part with faces coloured by feature (no CAD viewer needed); `--views above below side top above_back below_back` |
| `evaluate.py` | scoring against the known histories of the synthetic dataset |

## 1. Why rules first, and where the ML model fits

A STEP file stores the final B-Rep with **exact analytic surfaces** (plane, cylinder, cone, torus,
sphere, B-spline). For the common mechanical features the geometry alone determines the feature
and all its parameters:

| feature | geometric signature | parameters read from |
|---|---|---|
| hole | coaxial **concave full (360°) cylinders / cones** chained along one axis | cylinder radius, axial extents, cone semi-angle, floor plane |
| fillet | cylinder / torus (or free-form) face **tangent (G1)** to two neighbours | surface radius (torus: minor radius), rolling-ball centre side |
| chamfer | narrow planar / conical strip meeting two neighbours **at an angle** | distance from the recovered sharp edge to the strip's sides |
| extrude / cut | slab between planes ⊥ a direction is a **prism** → its section is the sketch | exact OCC section of the solid |

So no model is needed to *find* these features or to *measure* them — and rules are exact,
explainable and deterministic, which a review needs. The learned segmenter (`runs/seg`, 99.8 % face
accuracy on synthetic test parts) is used where geometry alone **cannot decide**:

1. **Tie-breaks between geometrically identical readings** (§4): fillet vs rounded sketch corner,
   chamfer vs bevelled sketch side, hole vs circular pocket. The segmenter was trained on the real
   modelling histories, so it knows which reading a designer used.
2. **Cross-check** (§5): every feature gets an agreement score; confident contradictions are listed
   as conflicts — the first things a human reviewer should look at.
3. **Hints for unrecognised faces**: the segmenter's class for faces no rule explains.

The pipeline still produces a valid result without the segmenter (`--no-seg`, or if torch / the
checkpoint are missing).

## 2. Pipeline

```
STEP ─► OCC B-Rep ─► per face: surface type + exact parameters, outward normals, samples (recon.geom)
                    face adjacency + shared edges
   1. holes       concave 360° cylinder / cone segments grouped by axis → chains of touching segments
                  → entry / exit, floor, head type, depth                 (recognize_holes)
   2. fillets     cylinder (< 360°) / torus / free-form faces tangent to 2 neighbours; sharp edge
                  recovered by intersecting the two support surfaces       (fillet_items, cadseq.blends)
   3. chamfers    planar / conical faces with ≥ 2 angled neighbours, narrow  (chamfer_items)
   (segmenter, optional: one class per face)
   4. hole ↔ circular pocket tie-break (segmenter)
   5. extrusions  geometry-first decomposition of recon (exact sketches in any direction), faces
                  assigned to the latest tool whose boundary they lie on   (recognize_extrusions)
   6. blend ↔ sketch-arc / sketch-line tie-break (segmenter, default: blend)
   7. corner patches between blends, ids, hole patterns, symmetry planes
   8. cross-check: agreement, conflicts, hints                            (crosscheck.annotate)
─► <out>_features.json + <out>_features.step (faces coloured by feature type, unrecognised = yellow)
```

### 2.1 Holes

* Concave cylinders / cones with the same axis (parallel, < 1e-6·diag apart) are grouped; faces of a
  split surface are merged into one **segment** when kind, radii and axial range agree; only
  segments covering 360° are hole candidates (a partial concave cylinder is a fillet or a broken hole).
* Touching segments along the axis form a **chain** (e.g. cone → Ø11 cylinder → Ø6.6 cylinder).
* **Edge cones are stripped** and left for the chamfer recognizer: a cone sitting on a countersink
  cone or on a counterbore (entry chamfer), and a cone narrowing into a flat floor (bottom-edge
  chamfer). A cone narrowing to ~0 is a **drill point** and stays.
* Each end is **closed** if a planar neighbour ⊥ axis lies entirely within the end radius at that
  end (a floor) or the end is a drill point; otherwise open. Two open ends → **through**.
* **Entry** = the open end with a head (larger cylinder → counterbore, cone → countersink),
  else the open end. `direction` points from the entry into the material, `position` is on the
  entry plane; `depth` (blind) is measured from entry to floor (or to the start of the drill point).
* Annular shoulders between the ends (counterbore floor) belong to the hole.
* Subtypes: `simple`, `counterbore` (`cb_diameter`, `cb_depth`), `countersink` (`cs_diameter`,
  `cs_angle`), `stepped` (> 2 diameters); `segments` lists the full chain as evidence.

### 2.2 Fillets

Candidates: cylinders spanning < 360°, tori, free-form (B-spline) faces. `cadseq.blends.sharp_edges`
requires two **tangent** neighbours (normals within ~2.6° across the shared edge), intersects their
surfaces and projects the blend onto the intersection — the edge the fillet replaced (reported as
`edges: [{curve, point}]`, the same edge-reference format as the design-history JSON).
`radius` = cylinder radius / torus minor radius (`null` + `variable_or_freeform` for B-spline
blends). `convexity`: the rolling-ball centre `p − r·n` lies on the cylinder axis / torus spine for a
convex (material-removing, outside-corner) fillet. Connected fillet faces of equal radius and
convexity form one feature (a fillet chain); spherical corner patches of the same radius join it.

### 2.3 Chamfers

Candidates: planar / conical faces with ≥ 2 neighbours whose normals make an angle with the face
(0.05 < cos < 0.999 — not tangent, not perpendicular). The sharp edge is recovered as for fillets;
the chamfer distances are the distances from that edge to the strip's two long sides. Accepted if
both distances are > 0, differ by ≤ 4×, and the strip is **narrow**: width ≤ 0.5 × its length and
≤ 6 % of the part's bounding-box diagonal. The width limits exist because a bevelled side of a
sketch profile is geometrically a chamfer too; very wide "chamfers" are almost always profile
geometry (on the test parts every rejected one was labelled extrude side by the segmenter).
Equal distances → `distance`, otherwise `distances: [d1, d2]`; `edge_angle` is the angle between
the support faces (90 = square edge).

### 2.4 Extrusions and cuts

Reuses the geometry-first decomposition of `recon/` (see the main README): for every direction
given by plane normals and cylinder axes — tilted ones included — the planes ⊥ the direction define
levels, each slab between levels is sectioned by OCC and every representable outline becomes a
sketch; a beam search picks pads / pockets / outline trims that reproduce the solid, and the kernel
verifies them. Each kept tool becomes a feature:

* `type`: `extrude` (pad) or `cut` (pocket; `subtype: outline_trim` for a cut of everything outside
  an outline — a modelling device that intersects profiles of different directions).
* `profile`: the loop classified as `circle` (`diameter`), `rectangle` (`width`, `height`),
  `rounded_rectangle` (`corner_radius`), `slot` (`width`, `length`), `polygon` (`sides`,
  `regular`) or `general` (`lines`, `arcs`), with `area`, `size_uv`, `center_uv` in the sketch frame.
* `direction`, `sketch_plane` (origin, normal, x_dir), `start_level`, `end_level`, `length` / `depth`
  / `through`.
* `faces`: input faces lying on the tool's boundary (material on the inner side for a pad, air on
  the outer side inside the cut tool for a cut), checked on ±δ samples of each face; a face goes to
  the **latest** tool it lies on. `end_faces` are the planar ones ⊥ direction.

A feature can end with `faces: []` when later tools re-cut all of its boundary (it still defines
material). Extrusion recognition is the slow step (seconds); `--no-extrusions` skips it.

### 2.5 Patterns and symmetry

Holes with the same subtype, diameters and axis direction are grouped; positions are compared in
the plane ⊥ axis with a **0.05 mm** tolerance (entry faces may sit at different heights; the
synthetic generator snaps positions to a 0.05 mm grid). Tested in order: `linear` (collinear, equal
pitch), `grid` (two directions, equal pitches), `circular` (equal radius, equal angular step;
`pcd`, `angle_step`, `full_circle`), else `identical_set`. `uniform_depth` / `coplanar_entries`
flag members that differ (another feature broke into one). Symmetry: mirror planes through the
bounding-box centre whose reflection keeps every sample's inside / outside label (`recon`).

## 3. Output JSON (`features-1.0`)

Abridged example (`00000229.step`, a flange; `…` = omitted):

```json
{
 "schema": "features-1.0", "part": "…/00000229.step", "units": "mm",
 "geometry": {"bbox": {"min": […], "max": […]}, "size": [103, 103, 47], "volume": …, "area": …,
              "surface_types": {"plane": 20, "cylinder": 30, …}},
 "symmetry_planes": [{"origin": […], "normal": [0, 1, 0]}],
 "features": [
  {"id": "E1", "type": "extrude", "subtype": "circle", "direction": [0, 0, 1],
   "sketch_plane": {"origin": [0, 0, 0], "normal": [0, 0, 1], "x_dir": [1, 0, 0]},
   "profile": {"shape": "circle", "diameter": 103.0, "area": 8332.3, "center_uv": [0, 0]},
   "start_level": 0.0, "end_level": 14.5, "length": 14.5, "faces": [5], "end_faces": [5],
   "seg": {"status": "agree", "agreement": 1.0, "classes": {"extrude_end": 1}}},
  {"id": "H1", "type": "hole", "subtype": "simple", "diameter": 3.0, "through": false, "depth": 9.0,
   "bottom": "flat", "axis": [0, 0, 1], "direction": [0, 0, -1], "position": [32.9, -19.0, 14.5],
   "length": 8.5, "edge_chamfers": 1, "segments": [{"surface": "cylinder", "diameter_start": 3.0, …}],
   "faces": [0, 3], "seg": {"status": "agree", …}},
  {"id": "F2", "type": "fillet", "radius": 1.5, "convexity": "convex", "variable_or_freeform": false,
   "edges": [{"curve": "circle", "point": […]}, …], "surfaces": ["torus"], "faces": [32, …],
   "alternatives": [], "seg": {…}},
  {"id": "C1", "type": "chamfer", "distance": 0.5, "distances": null, "edge_angle": 90.0,
   "edges": [{"curve": "circle", "point": […]}], "faces": [1], "alternatives": [], "seg": {…}}
 ],
 "patterns": [{"type": "hole_pattern", "kind": "circular", "count": 6, "pcd": 75.99,
               "angle_step": 60.0, "full_circle": true, "center": […], "axis": […],
               "holes": ["H1", "H2", "H4", "H8", "H11", "H13"],
               "hole": {"subtype": "simple", "diameter": 3.0},
               "uniform_depth": false, "coplanar_entries": false}],
 "ambiguous_faces": [{"faces": [44], "readings": ["fillet", "side of E4"],
                      "chosen": "extrusion side", "by": "segmenter"}],
 "unrecognised_faces": [{"face": 21, "surface": "plane", "area": 12.4}],
 "faces": [{"index": 0, "surface": "plane", "area": …, "feature": "H1",
            "seg_class": "hole_simple", "seg_prob": 1.0}, …],
 "summary": {"faces": 62, "faces_explained": 61, "features": {"extrude": 5, "cut": 1, "hole": 15, …},
             "hole_simple": 15, "hole_patterns": 3, "fillet_radii": [1.5],
             "chamfer_sizes": [0.5], "hole_diameters": [2.0, 3.0]},
 "cross_check": {"model": "runs/seg/last.pt", "matched_by": "order",
                 "features": {"agree": 29, "partial": 0, "disagree": 3, "no_faces": 0},
                 "conflicts": [{"face": 27, "rule_feature": "E4", "seg_class": "cut_side", "seg_prob": 1.0}],
                 "unrecognised_hints": [{"face": 21, "seg_class": "cut_end", "seg_prob": 0.98}]},
 "log": [], "seconds": 6.7
}
```

* Lengths in mm, angles in degrees, values rounded to 1e-6 mm (exact otherwise — e.g. 59.8134, not 59.8).
* Face indices follow OCC explorer order (`TopExp_Explorer`), the same order as the coloured STEP,
  the segmenter and `step_to_graph.py`.
* Feature ids: `E` extrude, `K` cut, `H` hole, `F` fillet, `C` chamfer.

## 4. Geometrically ambiguous readings

Some faces have two equally valid readings with **identical geometry**; no rule can choose:

| faces | reading A | reading B | tie-break |
|---|---|---|---|
| cylinder face parallel to an extrusion, tangent to its neighbours | fillet on that edge | arc of the extrusion's sketch (rounded rectangle, slot end) | segmenter; default fillet |
| planar strip at an angle, parallel to an extrusion | chamfer | line of the sketch (bevelled corner) | segmenter; default chamfer |
| flat-bottomed (or through) plain cylinder | hole | circular pocket | segmenter; default hole |

Every decision is recorded in `ambiguous_faces`; a blend that was kept lists the extrusion as
`alternatives`. For a design review both readings carry the same numbers (a R5 fillet and a R5
sketch corner are the same radius), so rule checks are unaffected — only naming differs.

## 5. Segmenter cross-check

`crosscheck.segment_faces` runs `identify.identify` (`runs/seg/last.pt`) and maps its faces to ours
by face order when surface type and area agree for every face, otherwise by nearest centroid of the
same surface type. Expected classes per feature: hole subtype → `hole_<subtype>`, fillet → `fillet`,
chamfer → `chamfer`, extrude → `extrude_side/end`, cut → `cut_side/end`, outline trim → extrude or
cut (it is a modelling device). Per feature: `seg.status` = `agree` (≥ 2/3 of its faces have an
expected class), `partial` (≥ 2/3 have the right family, e.g. simple vs counterbore hole) or
`disagree`; `conflicts` = faces where the segmenter is ≥ 90 % sure of a different family.

Disagreement does not mean the rule is wrong: the segmenter predicts the generator's modelling
choice (e.g. a boss modelled as pad-up vs a pocket-down around it, `extrude` vs `cut`), while the
rule reports what the geometry is. Conflicts are a review queue, not an error list.

## 6. Results — 200 synthetic test parts (with known histories)

`features.evaluate` compares the extracted features with each part's ground-truth history (holes
matched by axis line + diameter; a final mirror is applied to the ground-truth holes; fillet / chamfer
values compared as sets per part). Full output: `runs/features/eval_test200_summary.json` (also per family).

### 6.1 Overall

199 of 200 parts processed (1 timeout: a 123-face part in the extrusion search, > 300 s).

| metric | value |
|---|---|
| holes — recall (history holes found, same axis + Ø) | **92.9 %** (1064 holes) |
| holes — precision | **88.8 %** (79.7 % without the hole ↔ circular-pocket tie-break) |
| holes — type correct (simple / counterbore / countersink) | 90.5 % |
| fillet radii — recall / precision | **96.9 %** / 93.3 % (286 values) |
| chamfer sizes — recall / precision | 83.0 % / **98.1 %** (188 values) |
| input faces explained by a feature | **95.9 %** (52 % of parts fully explained) |
| features where the segmenter agrees | 84.2 % |
| segmenter conflicts per part (review queue) | 7.8 |
| time per part (median, 8 workers, incl. segmenter) | 7.1 s |

Spot checks against the histories: the circular 6× Ø3 (PCD 76) and 7× Ø2 (PCD 70) hole patterns of a
flange, counterbores Ø3.4 / Ø6.5 × 3.3, countersinks Ø3.4 / 90°, a fillet chain R2 over 8 edges,
0.5 chamfers on hole edges and blend corners — all with the history's exact values.

Notes on reading the numbers:

* **Hole precision**: the remaining extra holes are mostly circular pockets the segmenter did not
  claim (on a 40-part sample, before the tie-break, 21 of 27 unmatched holes were circular pockets
  in the history — identical geometry, §4). **Hole recall** misses holes another feature broke
  into (no longer a full 360° cylinder) or that a later pocket removed.
* **Hole type**: a chamfer on a countersink edge or a counterbore whose head reaches the wall can
  change what is visible; the rule reports the visible geometry.
* **Chamfer recall**: the history's chamfer value is missing when the chamfer was later consumed
  (e.g. a 2.5 chamfer on a countersink edge leaves a different visible cone), when it was applied to
  an edge along an extrusion and the segmenter read it as sketch geometry, or when its strip is
  wider than the narrowness limits.
* Times are per part on 8 workers, including the segmenter.

## 7. Limitations and next steps

* Feature vocabulary: no revolves (turned parts appear as stacked circular extrusions), ribs,
  bosses/lugs as named features, threads (cosmetic threads are not in a dumb STEP), sheet-metal
  bends, drafts. Slots are recognised as extrusion / cut profiles, not as standalone features.
* Free-form (B-spline) blends get `radius: null`; variable-radius fillets are not measured.
* Through-hole entry for symmetric holes is arbitrary (either end is valid).
* The two real parts used earlier (`WeirdPart.stp`, `6439884_PIA_WEBBING.stp`) were no longer on
  disk when this was built; validation on real parts is still to do. `examples/SimpleRectangle.stp`:
  1 rectangular extrude 122.1 × 88.8 × 20 mm, all faces explained, segmenter agrees.
* Next: the design-review rule engine on top of this JSON (guideline Markdown → structured rules →
  deterministic checks → findings with face ids), plus OCC measurements the rules need (hole-to-edge
  distance, wall thickness, hole spacing).
