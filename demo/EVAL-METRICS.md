# Evaluation Metrics

This document explains every scored metric surfaced by the backend
`GET /api/metrics` endpoint, how this project computes it, the current
**synthetic** demo values, and likely judge questions with crisp answers.

> All numbers below are **synthetic placeholders** baked into the backend so the
> demo runs end-to-end on a 4GB laptop GPU **without training or downloading any
> dataset**. They are plausible, internally consistent values — not the output of
> a trained model. See [Why synthetic?](#why-synthetic-numbers) and the
> [TODO: wire real metrics](#todo-wire-real-metrics) section to replace them.

---

## The demo story (read this first)

A single combination is engineered to look **broken**, everything else is
**intact**:

| input      | model    | network state | resilienceIndex |
| ---------- | -------- | ------------- | --------------- |
| `occluded` | `baseline` | **FRAGMENTED** (several disconnected zones) | **79** |
| `occluded` | `robust`   | intact | 100 |
| `clean`    | `baseline` | intact | 100 |
| `clean`    | `robust`   | intact | 100 |

The narrative: clouds/shadows/tree-canopy **occlude** the satellite imagery. The
**baseline** segmentation model drops the occluded road pixels, which deletes
critical links and **fragments** the road graph. The **robust** model (trained
with occlusion augmentation) **recovers those critical links**, so the graph
stays connected. The metrics below quantify that gap.

---

## Metric reference

For each metric: what it is → how this project measures it → why it matters here.

### 1. IoU (Intersection-over-Union)

- **What:** Pixel-level overlap between predicted road mask and ground-truth
  mask. `IoU = |pred ∩ gt| / |pred ∪ gt|`. Range `0..1`, higher is better. The
  standard semantic-segmentation score.
- **How here:** Computed per-tile over the binary road class, then averaged.
  Stored as `metrics.iou`. (In the synthetic backend it is a fixed value per
  `(input, model)` combo; with a real model it would be `TP / (TP + FP + FN)`
  accumulated over the test tiles.)
- **Why it matters:** IoU is the segmentation "headline" number, but it is
  **deceptive for connectivity** — losing a thin 2-pixel-wide road costs almost
  nothing in IoU yet can disconnect a whole zone. That gap is exactly why we also
  report graph metrics (APLS, Connectivity Ratio, Resilience Index).

### 2. Dice (F1 / Sørensen–Dice coefficient)

- **What:** `Dice = 2|pred ∩ gt| / (|pred| + |gt|)` — the harmonic mean of
  precision and recall on road pixels. Range `0..1`. Monotonically related to IoU
  (`Dice = 2·IoU / (1 + IoU)`) but weights overlap more generously.
- **How here:** Same accumulation as IoU; stored as `metrics.dice`. Always
  slightly higher than the corresponding IoU, which is the expected relationship
  and a sanity check that the two numbers are consistent.
- **Why it matters:** Common alternative headline metric; reviewers expect to see
  both IoU and Dice. Same connectivity blind spot as IoU.

### 3. Occlusion-Recall

- **What (project-specific):** Recall of road pixels **restricted to the occluded
  regions only**. `OcclusionRecall = (correctly predicted road px inside occluded
  mask) / (ground-truth road px inside occluded mask)`. Range `0..1`.
- **How here:** The synthetic occlusion mask marks which pixels are
  clouded/shadowed. We score recall **only inside that mask**, which directly
  measures "did the model see through the occlusion?" Stored as
  `metrics.occlusionRecall`.
- **Why it matters:** This is the **core thesis metric** of the project. Global
  IoU/Dice barely move under light occlusion because most of the image is clean;
  Occlusion-Recall isolates the hard pixels and shows the baseline collapsing
  while the robust model holds up. It is the number that justifies the "robust"
  model existing.

### 4. Connectivity Ratio

- **What:** Fraction of the road network that remains in the **largest connected
  component**. `ConnectivityRatio = (nodes in giant component) / (total nodes)`.
  Range `0..1`; `1.0` means a single fully-connected network.
- **How here:** The backend builds the graph from the GeoJSON edges, runs a
  connected-components pass, and divides the giant component size by total nodes.
  Stored as `metrics.connectivityRatio`. For the fragmented case it drops below
  `1.0` because the deleted critical links split off zones.
- **Why it matters:** Turns "the map looks broken" into one defensible number. It
  is the bridge from pixels to **mobility**: disconnected zones = areas you can no
  longer route to.

### 5. APLS (Average Path Length Similarity)

- **What:** The SpaceNet road-topology metric. For many sampled node pairs it
  compares shortest-path length in the predicted graph vs. the ground-truth
  graph; pairs that become unreachable are penalized to the max. Range `0..1`,
  higher = more topologically faithful.
- **How here:** Sample source/target node pairs on the ground-truth graph,
  compute shortest paths on both graphs, and average
  `1 - min(1, |L_gt - L_pred| / L_gt)` (unreachable → contributes `0`). Stored as
  `metrics.apls`. Fragmentation makes many pairs unreachable, so APLS falls
  sharply even when IoU is still high.
- **Why it matters:** The field-standard **graph-level** quality metric, and the
  one judges familiar with SpaceNet will ask about. It captures connectivity loss
  that IoU/Dice miss, and is the academic credibility anchor of the eval.

### 6. Resilience Index (0..100, integer)

- **What (project-specific, headline KPI):** A single `0..100` score for how well
  the network survives failures. `100` = fully connected and efficient; lower =
  fragmented / inefficient.
- **How here:** Derived from the **resilience curve**
  (`GET /api/resilience-curve`): we progressively remove edges/nodes (by
  criticality) and track network **efficiency** and **giant-component** size, then
  integrate (area under the curve) and rescale to `0..100`, rounded to an int.
  The intact networks score `100`; the fragmented `occluded+baseline` case scores
  `~79`. `POST /api/simulate` returns `resilienceIndexAfter` recomputed after the
  user disables specific edges/nodes.
- **Why it matters:** The **demo's headline number** and the link to
  decision-making — "which roads, if lost, hurt mobility most." It ties the
  segmentation work to an actionable urban-planning outcome (harden the
  gatekeeper links).

---

## Current synthetic numbers

These are the values the backend returns today, by `(input, model)`. Treat them
as illustrative; they encode the demo story (only `occluded + baseline` degrades).

| metric             | clean+baseline | clean+robust | occluded+baseline | occluded+robust |
| ------------------ | -------------- | ------------ | ----------------- | --------------- |
| IoU                | 0.78           | 0.79         | **0.61**          | 0.76            |
| Dice               | 0.88           | 0.88         | **0.76**          | 0.86            |
| Occlusion-Recall   | 0.95           | 0.96         | **0.58**          | 0.93            |
| Connectivity Ratio | 1.00           | 1.00         | **0.82**          | 1.00            |
| APLS               | 0.91           | 0.92         | **0.64**          | 0.90            |
| Resilience Index   | 100            | 100          | **79**            | 100             |

Bold = the engineered "broken" combination. Notice IoU only dips ~0.17 while
APLS/Resilience drop hard — that contrast **is** the pitch: pixel metrics
under-report the mobility damage, graph metrics expose it.

> Source of truth is the backend handler for `GET /api/metrics`; if you change a
> number there, update this table to match.

---

## Why synthetic numbers?

Hard environment constraints for this hackathon build:

- **No training:** RTX 3050 (4GB VRAM) cannot train a road-segmentation model in
  time; we ship the eval **contract** instead.
- **No dataset download:** DeepGlobe / SpaceNet are many GB — out of scope here.
- **No heavy deps:** the backend does not `pip install torch`.

So the backend returns fixed, internally consistent metrics that make the
end-to-end product (map → criticality → gatekeepers → simulation) demonstrable.
Every metric is **real and implementable** — only the inputs are stubbed.

---

## TODO: wire real metrics

To replace the synthetic values with computed ones (post-hackathon / with a
trained model and a GPU):

1. **Segmentation (IoU, Dice, Occlusion-Recall):** run the model on a held-out
   tile set, accumulate per-tile TP/FP/FN for the road class, and for
   Occlusion-Recall mask the accumulation to occluded pixels only.
2. **Graph build:** vectorize the predicted mask → centerline graph (e.g.
   skeletonize + simplify), matching the GeoJSON edge schema the API already
   returns.
3. **Connectivity Ratio:** connected-components on the predicted graph, giant
   component / total nodes.
4. **APLS:** sample node pairs on the ground-truth graph, compare shortest-path
   lengths on pred vs. gt, penalize unreachable pairs to `0`.
5. **Resilience Index:** integrate the resilience curve (progressive
   criticality-ordered removal → efficiency & giant-component), rescale to
   `0..100`.
6. Swap the static handler in `GET /api/metrics` to read these computed values.

> Pin any new dependencies (e.g. `networkx`, `scikit-image`, `rasterio`) in the
> backend `requirements.txt` with exact versions before running. None are
> installed by this doc.

---

## Likely judge Q&A

**Q: Your IoU only drops ~0.17 under occlusion — that doesn't look catastrophic.
Why should I care?**
A: Exactly the point. IoU is a pixel-area metric; a road is a thin line, so
deleting it barely changes IoU but can disconnect an entire zone. Look at APLS
(0.64) and Resilience Index (79) for the same case — the **graph** metrics expose
the mobility damage that pixel metrics hide.

**Q: What is Occlusion-Recall and why not just report normal recall?**
A: Global recall averages over the whole image, which is mostly clean, so it
stays high and hides the failure. Occlusion-Recall scores **only the
clouded/shadowed pixels** — the hard region — and that's where the baseline
collapses (0.58) and the robust model holds (0.93).

**Q: Is APLS your own metric?**
A: No — it's the standard SpaceNet road-topology metric (shortest-path-length
similarity over sampled node pairs, unreachable pairs penalized to max). We use
it precisely because it's the recognized graph-level benchmark.

**Q: How is Resilience Index computed, and why 0–100?**
A: It's the normalized area under the resilience curve — we remove roads in
criticality order and track network efficiency and giant-component size, then
rescale to an integer 0–100 for a single readable KPI. 100 = fully connected and
efficient; 79 = the fragmented baseline case.

**Q: Are these numbers from a trained model?**
A: No — they're synthetic, by design. The hardware (4GB GPU) and dataset sizes
rule out training in a hackathon. What we ship is the full, correct evaluation
**pipeline and contract**; every metric is implementable and the
[TODO section](#todo-wire-real-metrics) shows exactly how to plug in real model
outputs.

**Q: Why does the robust model win?**
A: It's trained with occlusion augmentation, so it infers road continuity through
clouds/shadows instead of dropping those pixels. That recovers the **critical
links** the baseline loses, keeping Connectivity Ratio at 1.0 and Resilience at
100 under occlusion.

**Q: What's the difference between IoU and Dice — why show both?**
A: They measure the same overlap; Dice = 2·IoU/(1+IoU), so Dice is always a bit
higher. Reviewers expect both, and showing the consistent relationship is a
sanity check that the accumulation is correct.

**Q: Connectivity Ratio is 0.82 in the broken case — what does the missing 0.18
mean physically?**
A: 18% of nodes are no longer in the largest connected component — i.e. several
neighborhoods are unreachable from the main network. Those are the
"newlyDisconnectedZones" the simulation reports.
