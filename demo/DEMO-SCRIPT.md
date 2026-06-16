# Route Resilience — Live Demo Runbook

A ~5 minute live walkthrough. The story: under sensor **occlusion**, the
**baseline** road-extraction model loses critical links and the city network
**fragments**. The **robust** model recovers those links. We then use
graph-theoretic **Gatekeepers** + a **what-if simulation** to show how a single
node failure collapses connectivity.

> One-line pitch to say out loud: *"Occlusion breaks the map, the map breaks the
> city, and we can pinpoint exactly which intersection holds it together."*

---

## 0. Pre-flight (do this BEFORE the audience is watching)

### Start the backend (FastAPI on :8000)
```bash
cd "<repo-root>/backend"        # quote the path: it contains spaces
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Start the frontend (Vite dev server, usually :5173)
```bash
cd "<repo-root>/frontend"        # quote the path: it contains spaces
npm install
npm run dev
```

### Smoke-test the backend contract (60 seconds, catches 90% of demo failures)
```bash
# Health: expect {"status":"ok", "service":..., "baseEfficiency":...}
curl -s http://localhost:8000/api/health

# Cities: expect ["Bengaluru"]
curl -s http://localhost:8000/api/cities

# The FRAGMENTED case — resilienceIndex should be ~79
curl -s "http://localhost:8000/api/metrics?city=Bengaluru&input=occluded&model=baseline"

# The INTACT case — resilienceIndex should be 100
curl -s "http://localhost:8000/api/metrics?city=Bengaluru&input=occluded&model=robust"
```
If those four return sane JSON, the demo will work. Leave both servers running.

**Open the app** at the Vite URL printed by `npm run dev` (e.g.
`http://localhost:5173`). Confirm the map renders and the city selector shows
**Bengaluru**.

---

## 1. The click path (the actual demo)

### Step 1 — Select the city
- Pick **Bengaluru** in the city selector.
- Network draws on the map. Mention edges are colored by **criticality (0..1)**.

### Step 2 — Establish the "healthy" baseline
- Set **Input = Clean**, **Model = Baseline**.
- Point out: network is **intact**, **Resilience Index = 100**, no disconnected
  zones. This is our reference.

### Step 3 — Introduce occlusion → FRAGMENTATION (the money shot)
- Keep **Model = Baseline**, switch **Input = Clean → Occluded**.
- The network visibly **fragments**: several **disconnected zones** appear and
  the **Resilience Index drops to ~79**.
- Say: *"Clouds/shadows/occlusion in the imagery made the baseline model drop
  real roads — and those dropped roads were load-bearing."*

### Step 4 — Recover with the robust model
- Keep **Input = Occluded**, switch **Model = Baseline → Robust**.
- The network **snaps back to intact**, **Resilience Index returns to 100**.
- Say: *"The occlusion-robust model re-infers the critical links the baseline
  lost. Same degraded input, recovered topology."*

> Demo truth table (memorize this): **only** `Occluded + Baseline` is fragmented
> (~79). Every other combination is intact (100).

### Step 5 — Switch back to the broken case for analysis
- Set **Input = Occluded**, **Model = Baseline** again (resilience ~79).
  We analyze *why* it's fragile.

### Step 6 — Open a Gatekeeper
- Open the **Gatekeepers** panel (top critical nodes by betweenness).
- Click the **top gatekeeper** (highest `betweenness`; many are also
  `isArticulation = true`).
- Highlight its **label**, **betweenness**, and that it's an **articulation
  point** — removing it splits the graph.

### Step 7 — Run the what-if (disable the gatekeeper)
- With the gatekeeper selected, run **Simulate / What-if** to disable that node.
  (This calls `POST /api/simulate` with the node id in `disabledNodeIds`.)
- Watch the response surface:
  - **Resilience Index After** drops further,
  - **avgTravelTimeIncreasePct** jumps,
  - **newlyDisconnectedZones** increases,
  - **brokenRoutesSampled / sampledRoutes** show concrete routes that now fail.
- Say: *"One intersection. Disable it and these specific routes break and travel
  time spikes — that's where to invest in redundancy."*

### Step 8 — Land the conclusion
- Re-select **Model = Robust** (still occluded) to show the recovered network is
  also more resilient to the same failure.
- Closing line: *"We turn noisy imagery into a defensible map, then tell you the
  single intersections that keep the city connected."*

---

## 2. Fallbacks (if something fails live)

### Backend down / frontend shows errors
- The frontend `src/lib/api.ts` is **fully mocked** in the current build. If the
  real backend is unreachable, the UI still demos against mock data — proceed
  with the same click path. Mention it's mock if asked; the story is identical.
- To force the mock path, simply **don't start uvicorn** (or stop it). The app
  falls back to local mock responses.

### Map tiles won't load (no internet)
- Tiles are cosmetic. The **graph overlay** (edges/nodes) is what matters —
  narrate over the abstract graph even on a blank/gray basemap.

### Resilience numbers look "wrong"
- Re-confirm the toggles. The fragmented case is **exactly**
  `Input=Occluded` + `Model=Baseline`. Any other combo = 100 by design.
- Re-run the Step-0 `curl` for `metrics` to verify the backend, then refresh.

### Gatekeeper panel empty
- Hit the endpoint directly and read off the top node id:
  ```bash
  curl -s "http://localhost:8000/api/gatekeepers?city=Bengaluru&input=occluded&model=baseline&top_k=5"
  ```
- If still empty, narrate the criticality coloring instead and skip to Step 8.

### Simulation returns nothing / errors
- Fall back to a raw call you can read aloud:
  ```bash
  curl -s -X POST http://localhost:8000/api/simulate \
    -H "Content-Type: application/json" \
    -d '{"city":"Bengaluru","model":"baseline","input":"occluded","disabledEdgeIds":[],"disabledNodeIds":["<gatekeeper-id>"]}'
  ```
- Point at `resilienceIndexAfter` and `avgTravelTimeIncreasePct` in the JSON.

### Criticality histogram missing
- There is **no backend histogram endpoint** — the frontend computes it
  client-side from `/api/network` `criticality` values. If the chart is blank,
  just point at the edge colors on the map; the distribution is the same data.

### Total meltdown (last resort)
- Have the Step-0 `curl` outputs **pre-saved to a text file** and screenshots of
  the fragmented vs. intact map ready. Walk the story from those.

---

## 3. Reset between runs
- Set toggles back to **Clean + Baseline** and clear any disabled nodes/edges
  (re-run simulate with empty `disabledNodeIds`/`disabledEdgeIds`, or refresh the
  page) so the next run starts from Resilience Index = 100.

---

## 4. 30-second elevator version (if time is cut)
1. Bengaluru → **Occluded + Baseline**: *"Network fragments, resilience 79."*
2. Switch to **Robust**: *"Recovered, resilience 100."*
3. Open top **Gatekeeper** → **disable it**: *"This one intersection drops
   resilience and breaks these routes."*
Done.
