# Problem Statement Feasibility Analysis (Hackathon Sprint)

**Context:** ISRO Bhartiya Antariksh Hackathon–style problem statements. Team is strong across deep learning/CV, classical ML + signal processing, remote sensing/GIS, and full-stack/systems. Goal = working MVP in **days to weeks**, so the ranking below weights *ready-to-use data* and *forkable code* heavily over raw novelty.

**Scoring criteria (each 1–5):**
- **Data** — can you get usable training/eval data today, free?
- **Repo** — is there a strong open-source repo / pretrained model to fork?
- **Maturity** — how solved is the core ML technique?
- **Demo** — how easy to show something impressive to judges?

---

## Tier 1 — Best hackathon bets (fork-and-adapt, strong demo)

### 4. Route Resilience: Road Extraction + Graph-Theoretic Criticality ⭐ TOP PICK
Data 5 · Repo 5 · Maturity 5 · Demo 5
- **Why it wins:** Two halves are both basically solved separately, and one repo already does both. Extract roads with a segmentation CNN, vectorize to a graph, then run criticality analysis (betweenness centrality, edge-removal impact) — a clean, judge-friendly story with map visuals.
- **Data:** DeepGlobe Road Extraction + SpaceNet (free); OpenStreetMap for validation.
- **Repos to fork:**
  - `avanetten/cresi` — satellite → **road network graph with travel-time** (this is the criticality backbone)
  - `zlckanata/DeepGlobe-Road-Extraction-Challenge` — D-LinkNet, 1st-place DeepGlobe solution
  - `satellite-image-deep-learning/techniques` — meta-index of alternatives
- **Best/efficient tech:** D-LinkNet (ResNet34 encoder) for extraction → `NetworkX` for graph metrics. The "occlusion-robust" angle = train with simulated occlusions/augmentation. Light on compute.

### 7. AI Detection of Exoplanets from Noisy Light Curves ⭐ TOP PICK (signal team)
Data 5 · Repo 5 · Maturity 5 · Demo 4
- **Why it wins:** Cleanest dataset of the lot, the canonical model is open-sourced by Google, trains fast on 1-D signals (no heavy GPU). Perfect for the classical-ML/signal-processing strength.
- **Data:** Kepler/TESS light curves via MAST + Kaggle "Kepler labelled time series"; labels included.
- **Repos to fork:**
  - `google-research/exoplanet-ml` — original Shallue & Vanderburg AstroNet
  - `yuliang419/Astronet-Triage` and `Astronet-Vetting` — TESS versions
- **Best/efficient tech:** 1-D CNN on global+local phase-folded views (AstroNet). Add a denoising step (Savitzky–Golay / wavelet) to directly address the "noisy" framing. Differentiator = robustness to noise + explainability.

### 3. Surface AQI & HCHO Hotspot Detection over India ⭐ TOP PICK (remote sensing)
Data 5 · Repo 4 · Maturity 4 · Demo 4
- **Why it wins:** All inputs are free and already cloud-hosted; core method is tabular regression (fast). India-specific and policy-relevant — strong narrative.
- **Data:** Sentinel-5P TROPOMI HCHO on Google Earth Engine (`COPERNICUS_S5P_OFFL_L3_HCHO`); ground-truth AQI from CPCB stations; meteorology from ERA5.
- **Repos/patterns:** GEE Python API workflows (see Nature *Sci. Rep.* country-level S5P + GEE pipeline); no single must-fork repo — you assemble it, which is quick.
- **Best/efficient tech:** XGBoost/Random Forest to regress surface AQI from S5P columns + met data; Getis-Ord Gi* / DBSCAN for HCHO hotspot clustering. Demo = interactive hotspot map.

### 10. Infrared Image Colorization & Enhancement
Data 4 · Repo 5 · Maturity 4 · Demo 5
- **Why it wins:** Pretrained GANs exist; the output is inherently visual = great demo. Low data-engineering burden.
- **Data:** KAIST Multispectral & FLIR ADAS thermal–RGB pairs (free).
- **Repos to fork:** `FuyaLuo/MornGAN`, StawGAN, `cyanymore/Awesome-Image-Colorization` (index); baseline = `pix2pix`.
- **Best/efficient tech:** Conditional GAN (pix2pix) baseline → swap to a nighttime-TIR model (MornGAN) for the "enhanced object interpretation" angle. Fine-tune a pretrained checkpoint, don't train from scratch.

### 12. Temporal Frame Interpolation of Satellite Imagery (Optical Flow)
Data 4 · Repo 5 · Maturity 5 · Demo 5
- **Why it wins:** State-of-the-art interpolators are open and pretrained; you adapt them to satellite cadence. There is even a paper specifically on geostationary-imagery interpolation to anchor your approach.
- **Data:** INSAT/geostationary frames or any Sentinel/MODIS time series; abundant.
- **Repos to fork:** `hzwer/Practical-RIFE` (real-time, multi-frame), Google FILM (`frame-interpolation`), `lyh-18/Video-Frame-Interpolation-Collections`.
- **Best/efficient tech:** RIFE (fast, light) fine-tuned on satellite pairs; reference *"Temporal Interpolation of Geostationary Satellite Imagery with Task-Specific Optical Flow."* Demo = smooth interpolated cloud/weather animation.

---

## Tier 2 — Strong, but need scoping or more data wrangling

### 1. Urban Heat Mitigation & Cooling Strategies (AI/ML)
Data 5 · Repo 4 · Maturity 4 · Demo 4
- Mapping the heat island is easy and well-supported; the **"mitigation/cooling strategy"** layer is the differentiator and needs framing (e.g., recommend where to add green/cool roofs and simulate ΔLST).
- **Data:** Landsat 8/MODIS LST + NDVI/NDBI on GEE. **Repos:** `KonlavachMengsuwan/Urban-Heat-Island-Mapping-with-Machine-Learning-using-GEE`, `wrtcd/US-Cities-UHI-Analysis`.
- **Tech:** Random Forest/XGBoost LST model + a simple optimization/scenario layer for interventions.

### 2. Generative AI Cloud Removal for LISS-IV Imagery
Data 3 · Repo 5 · Maturity 4 · Demo 5
- Generative cloud removal is well-served; risk is **LISS-IV-specific data** (via ISRO Bhoonidhi) — prototype on Sentinel-2 first, then adapt.
- **Repos:** `Shuaizhang7/AttentionGAN-for-Cloud-removal`, `alessandrosebastianelli/PLFM-Clouds-Removal`, DiffCR (diffusion, arXiv 2308.04417).
- **Tech:** SAR-optical fusion GAN or a conditional diffusion model (DiffCR) if GPU allows.

### 15. Solar Flare Forecasting/Nowcasting (Aditya-L1 soft+hard X-ray)
Data 4 · Repo 3 · Maturity 4 · Demo 3
- Prototype on decades of **GOES X-ray flux** (free, SWPC), then layer in Aditya-L1 SoLEXS/HEL1OS data from ISSDC. Good fit for signal team.
- **Repos:** `hayesla/flare_forecast_proj`; SunPy ecosystem. **Tech:** LSTM/Temporal CNN/transformer on X-ray time series; frame as binary "flare in next N hours."

### 14. Forecasting Energetic Particle Radiation for GEO Satellites
Data 4 · Repo 2 · Maturity 3 · Demo 3
- Data is available (NOAA GOES SEM particle flux, ACE, OMNI); fewer turnkey repos so you build the pipeline. Time-series forecasting; niche but tractable and ISRO-aligned.
- **Tech:** Gradient boosting / LSTM multi-horizon forecast of >2 MeV electron flux.

### 6. Crop Type + Moisture Stress + Irrigation Advisory (Optical + Microwave)
Data 5 · Repo 4 · Maturity 4 · Demo 3
- All inputs free on GEE; risk is **scope creep** — the statement bundles three tasks. For a sprint, nail crop-type classification + an NDVI/soil-moisture stress proxy; treat irrigation advisory as a rules layer.
- **Repos:** `ellaampy/CropTypeMapping` (Sentinel-1+2 fusion), `flyakon/H2Crop`. **Tech:** PSE+TAE or LSTM on S1/S2 time series.

### 11. Cross-Modal Satellite Image Retrieval (Multi-Sensor)
Data 4 · Repo 4 · Maturity 3 · Demo 3
- Contrastive cross-modal learning is established; retrieval demos are less flashy and pretraining wants some GPU (use existing foundation weights).
- **Data/repos:** QXS-SAROPT dataset; `jaychempan/Awesome-RSITR`; SSL4EO-S12 via `Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models`; MCRN. **Tech:** Dual-encoder CLIP-style contrastive model; fine-tune a pretrained RS foundation model.

---

## Tier 3 — High novelty / impact, but hard to finish well in a sprint

### 13. Air-Gapped Predictive Copilot for Secure MPLS Operations
Data 2 · Repo 2 · Maturity 3 · Demo 4
- Best fit for the **systems/networking** strength and very demo-able (a chat copilot), but there's no off-the-shelf dataset — you'd generate synthetic MPLS telemetry and build most of it.
- **Tech:** Local LLM via **Ollama** (Llama 3 / Mistral) + **LlamaIndex/LangChain** offline RAG over network runbooks + an anomaly-detection model on telemetry. Fully offline = the whole point.

### 9. Wavefront Reconstruction & Turbulence from SH-WFS Time-Series
Data 3 (simulated) · Repo 2 · Maturity 3 · Demo 2
- Specialized adaptive optics. You can **simulate** SH-WFS data, so data isn't blocking, but fewer forkable repos and lower visual payoff.
- **Tech:** Python AO libs `HCIPy` / `Soapy` / `aotools` to generate spots; CNN or classical modal reconstruction; ML to estimate turbulence (Cn², r0).

### 8. Subsurface Ice in Lunar South Pole (Chandrayaan-2 DFSAR)
Data 2 · Repo 1 · Maturity 2 · Demo 3
- Extremely ISRO-aligned and topical (ISRO published fresh results May 2026), but **PolSAR data access + processing is a steep curve** and labeled data is scarce — risky for a short sprint.
- **Data:** DFSAR L-band via PRADAN/ISSDC (`pradan.issdc.gov.in/ch2`). **Method:** CPR > 1 and Degree-of-Polarization < 0.13 indicators. **Tools:** ESA SNAP / PolSARpro + a classifier on PolSAR features.

### 5. AI-Powered Digital Twin of India's Climate
Data 3 · Repo 1 · Maturity 2 · Demo 2
- Huge, vaguely scoped, integration-heavy. Hardest to compress into a finished demo. High prestige, low sprint-feasibility — avoid unless you radically narrow it (e.g., a single-variable regional now-cast "twin").

---

## Bottom line

**Pick from Tier 1.** Given your cross-disciplinary team and a sprint timeline, the three best risk-adjusted bets are:

1. **#4 Route Resilience** — clearest path to a complete, impressive build; `cresi` + `NetworkX` get you most of the way.
2. **#7 Exoplanet detection** — fastest to a working model; clean data, Google's repo, runs on a laptop.
3. **#3 AQI/HCHO over India** — strongest "real-world Indian impact" story; all data free on GEE.

If you want a flashy visual demo specifically, add **#12 (frame interpolation)** or **#10 (IR colorization)** — both are fork-a-pretrained-model-and-fine-tune, so very fast to a wow moment.
