// Warm criticality ramp: mint → marigold → coral.
// `c` is an edge criticality on the backend's 0..1 scale
// (RoadFeatureProperties.criticality from GET /api/network); out-of-range
// values are clamped below so callers never need to pre-normalize.
export function criticalityColor(c: number): string {
  const t = Math.max(0, Math.min(1, c));
  const stops = [
    { t: 0, rgb: [167, 215, 197] }, // #A7D7C5 mint
    { t: 0.5, rgb: [255, 217, 125] }, // #FFD97D marigold
    { t: 1, rgb: [255, 122, 78] }, // #FF7A4E coral
  ];
  let a = stops[0];
  let b = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i].t && t <= stops[i + 1].t) {
      a = stops[i];
      b = stops[i + 1];
      break;
    }
  }
  const span = b.t - a.t || 1;
  const lt = (t - a.t) / span;
  const r = Math.round(a.rgb[0] + (b.rgb[0] - a.rgb[0]) * lt);
  const g = Math.round(a.rgb[1] + (b.rgb[1] - a.rgb[1]) * lt);
  const bl = Math.round(a.rgb[2] + (b.rgb[2] - a.rgb[2]) * lt);
  return `rgb(${r}, ${g}, ${bl})`;
}

export function resilienceColor(score: number): string {
  // Higher score = more resilient = mint (low crit)
  return criticalityColor(1 - score / 100);
}
