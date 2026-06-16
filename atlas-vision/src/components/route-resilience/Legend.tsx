export function Legend() {
  return (
    <div
      className="pointer-events-auto flex items-center gap-3 rounded-2xl border border-white/60 bg-white/95 px-4 py-3 shadow-[0_8px_24px_-8px_rgba(31,27,22,0.18)] backdrop-blur"
      role="img"
      aria-label="Criticality legend low to high"
    >
      <div className="flex flex-col">
        <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
          Criticality
        </p>
        <p className="text-[10px] text-muted-foreground/70">Low → High</p>
      </div>
      {/* Gradient stops mirror criticalityColor() in lib/criticality.ts
          (mint #A7D7C5 @0 → marigold #FFD97D @0.5 → coral #FF7A4E @1) so this
          swatch stays in lockstep with the MapView edge heat scale. The 0 / 1
          labels surface the backend's RoadFeatureProperties.criticality range. */}
      <div className="flex flex-col gap-0.5">
        <div
          className="h-3 w-32 rounded-full border border-card-hairline"
          style={{
            background:
              "linear-gradient(to right, #A7D7C5 0%, #FFD97D 50%, #FF7A4E 100%)",
          }}
        />
        <div className="flex w-32 justify-between text-[9px] tabular-nums text-muted-foreground/70">
          <span>0</span>
          <span>1</span>
        </div>
      </div>
      <span className="text-[10px] font-bold text-coral">CRITICAL</span>
    </div>
  );
}
