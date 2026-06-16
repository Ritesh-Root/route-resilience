import { useEffect, useState } from "react";
import type { ComponentProps, ReactElement } from "react";

type MapViewProps = ComponentProps<typeof import("./MapView").MapView>;

/**
 * Leaflet touches `window` at import time. Lazy-load it only on the client
 * after hydration so SSR doesn't crash.
 */
export function ClientMap(props: MapViewProps) {
  const [Comp, setComp] = useState<
    null | ((p: MapViewProps) => ReactElement)
  >(null);

  useEffect(() => {
    let cancelled = false;
    import("./MapView").then((m) => {
      if (!cancelled) {
        setComp(() => m.MapView as unknown as (p: MapViewProps) => ReactElement);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // LOADING — Leaflet touches `window`, so we only know the component is ready
  // once the dynamic import resolves on the client.
  if (!Comp) {
    return (
      <div className="grid h-full w-full place-items-center bg-background/60">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="h-2 w-2 animate-pulse rounded-full bg-primary" />
          Initialising map…
        </div>
      </div>
    );
  }

  // EMPTY / ERROR — the real backend returns a GeoJSON FeatureCollection whose
  // `.features` drives the whole map. `MapView` reads `network.features.length`
  // unconditionally, so a missing or malformed payload (backend down, bad shape)
  // would crash Leaflet. Guard here at the client-only boundary instead.
  const features = props.network?.features;
  if (!Array.isArray(features)) {
    return (
      <div className="grid h-full w-full place-items-center bg-background/60">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="h-2 w-2 rounded-full bg-destructive" />
          Network data unavailable
        </div>
      </div>
    );
  }
  if (features.length === 0) {
    return (
      <div className="grid h-full w-full place-items-center bg-background/60">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="h-2 w-2 rounded-full bg-muted-foreground" />
          No road segments for this selection
        </div>
      </div>
    );
  }

  // DATA — hand the validated network + gatekeepers to the Leaflet layer, which
  // performs the [lng,lat]→[lat,lng] swap, criticality coloring, and gatekeeper
  // markers. Coordinates from the backend are [lng,lat]; Leaflet wants [lat,lng].
  return <Comp {...props} />;
}
