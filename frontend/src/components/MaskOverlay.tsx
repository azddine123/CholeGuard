import { useEffect, useRef } from "react";
import { CLASS_COLORS } from "../lib/colors";

const cache = new Map<string, Promise<ImageData>>();

function decode(b64: string): Promise<ImageData> {
  if (!cache.has(b64)) {
    cache.set(b64, new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement("canvas");
        c.width = img.width; c.height = img.height;
        const ctx = c.getContext("2d")!;
        ctx.drawImage(img, 0, 0);
        resolve(ctx.getImageData(0, 0, c.width, c.height));
      };
      img.onerror = reject;
      img.src = `data:image/png;base64,${b64}`;
    }));
    if (cache.size > 400) cache.delete(cache.keys().next().value as string);
  }
  return cache.get(b64)!;
}

interface Props { maskB64?: string; visible: boolean[]; opacity: number }

export function MaskOverlay({ maskB64, visible, opacity }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    if (!maskB64) { ctx.clearRect(0, 0, canvas.width, canvas.height); return; }
    let cancelled = false;
    decode(maskB64).then((ids) => {
      if (cancelled) return;
      const small = new ImageData(ids.width, ids.height);
      for (let i = 0; i < ids.data.length; i += 4) {
        const k = ids.data[i];
        if (k > 0 && visible[k]) {
          const [r, g, b] = CLASS_COLORS[k];
          small.data[i] = r; small.data[i + 1] = g; small.data[i + 2] = b; small.data[i + 3] = Math.round(255 * opacity);
        }
      }
      const tmp = document.createElement("canvas");
      tmp.width = ids.width; tmp.height = ids.height;
      tmp.getContext("2d")!.putImageData(small, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
    });
    return () => { cancelled = true; };
  }, [maskB64, visible, opacity]);
  return <canvas ref={ref} width={854} height={480} className="pointer-events-none absolute inset-0 h-full w-full" />;
}
