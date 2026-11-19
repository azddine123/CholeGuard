import type { FrameResult, Marker } from "./types";

const CRITERIA = ["C1", "C2", "C3"];

/** Moments where a criterion is acquired / lost, and the first GREEN. Uncertain frames are skipped. */
export function transitions(frames: FrameResult[]): Marker[] {
  const out: Marker[] = [];
  let prev: Record<string, boolean> | null = null;
  let wasGreen = false;
  for (const fr of frames) {
    if (fr.state === "uncertain") continue;
    const cur = Object.fromEntries(CRITERIA.map((c) => [c, !fr.missing.includes(c)]));
    if (prev) {
      for (const c of CRITERIA) {
        if (cur[c] && !prev[c]) out.push({ t: fr.t, kind: "acquired", criterion: c });
        if (!cur[c] && prev[c]) out.push({ t: fr.t, kind: "lost", criterion: c });
      }
    }
    if (fr.state === "green" && !wasGreen) out.push({ t: fr.t, kind: "green" });
    wasGreen = fr.state === "green";
    prev = cur;
  }
  return out;
}
