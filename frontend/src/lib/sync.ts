import type { FrameResult } from "./types";

/** Last analysed frame at or before `t` (binary search). */
export function frameAt(frames: FrameResult[], t: number): FrameResult | undefined {
  if (!frames.length) return undefined;
  let lo = 0, hi = frames.length - 1;
  if (t <= frames[0].t) return frames[0];
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (frames[mid].t <= t) lo = mid; else hi = mid - 1;
  }
  return frames[lo];
}
