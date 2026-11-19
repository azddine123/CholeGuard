import { describe, expect, it } from "vitest";
import { frameAt } from "./sync";
import { signalView, CRITERIA_LABELS } from "./signal";
import { transitions } from "./timeline";
import type { FrameResult, SignalState } from "./types";

const f = (t: number, state: SignalState, missing: string[] = []): FrameResult => ({
  index: Math.round(t * 25), t, state, smoothed: [0, 0, 0], raw: [], missing, quality_ok: true, mask_png_b64: "",
});

describe("frameAt", () => {
  const frames = [f(0, "red"), f(0.08, "red"), f(0.16, "orange")];
  it("returns the first frame before the start", () => expect(frameAt(frames, -1)?.t).toBe(0));
  it("returns the last frame at or before t", () => expect(frameAt(frames, 0.1)?.t).toBe(0.08));
  it("clamps after the end", () => expect(frameAt(frames, 99)?.t).toBe(0.16));
  it("handles empty input", () => expect(frameAt([], 1)).toBeUndefined());
});

describe("signalView", () => {
  it("has the four exact labels", () => {
    expect(signalView("red", []).label).toBe("CVS non atteinte");
    expect(signalView("orange", ["C2"]).label).toBe("CVS partielle — manque : C2");
    expect(signalView("green", []).label).toBe("CVS détectée par le modèle");
    expect(signalView("uncertain", []).label).toBe("Incertain — image ou confiance insuffisante");
  });
  it("orange with nothing missing means confirmation in progress", () => {
    expect(signalView("orange", []).label).toBe("CVS en cours de confirmation");
  });
  it("labels all three criteria", () => expect(Object.keys(CRITERIA_LABELS)).toEqual(["C1", "C2", "C3"]));
});

describe("transitions", () => {
  it("marks acquisitions, losses and the first green", () => {
    const frames = [
      f(0, "red", ["C1", "C2", "C3"]),
      f(1, "orange", ["C2", "C3"]),
      f(2, "uncertain", []),            // ignored
      f(3, "orange", []),
      f(4, "green", []),
      f(5, "green", []),
      f(6, "orange", ["C3"]),
    ];
    const m = transitions(frames);
    expect(m.filter((x) => x.kind === "acquired").map((x) => [x.t, x.criterion])).toEqual([
      [1, "C1"], [3, "C2"], [3, "C3"],
    ]);
    expect(m.filter((x) => x.kind === "green").map((x) => x.t)).toEqual([4]);
    expect(m.filter((x) => x.kind === "lost").map((x) => [x.t, x.criterion])).toEqual([[6, "C3"]]);
  });
});
