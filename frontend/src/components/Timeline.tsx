import { useMemo } from "react";
import { keyMoments, transitions } from "../lib/timeline";
import type { FrameResult } from "../lib/types";

const STATE_COLORS: Record<string, string> = { red: "#ef4444", orange: "#f59e0b", green: "#22c55e", uncertain: "#64748b" };

interface Props { frames: FrameResult[]; duration: number; current: number; onSeek: (t: number) => void }

export function Timeline({ frames, duration, current, onSeek }: Props) {
  const markers = useMemo(() => keyMoments(transitions(frames)), [frames]);
  const segments = useMemo(() => {
    const out: { start: number; end: number; state: string }[] = [];
    frames.forEach((f, i) => {
      const end = frames[i + 1]?.t ?? duration;
      const last = out[out.length - 1];
      if (last && last.state === f.state) last.end = end; else out.push({ start: f.t, end, state: f.state });
    });
    return out;
  }, [frames, duration]);
  const pct = (t: number) => `${(100 * t) / Math.max(duration, 1e-6)}%`;
  const fmt = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

  return (
    <div className="select-none">
      <div className="relative h-10 cursor-pointer overflow-hidden rounded-lg bg-slate-800"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          onSeek(((e.clientX - r.left) / r.width) * duration);
        }}>
        {segments.map((s, i) => (
          <div key={i} className="absolute top-0 h-full" style={{ left: pct(s.start), width: pct(s.end - s.start), background: STATE_COLORS[s.state] }} />
        ))}
        <div className="absolute top-0 h-full w-0.5 bg-white" style={{ left: pct(current) }} />
      </div>
      <div className="relative mt-1 h-7">
        {markers.map((m, i) => (
          <button key={i} title={`${fmt(m.t)} — ${m.kind === "green" ? "CVS détectée" : `${m.criterion} ${m.kind === "acquired" ? "acquis" : "perdu"}`}`}
            onClick={() => onSeek(m.t)}
            className="absolute -translate-x-1/2 rounded px-1 text-xs font-bold"
            style={{ left: pct(m.t), color: m.kind === "green" ? "#22c55e" : m.kind === "lost" ? "#ef4444" : "#e2e8f0" }}>
            {m.kind === "green" ? "★" : m.criterion}
          </button>
        ))}
      </div>
    </div>
  );
}
