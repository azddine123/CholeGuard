import { CRITERIA_LABELS } from "../lib/signal";
import type { FrameResult } from "../lib/types";

export function CriteriaPanel({ frame }: { frame?: FrameResult }) {
  return (
    <div className="space-y-3">
      {["C1", "C2", "C3"].map((c, j) => {
        const p = frame?.smoothed?.[j];
        const known = p !== undefined && p !== null && Number.isFinite(p);
        const uncertain = !frame || frame.state === "uncertain";
        const ok = !uncertain && known && !frame!.missing.includes(c);
        const pct = known ? Math.round((p as number) * 100) : 0;
        return (
          <div key={c} className="rounded-xl bg-slate-900 p-3">
            <div className="flex items-center justify-between">
              <span className="text-lg font-semibold text-white">{ok ? "✅" : uncertain ? "❔" : "⬜"} {c}</span>
              <span className="font-mono text-lg text-slate-200">{known ? `${pct} %` : "—"}</span>
            </div>
            <div className="text-sm text-slate-400">{CRITERIA_LABELS[c]}</div>
            <div className="mt-2 h-2 rounded bg-slate-800">
              <div className="h-2 rounded transition-all" style={{ width: `${pct}%`, background: ok ? "#22c55e" : uncertain ? "#64748b" : "#f59e0b" }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}
