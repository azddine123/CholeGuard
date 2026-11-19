import { signalView } from "../lib/signal";
import type { SignalState } from "../lib/types";

export function SignalLight({ state, missing }: { state: SignalState; missing: string[] }) {
  const v = signalView(state, missing);
  return (
    <div className="rounded-2xl border-2 p-5 transition-colors" style={{ borderColor: v.color, background: `${v.color}1f` }}>
      <div className="flex items-center gap-4">
        <div className="h-14 w-14 shrink-0 rounded-full shadow-lg" style={{ background: v.color, boxShadow: `0 0 28px ${v.color}` }} />
        <div>
          <div className="text-2xl font-bold text-white">{v.label}</div>
          <div className="mt-1 text-sm text-slate-300">{v.detail}</div>
        </div>
      </div>
    </div>
  );
}
