import { CLASS_COLORS, CLASS_NAMES } from "../lib/colors";

interface Props { visible: boolean[]; setVisible: (v: boolean[]) => void; opacity: number; setOpacity: (o: number) => void }

export function MaskToggles({ visible, setVisible, opacity, setOpacity }: Props) {
  return (
    <div className="rounded-xl bg-slate-900 p-3">
      <div className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-400">Structures affichées</div>
      <div className="grid grid-cols-2 gap-1.5">
        {CLASS_NAMES.map((name, k) => k === 0 ? null : (
          <label key={k} className="flex cursor-pointer items-center gap-2 text-sm text-slate-200">
            <input type="checkbox" checked={visible[k]} onChange={() => setVisible(visible.map((v, i) => (i === k ? !v : v)))} />
            <span className="h-3 w-3 rounded-sm" style={{ background: `rgb(${CLASS_COLORS[k].join(",")})` }} />
            {name}
          </label>
        ))}
      </div>
      <label className="mt-3 flex items-center gap-2 text-sm text-slate-300">
        Opacité
        <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(+e.target.value)} className="flex-1" />
      </label>
    </div>
  );
}
