import { useEffect, useState } from "react";

type Metrics = { cvs?: Record<string, { mean_ap: number; mean_bacc: number }>; signal?: { test?: { false_green_rate: number; video_sensitivity: number } }; segmentation?: { test_dice_fg: number } };

export function About({ onClose }: { onClose: () => void }) {
  const [m, setM] = useState<Metrics | null>(null);
  useEffect(() => { fetch("/metrics.json").then((r) => (r.ok ? r.json() : null)).then(setM).catch(() => setM(null)); }, []);
  const pct = (x?: number) => (x === undefined || x === null || Number.isNaN(x) ? "—" : `${(100 * x).toFixed(1)} %`);
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6" onClick={onClose}>
      <div className="max-h-full max-w-3xl overflow-auto rounded-2xl bg-slate-900 p-8 text-slate-200" onClick={(e) => e.stopPropagation()}>
        <h2 className="text-2xl font-bold text-white">À propos de CholeGuard</h2>
        <p className="mt-3">Copilote expérimental d'aide à l'évaluation de la <b>vue critique de sécurité (CVS)</b> pendant une cholécystectomie cœlioscopique. Le modèle segmente l'anatomie, puis évalue les 3 critères CVS <b>à partir des structures qu'il a reconnues</b>.</p>
        <h3 className="mt-6 text-lg font-semibold text-white">Résultats (jeu de test Endoscapes, jamais vu à l'entraînement)</h3>
        {m?.cvs ? (
          <table className="mt-2 w-full text-left text-sm">
            <thead className="text-slate-400"><tr><th>Variante</th><th>mAP</th><th>Précision équilibrée</th></tr></thead>
            <tbody>
              {[["image", "A — image seule"], ["nomask", "B sans masques"], ["masks", "B guidé par l'anatomie"]].map(([k, l]) => (
                <tr key={k}><td>{l}</td><td>{pct(m.cvs?.[k]?.mean_ap)}</td><td>{pct(m.cvs?.[k]?.mean_bacc)}</td></tr>
              ))}
            </tbody>
          </table>
        ) : <p className="mt-2 text-slate-400">Résultats disponibles après l'entraînement.</p>}
        {m?.signal?.test && <p className="mt-3 text-sm">Faux feux verts : <b>{pct(m.signal.test.false_green_rate)}</b> · Vidéos où la CVS est détectée : <b>{pct(m.signal.test.video_sensitivity)}</b> · Dice segmentation : <b>{pct(m.segmentation?.test_dice_fg)}</b></p>}
        <h3 className="mt-6 text-lg font-semibold text-white">Limites</h3>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">
          <li>Outil de recherche, <b>non certifié</b> : il ne remplace jamais le jugement du chirurgien.</li>
          <li>Entraîné sur 201 vidéos d'un seul centre (Endoscapes2023) ; performances inconnues sur d'autres équipements.</li>
          <li>Les experts eux-mêmes ne sont que modérément d'accord sur la CVS (kappa de Fleiss 0,28 à 0,62).</li>
          <li>Ne pas interpréter en cas d'anatomie inhabituelle, de saignement important ou d'image dégradée (état ⚪).</li>
        </ul>
        <p className="mt-6 text-xs text-slate-500">Données : Endoscapes2023 (CAMMA), CholecSeg8k — licence CC BY-NC-SA 4.0, usage académique non commercial.</p>
        <button onClick={onClose} className="mt-6 rounded-lg bg-cyan-600 px-4 py-2 font-semibold text-white hover:bg-cyan-500">Fermer</button>
      </div>
    </div>
  );
}
