import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toPng } from "html-to-image";
import { api } from "./api";
import { About } from "./components/About";
import { CriteriaPanel } from "./components/CriteriaPanel";
import { MaskOverlay } from "./components/MaskOverlay";
import { MaskToggles } from "./components/MaskToggles";
import { SignalLight } from "./components/SignalLight";
import { Timeline } from "./components/Timeline";
import { frameAt } from "./lib/sync";
import { keyMoments, transitions } from "./lib/timeline";
import type { AnalysisResult } from "./lib/types";

const SPEEDS = [0.25, 0.5, 1, 2, 5];

export default function App() {
  const video = useRef<HTMLVideoElement>(null);
  const report = useRef<HTMLDivElement>(null);
  const [samples, setSamples] = useState<string[]>([]);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [src, setSrc] = useState<string>("");
  const [title, setTitle] = useState("");
  const [t, setT] = useState(0);
  const [duration, setDuration] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [visible, setVisible] = useState([false, true, true, true, true, true, true]);
  const [opacity, setOpacity] = useState(0.45);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [about, setAbout] = useState(false);

  useEffect(() => { api.samples().then((s) => setSamples(s.map((x) => x.name))).catch(() => setSamples([])); }, []);

  useEffect(() => {
    let raf = 0;
    const tick = () => { if (video.current) setT(video.current.currentTime); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  useEffect(() => { if (video.current) video.current.playbackRate = speed; }, [speed, src]);

  const frame = useMemo(() => (result ? frameAt(result.frames, t) : undefined), [result, t]);
  const markers = useMemo(() => (result ? keyMoments(transitions(result.frames)) : []), [result]);

  const loadSample = async (name: string) => {
    setError(""); setResult(null);
    try {
      setResult(await api.sample(name));
      setSrc(api.sampleVideoUrl(name)); setTitle(`Exemple : ${name}`);
    } catch (e) { setError((e as Error).message); }
  };

  const upload = async (file: File) => {
    setError(""); setResult(null); setProgress(0);
    setSrc(URL.createObjectURL(file)); setTitle(file.name);
    try {
      const { job_id } = await api.analyze(file);
      for (;;) {
        const s = await api.job(job_id);
        setProgress(s.progress);
        if (s.status === "done") break;
        if (s.status === "error") throw new Error(s.error || "Échec de l'analyse");
        await new Promise((r) => setTimeout(r, 500));
      }
      setResult(await api.result(job_id));
    } catch (e) { setError((e as Error).message); } finally { setProgress(null); }
  };

  const seek = useCallback((time: number) => { if (video.current) video.current.currentTime = Math.max(0, time); }, []);

  const exportReport = async () => {
    if (!report.current) return;
    const url = await toPng(report.current, { backgroundColor: "#0f172a", pixelRatio: 2 });
    const a = document.createElement("a");
    a.href = url; a.download = `choleguard-rapport-${Date.now()}.png`; a.click();
  };

  const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  return (
    <div className="min-h-full bg-slate-950 text-slate-100">
      <header className="flex items-center justify-between border-b border-slate-800 px-6 py-3">
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight text-white">Chole<span className="text-cyan-400">Guard</span></h1>
          <p className="text-sm text-slate-400">Copilote de vue critique de sécurité — cholécystectomie cœlioscopique</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="rounded-full border border-amber-500/50 bg-amber-500/10 px-3 py-1 text-xs text-amber-300">Outil de recherche — non certifié pour usage clinique</span>
          <button onClick={() => setAbout(true)} className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm hover:bg-slate-800">À propos & résultats</button>
        </div>
      </header>

      <div className="flex flex-wrap items-center gap-3 px-6 py-3">
        <label className="cursor-pointer rounded-lg bg-cyan-600 px-4 py-2 font-semibold text-white hover:bg-cyan-500">
          Analyser une vidéo
          <input type="file" accept="video/*" className="hidden" onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
        </label>
        {samples.length > 0 && (
          <select className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-2" value="" onChange={(e) => e.target.value && loadSample(e.target.value)}>
            <option value="">Vidéos d'exemple (hors ligne)…</option>
            {samples.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        )}
        {progress !== null && (
          <div className="flex min-w-64 items-center gap-2">
            <div className="h-2 flex-1 rounded bg-slate-800"><div className="h-2 rounded bg-cyan-500 transition-all" style={{ width: `${Math.round(progress * 100)}%` }} /></div>
            <span className="text-sm text-slate-300">Analyse… {Math.round(progress * 100)} %</span>
          </div>
        )}
        {result && <span className="text-sm text-slate-400">{result.frames.length} images analysées · {result.processing_fps.toFixed(1)} img/s · {result.fps.toFixed(0)} FPS vidéo</span>}
        {error && <span className="rounded bg-red-500/15 px-3 py-1 text-sm text-red-300">⚠ {error}</span>}
      </div>

      <main className="grid gap-6 px-6 pb-6 lg:grid-cols-[1fr_400px]">
        <section>
          <div className="relative overflow-hidden rounded-2xl bg-black" style={{ aspectRatio: "16 / 9" }}>
            {src ? (
              <>
                <video ref={video} src={src} className="h-full w-full object-contain" controls={false}
                  onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)} onClick={(e) => (e.currentTarget.paused ? e.currentTarget.play() : e.currentTarget.pause())} />
                <MaskOverlay maskB64={frame?.mask_png_b64} visible={visible} opacity={opacity} />
              </>
            ) : (
              <div className="flex h-full items-center justify-center text-lg text-slate-500">Choisissez une vidéo d'exemple ou analysez votre propre vidéo</div>
            )}
          </div>
          {src && (
            <div className="mt-3 flex items-center gap-3">
              <button onClick={() => (video.current?.paused ? video.current.play() : video.current?.pause())} className="rounded-lg bg-slate-800 px-4 py-2 font-semibold hover:bg-slate-700">▶ ❚❚  Lecture / pause</button>
              {SPEEDS.map((s) => (
                <button key={s} onClick={() => setSpeed(s)} className={`rounded-lg px-3 py-2 text-sm ${speed === s ? "bg-cyan-600 text-white" : "bg-slate-800 hover:bg-slate-700"}`}>{s}×</button>
              ))}
              <span className="ml-auto font-mono text-slate-300">{fmt(t)} / {fmt(duration)}</span>
            </div>
          )}
          {result && (
            <div ref={report} className="mt-4 rounded-2xl bg-slate-900 p-4">
              <div className="mb-2 flex items-center justify-between">
                <h2 className="font-semibold text-white">Chronologie — {title}</h2>
                <button onClick={exportReport} className="rounded-lg border border-slate-700 px-3 py-1 text-sm hover:bg-slate-800">Exporter le rapport (PNG)</button>
              </div>
              <Timeline frames={result.frames} duration={duration || result.n_frames / result.fps} current={t} onSeek={seek} />
              <div className="mt-2 flex flex-wrap gap-2 text-sm text-slate-300">
                {markers.length === 0 ? <span className="text-slate-500">Aucun critère acquis pendant la vidéo.</span> :
                  markers.map((m, i) => (
                    <button key={i} onClick={() => seek(m.t)} className="rounded bg-slate-800 px-2 py-0.5 hover:bg-slate-700">
                      {fmt(m.t)} · {m.kind === "green" ? "🟢 CVS détectée" : `${m.criterion} ${m.kind === "acquired" ? "acquis" : "perdu"}`}
                    </button>
                  ))}
              </div>
            </div>
          )}
        </section>

        <aside className="space-y-4">
          <SignalLight state={frame?.state ?? "uncertain"} missing={frame?.missing ?? []} />
          <CriteriaPanel frame={frame} />
          <MaskToggles visible={visible} setVisible={setVisible} opacity={opacity} setOpacity={setOpacity} />
        </aside>
      </main>
      {about && <About onClose={() => setAbout(false)} />}
    </div>
  );
}
