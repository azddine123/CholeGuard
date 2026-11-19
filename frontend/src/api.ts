import type { AnalysisResult } from "./lib/types";

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = `Erreur ${r.status}`;
    try { msg = (await r.json()).detail ?? msg; } catch { /* keep default */ }
    throw new Error(msg);
  }
  return r.json();
}

export const api = {
  samples: () => fetch("/api/samples").then((r) => json<{ name: string }[]>(r)),
  sample: (name: string) => fetch(`/api/samples/${name}`).then((r) => json<AnalysisResult>(r)),
  sampleVideoUrl: (name: string) => `/api/samples/${name}/video`,
  analyze: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return fetch("/api/analyze", { method: "POST", body: fd }).then((r) => json<{ job_id: string }>(r));
  },
  job: (id: string) => fetch(`/api/jobs/${id}`).then((r) =>
    json<{ status: string; progress: number; error?: string }>(r)),
  result: (id: string) => fetch(`/api/jobs/${id}/result`).then((r) => json<AnalysisResult>(r)),
};
