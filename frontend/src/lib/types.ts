export type SignalState = "red" | "orange" | "green" | "uncertain";

export interface FrameResult {
  index: number;
  t: number;
  state: SignalState;
  smoothed: number[];
  raw: number[];
  missing: string[];
  quality_ok: boolean;
  mask_png_b64: string;
}

export interface AnalysisResult {
  fps: number;
  n_frames: number;
  width: number;
  height: number;
  stride: number;
  processing_fps: number;
  frames: FrameResult[];
}

export interface Marker {
  t: number;
  kind: "acquired" | "lost" | "green";
  criterion?: string;
}
