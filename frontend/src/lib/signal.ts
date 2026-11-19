import type { SignalState } from "./types";

export const CRITERIA_LABELS: Record<string, string> = {
  C1: "Triangle hépatocystique dégagé",
  C2: "Tiers inférieur de la vésicule séparé du lit hépatique",
  C3: "Deux structures seulement entrent dans la vésicule",
};

export interface SignalView { color: string; label: string; detail: string; emoji: string }

export function signalView(state: SignalState, missing: string[]): SignalView {
  switch (state) {
    case "red":
      return { color: "#ef4444", emoji: "🔴", label: "CVS non atteinte", detail: "Aucun critère de sécurité n'est encore visible." };
    case "orange":
      return missing.length
        ? { color: "#f59e0b", emoji: "🟠", label: `CVS partielle — manque : ${missing.join(", ")}`,
            detail: missing.map((c) => `${c} : ${CRITERIA_LABELS[c]}`).join(" · ") }
        : { color: "#f59e0b", emoji: "🟠", label: "CVS en cours de confirmation",
            detail: "Les 3 critères sont visibles ; le signal attend qu'ils restent stables." };
    case "green":
      return { color: "#22c55e", emoji: "🟢", label: "CVS détectée par le modèle",
               detail: "Indication algorithmique expérimentale — la décision reste celle du chirurgien." };
    default:
      return { color: "#94a3b8", emoji: "⚪", label: "Incertain — image ou confiance insuffisante",
               detail: "Image floue, enfumée, obstruée, ou modèle hésitant." };
  }
}
