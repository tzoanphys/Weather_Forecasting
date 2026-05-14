import "./style.css";

function getAppRoot(): HTMLElement {
  const el = document.getElementById("app");
  if (!el) throw new Error("Missing #app");
  return el;
}

const appRoot = getAppRoot();

type TrainingHistory = {
  epochs: number[];
  train: { mse: number[]; mae: number[] };
  val: { mse: number[]; mae: number[] };
  ratio_val_to_train_mse: number[];
  best_epoch: number | null;
  best_val_mse: number | null;
};

type EvalSummary = {
  raw_mse?: number;
  raw_mae?: number;
  corrected_mse?: number;
  corrected_mae?: number;
  validation_samples?: number;
  plotted_sample_index?: number;
  final_plot?: string;
};

const fmt = (x: unknown, d = 4): string => {
  const n = typeof x === "number" ? x : Number(x);
  return Number.isFinite(n) ? n.toFixed(d) : "—";
};

function last<T>(arr: T[] | undefined): T | undefined {
  if (!arr?.length) return undefined;
  return arr[arr.length - 1];
}

function overfittingVerdict(h: TrainingHistory | null): { text: string; tone: "good" | "warn" | "bad" | "muted" } {
  if (!h?.train?.mse?.length || !h.val?.mse?.length) {
    return { text: "Overfitting: unknown (no training export).", tone: "muted" };
  }
  if (h.train.mse.length < 8) {
    return { text: "Overfitting: unknown (need more epochs in export).", tone: "muted" };
  }
  const ratio = last(h.ratio_val_to_train_mse) ?? NaN;
  const t = h.train.mse;
  const v = h.val.mse;
  const t0 = t[Math.max(0, t.length - 6)];
  const t1 = last(t)!;
  const v0 = v[Math.max(0, v.length - 6)];
  const v1 = last(v)!;
  const diverge = t1 < t0 * 0.98 && v1 > v0 * 1.03;
  if (Number.isFinite(ratio) && ratio >= 1.35 && diverge) {
    return { text: `Overfitting observed: likely (val/train MSE ≈ ${fmt(ratio)}).`, tone: "bad" };
  }
  if (Number.isFinite(ratio) && ratio >= 1.35) {
    return { text: `Overfitting observed: possible (val/train MSE ≈ ${fmt(ratio)}).`, tone: "warn" };
  }
  if (diverge) {
    return { text: `Overfitting observed: mild (val rising, train falling in last epochs).`, tone: "warn" };
  }
  return { text: `Overfitting observed: no strong signal (val/train MSE ≈ ${fmt(ratio)}).`, tone: "good" };
}

function variablesHtml(h: TrainingHistory | null, ev: EvalSummary | null): string {
  const rows: [string, string][] = [];
  if (h?.epochs?.length) {
    const n = h.epochs.length;
    rows.push(["Epoch (last)", String(h.epochs[n - 1])]);
    rows.push(["Train MSE", fmt(last(h.train.mse))]);
    rows.push(["Val MSE", fmt(last(h.val.mse))]);
    rows.push(["Train MAE", fmt(last(h.train.mae))]);
    rows.push(["Val MAE", fmt(last(h.val.mae))]);
    if (h.best_epoch != null) rows.push(["Best val epoch", String(h.best_epoch)]);
  }
  if (ev?.raw_mse != null) {
    rows.push(["Eval raw MSE", fmt(ev.raw_mse)]);
    rows.push(["Eval raw MAE", fmt(ev.raw_mae)]);
    rows.push(["Eval corr. MSE", fmt(ev.corrected_mse)]);
    rows.push(["Eval corr. MAE", fmt(ev.corrected_mae)]);
    rows.push(["Val tiles", String(ev.validation_samples ?? "—")]);
  }
  if (!rows.length) return '<p class="hint">Press Run to train + evaluate, then values appear here.</p>';
  return `<dl class="var-grid">${rows.map(([k, v]) => `<div class="var-row"><dt>${k}</dt><dd>${v}</dd></div>`).join("")}</dl>`;
}

function parseJson<T>(text: string): T | null {
  const t = text.trim();
  if (!t) return null;
  try {
    return JSON.parse(t) as T;
  } catch {
    return null;
  }
}

async function loadJson<T>(url: string): Promise<T | null> {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) return null;
  return parseJson<T>(await res.text());
}

async function runPipeline(): Promise<void> {
  const res = await fetch("/api/run?task=train%2Bevaluate", { method: "POST" });
  const text = await res.text();
  const body = parseJson<{ error?: string }>(text) ?? {};
  if (!res.ok) throw new Error(body.error ?? `HTTP ${res.status}`);
}

function plotSrc(ev: EvalSummary | null): string {
  const name = ev?.final_plot?.replace(/\\/g, "/").split("/").pop()?.trim();
  return `/ml/${encodeURIComponent(name || "final_evaluation_plot.png")}`;
}

function mount(): void {
  appRoot.innerHTML = `
    <div class="wrap">
      <header class="hero">
        <p class="hero__kicker">Belgium · gridded 10&nbsp;m wind</p>
        <h1 class="hero__title">
          <span class="hero__title-flair">Belgian winds</span>
          <span class="hero__title-sub">ML model dashboard</span>
        </h1>
        <p class="hero__lead">
          The data are <strong>Belgian</strong> wind fields on a lat–lon grid over the country; training and evaluation run on your machine and refresh the metrics, overfitting readout, and map-style plot you see in the card.
          Press <strong>Run</strong> whenever you want another full train/eval cycle so the chart and numbers below match the latest export.
        </p>
      </header>
      <main class="panel">
        <button type="button" id="run" class="run">Run</button>
        <p id="note" class="note"></p>
        <section id="vars" class="vars"></section>
        <p id="fit" class="fit" data-tone="muted"></p>
        <figure class="plot" id="plotBox" hidden>
          <img id="chart" alt="Evaluation plot" />
          <figcaption id="cap" class="cap" hidden>Could not load plot image.</figcaption>
        </figure>
      </main>
    </div>
  `;
}

function clear(): void {
  document.getElementById("vars")!.innerHTML = "";
  const fit = document.getElementById("fit")!;
  fit.textContent = "";
  fit.dataset.tone = "muted";
  const plotBox = document.getElementById("plotBox")!;
  plotBox.hidden = true;
  const img = document.getElementById("chart") as HTMLImageElement;
  img.removeAttribute("src");
  img.style.display = "";
  document.getElementById("cap")!.hidden = true;
}

async function show(): Promise<void> {
  const note = document.getElementById("note")!;
  const vars = document.getElementById("vars")!;
  const fit = document.getElementById("fit")!;
  const img = document.getElementById("chart") as HTMLImageElement;
  const cap = document.getElementById("cap")!;
  const plotBox = document.getElementById("plotBox")!;

  const [h, ev] = await Promise.all([
    loadJson<TrainingHistory>("/ml/training_history.json"),
    loadJson<EvalSummary>("/ml/evaluation_summary.json"),
  ]);

  note.textContent = "Done.";
  vars.innerHTML = variablesHtml(h, ev);
  const verdict = overfittingVerdict(h);
  fit.textContent = verdict.text;
  fit.dataset.tone = verdict.tone;

  const url = plotSrc(ev) + `?t=${Date.now()}`;
  img.style.display = "";
  img.onload = () => {
    cap.hidden = true;
    plotBox.hidden = false;
  };
  img.onerror = () => {
    img.style.display = "none";
    cap.hidden = false;
    plotBox.hidden = false;
  };
  img.src = url;
}

async function init(): Promise<void> {
  mount();
  const run = document.getElementById("run") as HTMLButtonElement;
  const note = document.getElementById("note")!;

  run.addEventListener("click", async () => {
    run.disabled = true;
    clear();
    note.textContent = "Running…";
    try {
      await runPipeline();
      note.textContent = "Loading…";
      await show();
    } catch (e) {
      note.textContent = String((e as Error).message ?? e);
    } finally {
      run.disabled = false;
    }
  });

  clear();
  note.textContent = "Press Run to train, evaluate, and show the plot and numbers.";
}

init().catch((e) => {
  appRoot.innerHTML = `<pre class="err">${String((e as Error).stack ?? e)}</pre>`;
});
