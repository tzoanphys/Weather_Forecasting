import "./style.css";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("Missing #app element");
const appEl: HTMLDivElement = app;

type TrainingHistory = {
  epochs: number[];
  train: { mse: number[]; mae: number[] };
  val: { mse: number[]; mae: number[] };
  ratio_val_to_train_mse: number[];
  best_epoch: number | null;
  best_val_mse: number | null;
  stopped_early: boolean;
};

type EvalSummary = {
  mse: { mean: number; std: number; min: number; max: number };
  mae: { mean: number; std: number; min: number; max: number };
  best: { dataset_index: number; evaluation_image: string; error_map_image: string };
  worst: { dataset_index: number; evaluation_image: string; error_map_image: string };
};

const fmt = (x: number) => (Number.isFinite(x) ? x.toFixed(6) : String(x));

function sparklineSvg(values: number[], width = 560, height = 160) {
  if (values.length < 2) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const pad = 10;
  const w = width - pad * 2;
  const h = height - pad * 2;

  const pts = values
    .map((v, i) => {
      const x = pad + (i / (values.length - 1)) * w;
      const y = pad + (1 - (v - min) / range) * h;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");

  return `
    <svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" role="img" aria-label="metric chart">
      <polyline fill="none" stroke="rgba(120,190,255,0.95)" stroke-width="2.2" points="${pts}" />
      <rect x="0" y="0" width="${width}" height="${height}" fill="none" stroke="rgba(255,255,255,0.10)" />
      <text x="${pad}" y="${height - 8}" fill="rgba(234,240,255,0.7)" font-size="12">min ${fmt(min)} · max ${fmt(max)}</text>
    </svg>
  `;
}

function dualSparklineSvg(
  a: number[],
  b: number[],
  labels: { a: string; b: string } = { a: "Train", b: "Val" },
  width = 560,
  height = 180,
) {
  const n = Math.min(a.length, b.length);
  if (n < 2) return "";

  const aN = a.slice(0, n);
  const bN = b.slice(0, n);
  const all = [...aN, ...bN];

  const min = Math.min(...all);
  const max = Math.max(...all);
  const range = max - min || 1;
  const pad = 12;
  const w = width - pad * 2;
  const h = height - pad * 2;

  const pts = (values: number[]) =>
    values
      .map((v, i) => {
        const x = pad + (i / (n - 1)) * w;
        const y = pad + (1 - (v - min) / range) * h;
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");

  const aPts = pts(aN);
  const bPts = pts(bN);

  return `
    <svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" role="img" aria-label="metric chart">
      <rect x="0" y="0" width="${width}" height="${height}" fill="none" stroke="rgba(255,255,255,0.10)" />
      <polyline fill="none" stroke="rgba(120,190,255,0.95)" stroke-width="2.4" points="${aPts}" />
      <polyline fill="none" stroke="rgba(255,170,110,0.95)" stroke-width="2.4" points="${bPts}" />

      <g font-size="12" fill="rgba(234,240,255,0.75)">
        <rect x="${pad}" y="${pad - 2}" width="12" height="3" fill="rgba(120,190,255,0.95)" />
        <text x="${pad + 18}" y="${pad + 2}">${labels.a}</text>
        <rect x="${pad + 90}" y="${pad - 2}" width="12" height="3" fill="rgba(255,170,110,0.95)" />
        <text x="${pad + 108}" y="${pad + 2}">${labels.b}</text>
      </g>

      <text x="${pad}" y="${height - 10}" fill="rgba(234,240,255,0.7)" font-size="12">min ${fmt(min)} · max ${fmt(max)}</text>
    </svg>
  `;
}

function overfittingBadge(history: TrainingHistory) {
  const t = history.train.mse;
  const v = history.val.mse;
  if (t.length < 8 || v.length < 8) return { label: "Not enough data", level: "pill" };

  const lastN = 6;
  const t0 = t[t.length - lastN];
  const t1 = t[t.length - 1];
  const v0 = v[v.length - lastN];
  const v1 = v[v.length - 1];
  const ratio = history.ratio_val_to_train_mse[history.ratio_val_to_train_mse.length - 1] ?? Infinity;

  // Heuristic: train improves while val worsens, plus a big gap.
  const trainImproved = t1 < t0 * 0.98;
  const valWorsened = v1 > v0 * 1.03;
  const bigGap = ratio >= 1.35;

  if (trainImproved && valWorsened && bigGap) return { label: "Likely overfitting", level: "pill pill--bad" };
  if (bigGap) return { label: "Possible overfitting (val≫train)", level: "pill pill--warn" };
  return { label: "Looks OK", level: "pill pill--good" };
}

function mean(xs: number[]) {
  const ok = xs.filter((x) => Number.isFinite(x));
  if (ok.length === 0) return NaN;
  return ok.reduce((a, b) => a + b, 0) / ok.length;
}

function overfittingAnalysis(history: TrainingHistory) {
  const trainMse = history.train.mse;
  const valMse = history.val.mse;
  const n = Math.min(trainMse.length, valMse.length);
  if (n < 8) {
    return {
      verdict: "Not enough epochs to assess overfitting.",
      level: "good" as const,
      bullets: ["Train/val comparison becomes meaningful after ~8–10 epochs."],
    };
  }

  const bestEpoch = history.best_epoch ?? null;
  const lastRatio = history.ratio_val_to_train_mse[history.ratio_val_to_train_mse.length - 1] ?? Infinity;

  const tail = 6;
  const tTail = trainMse.slice(-tail);
  const vTail = valMse.slice(-tail);
  const tDelta = tTail[tTail.length - 1] - tTail[0];
  const vDelta = vTail[vTail.length - 1] - vTail[0];

  const gapLabel =
    lastRatio < 1.15 ? "small" : lastRatio < 1.35 ? "moderate" : "large";

  const bullets: string[] = [
    `Generalization gap (val/train MSE) is ${gapLabel}: ${fmt(lastRatio)}.`,
    `Last ${tail} epochs trend: train Δ ${fmt(tDelta)} (↓ is good), val Δ ${fmt(vDelta)} (↑ is bad).`,
  ];

  // If the best epoch is meaningfully earlier than the last, that’s a common overfit symptom.
  if (bestEpoch && bestEpoch < history.epochs[history.epochs.length - 1]) {
    bullets.push(`Best validation epoch was ${bestEpoch}, earlier than the last epoch (${history.epochs.at(-1)}).`);
  }

  // Verdict rules: prioritize (1) big gap, (2) opposite trends, (3) best epoch earlier.
  const oppositeTrends = tDelta < 0 && vDelta > 0;
  const bigGap = lastRatio >= 1.35;

  if (bigGap && oppositeTrends) {
    return {
      verdict: "Likely overfitting: training keeps improving but validation is getting worse.",
      level: "bad" as const,
      bullets,
    };
  }

  if (bigGap) {
    return {
      verdict: "Possible overfitting: validation error is much higher than training.",
      level: "warn" as const,
      bullets: [...bullets, "This can also happen with a small/noisy validation set."],
    };
  }

  if (oppositeTrends) {
    return {
      verdict: "Some overfitting signal in the last epochs (val rising while train falls).",
      level: "warn" as const,
      bullets,
    };
  }

  return {
    verdict: "No strong overfitting signal in the metrics shown.",
    level: "good" as const,
    bullets,
  };
}

async function loadJson<T>(url: string): Promise<T | null> {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) return null;
  return (await res.json()) as T;
}

async function runBackend(task: "train" | "evaluate" | "train+evaluate") {
  const res = await fetch(`/api/run?task=${encodeURIComponent(task)}`, { method: "POST" });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body?.error ?? `Backend run failed (${res.status})`);
  return body;
}

async function backendStatus() {
  const res = await fetch("/api/status", { cache: "no-store" });
  if (!res.ok) return null;
  return (await res.json()) as any;
}

async function render() {
  appEl.innerHTML = `
    <div class="card">
      <div class="card__inner">
        <div class="row" style="justify-content: space-between;">
          <div>
            <h1 class="title">Weather Forecasting — Training Dashboard</h1>
            <p class="muted">Shows best evaluation sample + error maps, epoch metrics (MSE/MAE), and an overfitting signal.</p>
          </div>
          <div class="row" style="gap: 10px;">
            <button id="runBackendBtn" class="btn">Run backend (train + evaluate)</button>
          </div>
        </div>

        <div id="status" class="muted">Loading ML artifacts from <code>/ml/</code> …</div>
        <div id="backendStatus" class="muted" style="margin-top: 6px;"></div>

        <div class="row" style="align-items: stretch; gap: 16px;">
          <div style="flex: 1; min-width: 260px;">
            <div class="pill" style="display:inline-block;margin-bottom:10px;">Training (per epoch)</div>
            <div id="trainCharts"></div>
            <div id="trainCards"></div>
          </div>
          <div style="flex: 1; min-width: 260px;">
            <div class="pill" style="display:inline-block;margin-bottom:10px;">Evaluation (best / worst)</div>
            <div id="evalCards"></div>
            <div id="evalImages"></div>
          </div>
        </div>
      </div>
    </div>
  `;

  const status = document.querySelector<HTMLDivElement>("#status")!;
  const backendStatusEl = document.querySelector<HTMLDivElement>("#backendStatus")!;
  const trainCards = document.querySelector<HTMLDivElement>("#trainCards")!;
  const trainCharts = document.querySelector<HTMLDivElement>("#trainCharts")!;
  const evalCards = document.querySelector<HTMLDivElement>("#evalCards")!;
  const evalImages = document.querySelector<HTMLDivElement>("#evalImages")!;
  const runBtn = document.querySelector<HTMLButtonElement>("#runBackendBtn")!;

  const refreshBackendStatus = async () => {
    const st = await backendStatus();
    if (!st) {
      backendStatusEl.textContent = "";
      return;
    }
    if (st.running) {
      backendStatusEl.textContent = `Backend is running (${st.task})… started ${st.startedAt}`;
    } else if (st.lastFinishedAt) {
      const ok = st.lastExitCode === 0;
      backendStatusEl.textContent = `Last backend run: ${ok ? "OK" : "FAILED"} (exit ${st.lastExitCode}) at ${st.lastFinishedAt}`;
    } else {
      backendStatusEl.textContent = "";
    }
  };

  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    status.textContent = "Running backend… (this can take a while)";
    try {
      await runBackend("train+evaluate");
      status.textContent = "Backend finished. Reloading artifacts…";
      await render();
    } catch (e: any) {
      status.textContent = `Backend run failed: ${String(e?.message ?? e)}`;
      await refreshBackendStatus();
    } finally {
      runBtn.disabled = false;
    }
  });

  await refreshBackendStatus();

  const [history, evalSummary] = await Promise.all([
    loadJson<TrainingHistory>("/ml/training_history.json"),
    loadJson<EvalSummary>("/ml/evaluation_summary.json"),
  ]);

  if (!history && !evalSummary) {
    status.innerHTML = `
      Missing ML artifact files.
      Generate them by running training + evaluation, then copy artifacts:
      <br/><br/>
      <code>python weather-ml-project/main.py</code> (or run <code>src/train.py</code> + <code>src/evaluate.py</code>)
      <br/>
      <code>cd frontend && npm run sync:ml</code>
    `;
    return;
  }

  status.textContent = "Loaded.";

  if (history) {
    const badge = overfittingBadge(history);
    const analysis = overfittingAnalysis(history);
    const lastEpoch = history.epochs[history.epochs.length - 1] ?? 0;
    const lastTrainMse = history.train.mse[history.train.mse.length - 1] ?? NaN;
    const lastValMse = history.val.mse[history.val.mse.length - 1] ?? NaN;
    const lastTrainMae = history.train.mae[history.train.mae.length - 1] ?? NaN;
    const lastValMae = history.val.mae[history.val.mae.length - 1] ?? NaN;
    const ratio = history.ratio_val_to_train_mse[history.ratio_val_to_train_mse.length - 1] ?? NaN;

    trainCards.innerHTML = `
      <div class="section">
        <div class="section__title">Training summary</div>
        <div class="row" style="gap: 10px; margin-top: 6px;">
          <span class="${badge.level}">${badge.label}</span>
          <span class="pill">Best epoch: ${history.best_epoch ?? "—"}</span>
          <span class="pill">Early stop: ${history.stopped_early ? "yes" : "no"}</span>
        </div>
        <div class="grid" style="margin-top: 10px;">
          <div class="stat"><div class="stat__k">Last epoch</div><div class="stat__v">${lastEpoch}</div></div>
          <div class="stat"><div class="stat__k">Train MSE</div><div class="stat__v">${fmt(lastTrainMse)}</div></div>
          <div class="stat"><div class="stat__k">Val MSE</div><div class="stat__v">${fmt(lastValMse)}</div></div>
          <div class="stat"><div class="stat__k">Train MAE</div><div class="stat__v">${fmt(lastTrainMae)}</div></div>
          <div class="stat"><div class="stat__k">Val MAE</div><div class="stat__v">${fmt(lastValMae)}</div></div>
          <div class="stat"><div class="stat__k">Val/Train MSE</div><div class="stat__v">${fmt(ratio)}</div></div>
        </div>
      </div>

      <div class="section" style="margin-top: 12px;">
        <div class="section__title">Overfitting interpretation</div>
        <div class="callout callout--${analysis.level}">
          <div class="callout__title">${analysis.verdict}</div>
          <ul class="callout__list">
            ${analysis.bullets.map((b) => `<li>${b}</li>`).join("")}
          </ul>
        </div>
      </div>
    `;

    trainCharts.innerHTML = `
      <div style="margin-top: 8px;">
        <div class="muted" style="margin-bottom: 6px;">MSE (Train vs Val)</div>
        ${dualSparklineSvg(history.train.mse, history.val.mse)}
      </div>
      <div style="margin-top: 14px;">
        <div class="muted" style="margin-bottom: 6px;">MAE (Train vs Val)</div>
        ${dualSparklineSvg(history.train.mae, history.val.mae)}
      </div>
    `;
  } else {
    trainCards.innerHTML = `<div class="muted">No <code>training_history.json</code> found.</div>`;
  }

  if (evalSummary) {
    evalCards.innerHTML = `
      <div class="row" style="gap: 10px; margin-bottom: 12px;">
        <span class="pill">Best sample: ${evalSummary.best.dataset_index}</span>
        <span class="pill">Worst sample: ${evalSummary.worst.dataset_index}</span>
      </div>
      <div class="row" style="gap: 10px; margin-bottom: 12px;">
        <span class="pill">MSE mean: ${fmt(evalSummary.mse.mean)}</span>
        <span class="pill">MAE mean: ${fmt(evalSummary.mae.mean)}</span>
        <span class="pill">MSE min: ${fmt(evalSummary.mse.min)}</span>
        <span class="pill">MSE max: ${fmt(evalSummary.mse.max)}</span>
      </div>
    `;

    const bestEval = `/ml/${evalSummary.best.evaluation_image}`;
    const bestErr = `/ml/${evalSummary.best.error_map_image}`;

    evalImages.innerHTML = `
      <div style="display:grid; gap: 12px;">
        <div>
          <div class="muted" style="margin-bottom: 6px;">Best sample — prediction vs truth</div>
          <img src="${bestEval}" alt="Best evaluation sample" style="width:100%; border-radius: 12px; border: 1px solid rgba(255,255,255,0.10);" />
        </div>
        <div>
          <div class="muted" style="margin-bottom: 6px;">Best sample — error maps</div>
          <img src="${bestErr}" alt="Best error maps" style="width:100%; border-radius: 12px; border: 1px solid rgba(255,255,255,0.10);" />
        </div>
      </div>
    `;
  } else {
    evalCards.innerHTML = `<div class="muted">No <code>evaluation_summary.json</code> found.</div>`;
  }
}

render().catch((e) => {
  appEl.innerHTML = `<pre class="muted">${String(e?.stack ?? e)}</pre>`;
});
