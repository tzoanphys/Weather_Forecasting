import "./style.css";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("Missing #app element");

app.innerHTML = `
  <div class="card">
    <div class="card__inner">
      <h1 class="title">Weather Forecasting</h1>
      <p class="muted">
        Frontend is running. Next step: connect to your backend/ML API and render forecasts.
      </p>
      <div class="row">
        <span class="pill">Vite</span>
        <span class="pill">TypeScript</span>
        <span class="pill">Port: ${location.port || "(default)"}</span>
      </div>
    </div>
  </div>
`;
