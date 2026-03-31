import { defineConfig, Plugin } from "vite";
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

type RunState =
  | { running: false; lastExitCode?: number; lastError?: string; lastFinishedAt?: string }
  | { running: true; startedAt: string; task: string; pid?: number };

let state: RunState = { running: false };

function json(res: any, status: number, body: unknown) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(body));
}

function copyArtifacts() {
  const frontendRoot = process.cwd();
  const repoRoot = path.resolve(frontendRoot, "..");
  const mlOutputs = path.resolve(repoRoot, "weather-ml-project", "outputs");
  const destDir = path.resolve(frontendRoot, "public", "ml");
  fs.mkdirSync(destDir, { recursive: true });

  for (const file of ["training_history.json", "evaluation_summary.json"]) {
    const src = path.resolve(mlOutputs, file);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.resolve(destDir, file));
  }

  if (fs.existsSync(mlOutputs)) {
    for (const entry of fs.readdirSync(mlOutputs)) {
      if (!entry.toLowerCase().endsWith(".png")) continue;
      fs.copyFileSync(path.resolve(mlOutputs, entry), path.resolve(destDir, entry));
    }
  }
}

function backendRunnerPlugin(): Plugin {
  return {
    name: "backend-runner",
    configureServer(server) {
      server.middlewares.use("/api/status", (req, res) => {
        json(res, 200, state);
      });

      server.middlewares.use("/api/run", async (req, res) => {
        if (req.method !== "POST") return json(res, 405, { error: "Use POST" });
        if (state.running) return json(res, 409, { error: "Already running", state });

        const url = new URL(req.url ?? "", "http://localhost");
        const task = url.searchParams.get("task") ?? "train+evaluate";

        const frontendRoot = process.cwd();
        const repoRoot = path.resolve(frontendRoot, "..");
        const projectRoot = path.resolve(repoRoot, "weather-ml-project");

        const run = (cmd: string, args: string[]) =>
          new Promise<number>((resolve, reject) => {
            const child = spawn(cmd, args, { cwd: projectRoot, stdio: "inherit" });
            child.on("error", reject);
            child.on("close", (code) => resolve(code ?? 0));
            state = { running: true, startedAt: new Date().toISOString(), task, pid: child.pid ?? undefined };
          });

        try {
          if (task === "train") {
            const code = await run("python3", ["src/train.py"]);
            state = { running: false, lastExitCode: code, lastFinishedAt: new Date().toISOString() };
          } else if (task === "evaluate") {
            const code = await run("python3", ["src/evaluate.py"]);
            state = { running: false, lastExitCode: code, lastFinishedAt: new Date().toISOString() };
          } else {
            const code1 = await run("python3", ["src/train.py"]);
            if (code1 !== 0) {
              state = { running: false, lastExitCode: code1, lastFinishedAt: new Date().toISOString() };
              return json(res, 500, { ok: false, exitCode: code1 });
            }
            const code2 = await run("python3", ["src/evaluate.py"]);
            state = { running: false, lastExitCode: code2, lastFinishedAt: new Date().toISOString() };
          }

          copyArtifacts();
          return json(res, 200, { ok: true, state });
        } catch (e: any) {
          state = { running: false, lastExitCode: 1, lastError: String(e?.stack ?? e), lastFinishedAt: new Date().toISOString() };
          return json(res, 500, { ok: false, error: state.lastError });
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [backendRunnerPlugin()],
});

