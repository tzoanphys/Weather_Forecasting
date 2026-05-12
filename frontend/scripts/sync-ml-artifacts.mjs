import fs from "node:fs";
import path from "node:path";

const ROOT = process.cwd(); // frontend/
const repoRoot = path.resolve(ROOT, "..");

const mlOutputs =
  process.env.ML_OUTPUT_DIR ?? path.resolve(repoRoot, "weather-ml-project", "outputs");

const destDir = path.resolve(ROOT, "public", "ml");
fs.mkdirSync(destDir, { recursive: true });

const filesToCopy = [
  "training_history.json",
  "evaluation_summary.json",
];

for (const file of filesToCopy) {
  const src = path.resolve(mlOutputs, file);
  if (!fs.existsSync(src)) continue;
  fs.copyFileSync(src, path.resolve(destDir, file));
  console.log(`Copied ${file}`);
}

// Copy any rendered PNGs so the dashboard can show them.
if (fs.existsSync(mlOutputs)) {
  for (const entry of fs.readdirSync(mlOutputs)) {
    if (!entry.toLowerCase().endsWith(".png")) continue;
    fs.copyFileSync(path.resolve(mlOutputs, entry), path.resolve(destDir, entry));
  }
  console.log("Copied PNG artifacts");
}

console.log(`Done. Frontend artifacts in: ${destDir}`);

