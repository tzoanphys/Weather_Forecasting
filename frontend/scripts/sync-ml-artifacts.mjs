import fs from "node:fs";
import path from "node:path";

const ROOT = process.cwd(); // frontend/
const repoRoot = path.resolve(ROOT, "..");

const mlOutputs =
  process.env.ML_OUTPUT_DIR ?? path.resolve(repoRoot, "weather-ml-project", "outputs");

const destDir = path.resolve(ROOT, "public", "ml");
fs.mkdirSync(destDir, { recursive: true });

const filesToCopy = ["training_history.json", "evaluation_summary.json"];

for (const file of filesToCopy) {
  const src = path.resolve(mlOutputs, file);
  const dest = path.resolve(destDir, file);
  if (fs.existsSync(src)) {
    fs.copyFileSync(src, dest);
    console.log(`Copied ${file}`);
  } else if (fs.existsSync(dest)) {
    fs.unlinkSync(dest);
    console.log(`Removed stale ${file} (not present in ML outputs)`);
  }
}

if (fs.existsSync(mlOutputs)) {
  for (const entry of fs.readdirSync(mlOutputs)) {
    if (!entry.toLowerCase().endsWith(".png")) continue;
    if (entry !== "final_evaluation_plot.png") continue;
    fs.copyFileSync(path.resolve(mlOutputs, entry), path.resolve(destDir, entry));
  }
  console.log("Copied final_evaluation_plot.png (if present)");
}

if (fs.existsSync(destDir)) {
  for (const entry of fs.readdirSync(destDir)) {
    if (!/^error_maps_/i.test(entry)) continue;
    fs.unlinkSync(path.resolve(destDir, entry));
    console.log(`Removed ${entry}`);
  }
}

console.log(`Done. Frontend artifacts in: ${destDir}`);
