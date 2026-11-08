import fs from "node:fs";
import path from "node:path";

const requiredAssets = [
  "Complete-project/frontend/assets/editions/hero-chiaroscuro-swarm.webp",
  "Complete-project/frontend/assets/editions/cutout-mercury-agent.png",
  "Complete-project/frontend/assets/editions/cutout-federated-astrolabe.png",
  "Complete-project/frontend/assets/editions/centralized-sun-hub.webp",
  "Complete-project/frontend/assets/editions/federated-armillary-map.webp",
  "Complete-project/reports/figures/03_campaign_network_scored.png",
  "Complete-project/reports/figures/03_feature_separation.png",
  "Complete-project/reports/figures/02_roc_pr.png",
  "Complete-project/reports/figures/04_fusion_roc_pr.png",
  "Complete-project/reports/figures/04_meta_coefficients.png",
];

let failed = false;
for (const rel of requiredAssets) {
  if (!fs.existsSync(rel) || fs.statSync(rel).size < 1024) {
    console.error(`Missing or empty visual asset: ${rel}`);
    failed = true;
  }
}

if (failed) {
  process.exit(1);
}

console.log("PASS: All 10 required visual assets exist and are non-empty.");
