import fs from "node:fs";
import path from "node:path";

const projectRoot = path.resolve(import.meta.dirname, "..");
const runRoot = path.join(
  projectRoot,
  "attack_results",
  "paper_suite_theia_final_v1",
  "runs",
);
const outputDir = path.join(projectRoot, "paper_figures");

const policies = [
  {
    key: "linucb",
    label: "LinUCB",
    prefix: "t3_linucb_margin_k10_r5_seed",
    stroke: "#111111",
    fill: "#b8b8b8",
    dash: "",
  },
  {
    key: "ddqn",
    label: "Double DQN",
    prefix: "t3_double_dqn_margin_k10_r5_seed",
    stroke: "#555555",
    fill: "#d0d0d0",
    dash: "10 6",
  },
  {
    key: "ppo",
    label: "Online PPO",
    prefix: "t3_ppo_online_margin_k10_r5_seed",
    stroke: "#888888",
    fill: "#e0e0e0",
    dash: "13 5 2 5",
  },
];
const seeds = [42, 43, 44];
const windowSize = 10;

function sampleStd(values, meanValue) {
  if (values.length < 2) return 0;
  return Math.sqrt(
    values.reduce((sum, value) => sum + (value - meanValue) ** 2, 0)
      / (values.length - 1),
  );
}

for (const policy of policies) {
  policy.runs = seeds.map((seed) => {
    const trajectoryPath = path.join(
      runRoot,
      `${policy.prefix}${seed}`,
      "trajectory.json",
    );
    const payload = JSON.parse(fs.readFileSync(trajectoryPath, "utf8"));
    const feedback = payload.queries.filter((item) => item.kind === "attack_feedback");
    return {
      seed,
      rewards: feedback.map((item) => Number(item.total_reward ?? 0)),
      successes: feedback.map((item) => item.crossed_threshold ? 1 : 0),
    };
  });
}

const commonLength = Math.min(
  ...policies.flatMap((policy) => policy.runs.map((run) => run.rewards.length)),
);

for (const policy of policies) {
  policy.reward = [];
  policy.cumulative = [];
  for (let query = 1; query <= commonLength; query += 1) {
    const cumulativeValues = policy.runs.map((run) =>
      run.successes.slice(0, query).reduce((sum, value) => sum + value, 0));
    const cumulativeMean = cumulativeValues.reduce((a, b) => a + b, 0)
      / cumulativeValues.length;
    policy.cumulative.push({
      x: query,
      mean: cumulativeMean,
      std: sampleStd(cumulativeValues, cumulativeMean),
    });

    if (query >= windowSize) {
      const rewardValues = policy.runs.map((run) => {
        const values = run.rewards.slice(query - windowSize, query);
        return values.reduce((a, b) => a + b, 0) / windowSize;
      });
      const rewardMean = rewardValues.reduce((a, b) => a + b, 0)
        / rewardValues.length;
      policy.reward.push({
        x: query,
        mean: rewardMean,
        std: sampleStd(rewardValues, rewardMean),
      });
    }
  }
}

const csvHeader = ["query"];
for (const policy of policies) {
  csvHeader.push(
    `${policy.key}_reward_mean`,
    `${policy.key}_reward_std`,
    `${policy.key}_cumulative_evasions_mean`,
    `${policy.key}_cumulative_evasions_std`,
  );
}
const csvRows = [csvHeader.join(",")];
for (let query = 1; query <= commonLength; query += 1) {
  const row = [String(query)];
  for (const policy of policies) {
    const reward = policy.reward.find((item) => item.x === query);
    const cumulative = policy.cumulative[query - 1];
    row.push(
      reward ? reward.mean.toFixed(8) : "",
      reward ? reward.std.toFixed(8) : "",
      cumulative.mean.toFixed(8),
      cumulative.std.toFixed(8),
    );
  }
  csvRows.push(row.join(","));
}
fs.writeFileSync(
  path.join(outputDir, "action-policy-convergence-data.csv"),
  `${csvRows.join("\n")}\n`,
  "utf8",
);

const width = 1600;
const height = 650;
const plotTop = 105;
const plotHeight = 440;
const left1 = 95;
const plotWidth = 650;
const left2 = 875;
const rewardMin = -0.25;
const rewardMax = 0.40;
const successMin = 0;
const successMax = 30;

const xScale = (left, query) => left + ((query - 1) / (commonLength - 1)) * plotWidth;
const yScale = (value, min, max) =>
  plotTop + plotHeight - ((value - min) / (max - min)) * plotHeight;
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const fmt = (value) => Number(value).toFixed(1).replace("-0.0", "0.0");

function linePath(series, left, min, max, valueKey = "mean") {
  return series.map((point, index) => {
    const x = xScale(left, point.x).toFixed(2);
    const y = yScale(clamp(point[valueKey], min, max), min, max).toFixed(2);
    return `${index === 0 ? "M" : "L"}${x} ${y}`;
  }).join(" ");
}

function bandPath(series, left, min, max) {
  const upper = series.map((point) => {
    const x = xScale(left, point.x).toFixed(2);
    const y = yScale(clamp(point.mean + point.std, min, max), min, max).toFixed(2);
    return `${x} ${y}`;
  });
  const lower = [...series].reverse().map((point) => {
    const x = xScale(left, point.x).toFixed(2);
    const y = yScale(clamp(point.mean - point.std, min, max), min, max).toFixed(2);
    return `${x} ${y}`;
  });
  return `M${upper.join(" L")} L${lower.join(" L")} Z`;
}

function axes(left, yTicks, min, max, title, yLabel, xTicks) {
  const parts = [];
  parts.push(`<text class="panelTitle" x="${left + plotWidth / 2}" y="78">${title}</text>`);
  for (const tick of yTicks) {
    const y = yScale(tick, min, max);
    parts.push(`<line class="grid" x1="${left}" y1="${y}" x2="${left + plotWidth}" y2="${y}"/>`);
    parts.push(`<text class="tick" x="${left - 14}" y="${y + 6}" text-anchor="end">${Number.isInteger(tick) ? tick : fmt(tick)}</text>`);
  }
  for (const tick of xTicks) {
    const x = xScale(left, tick);
    parts.push(`<line class="vgrid" x1="${x}" y1="${plotTop}" x2="${x}" y2="${plotTop + plotHeight}"/>`);
    parts.push(`<text class="tick" x="${x}" y="${plotTop + plotHeight + 30}" text-anchor="middle">${tick}</text>`);
  }
  parts.push(`<line class="axis" x1="${left}" y1="${plotTop}" x2="${left}" y2="${plotTop + plotHeight}"/>`);
  parts.push(`<line class="axis" x1="${left}" y1="${plotTop + plotHeight}" x2="${left + plotWidth}" y2="${plotTop + plotHeight}"/>`);
  parts.push(`<text class="axisLabel" x="${left + plotWidth / 2}" y="${plotTop + plotHeight + 68}">Detector Feedback Query</text>`);
  parts.push(`<text class="axisLabel" transform="translate(${left - 67} ${plotTop + plotHeight / 2}) rotate(-90)">${yLabel}</text>`);
  return parts.join("\n");
}

const svg = [];
svg.push(`<?xml version="1.0" encoding="UTF-8"?>`);
svg.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">`);
svg.push(`<title id="title">Learning dynamics of adaptive action-selection policies</title>`);
svg.push(`<desc id="desc">Ten-query moving-average reward and cumulative target evasions for LinUCB, Double DQN, and Online PPO over the first 80 detector feedback queries. Lines show means and shaded bands show one standard deviation across seeds 42, 43, and 44.</desc>`);
svg.push(`<style>
  .bg{fill:#fff}.axis{stroke:#111;stroke-width:2}.grid{stroke:#d5d5d5;stroke-width:1}.vgrid{stroke:#ececec;stroke-width:1}.zero{stroke:#777;stroke-width:1.5}.curve{fill:none;stroke-width:3;stroke-linecap:round;stroke-linejoin:round}.band{stroke:none;opacity:.42}.tick{font:17px Arial,Helvetica,sans-serif;fill:#222}.axisLabel{font:20px Arial,Helvetica,sans-serif;fill:#111;text-anchor:middle}.panelTitle{font:700 22px Georgia,'Times New Roman',serif;fill:#111;text-anchor:middle}.legend{font:18px Arial,Helvetica,sans-serif;fill:#111}.note{font:15px Georgia,'Times New Roman',serif;fill:#444;text-anchor:middle}
</style>`);
svg.push(`<rect class="bg" width="${width}" height="${height}"/>`);

let legendX = 430;
for (const policy of policies) {
  svg.push(`<line x1="${legendX}" y1="28" x2="${legendX + 58}" y2="28" stroke="${policy.stroke}" stroke-width="3"${policy.dash ? ` stroke-dasharray="${policy.dash}"` : ""}/>`);
  svg.push(`<text class="legend" x="${legendX + 68}" y="34">${policy.label}</text>`);
  legendX += policy.label === "Double DQN" ? 250 : 210;
}

svg.push(axes(
  left1,
  [-0.2, 0, 0.2, 0.4],
  rewardMin,
  rewardMax,
  "(a) Smoothed Reward",
  "10-Query Moving-Average Reward",
  [10, 20, 40, 60, 80],
));
const zeroY = yScale(0, rewardMin, rewardMax);
svg.push(`<line class="zero" x1="${left1}" y1="${zeroY}" x2="${left1 + plotWidth}" y2="${zeroY}"/>`);

svg.push(axes(
  left2,
  [0, 5, 10, 15, 20, 25, 30],
  successMin,
  successMax,
  "(b) Cumulative Target Evasions",
  "Successful Target Evasions",
  [1, 20, 40, 60, 80],
));

for (const policy of [...policies].reverse()) {
  svg.push(`<path class="band" d="${bandPath(policy.reward, left1, rewardMin, rewardMax)}" fill="${policy.fill}"/>`);
  svg.push(`<path class="band" d="${bandPath(policy.cumulative, left2, successMin, successMax)}" fill="${policy.fill}"/>`);
}
for (const policy of policies) {
  const dash = policy.dash ? ` stroke-dasharray="${policy.dash}"` : "";
  svg.push(`<path class="curve" d="${linePath(policy.reward, left1, rewardMin, rewardMax)}" stroke="${policy.stroke}"${dash}/>`);
  svg.push(`<path class="curve" d="${linePath(policy.cumulative, left2, successMin, successMax)}" stroke="${policy.stroke}"${dash}/>`);
}

svg.push(`</svg>`);

fs.writeFileSync(
  path.join(outputDir, "action-policy-convergence.svg"),
  `${svg.join("\n")}\n`,
  "utf8",
);

console.log(JSON.stringify({
  commonLength,
  policies: Object.fromEntries(policies.map((policy) => [
    policy.label,
    {
      runLengths: policy.runs.map((run) => run.rewards.length),
      finalSmoothedReward: policy.reward.at(-1).mean,
      finalSmoothedRewardStd: policy.reward.at(-1).std,
      cumulativeEvasionsAtCommonBudget: policy.cumulative.at(-1).mean,
      cumulativeEvasionsStd: policy.cumulative.at(-1).std,
    },
  ])),
}, null, 2));
