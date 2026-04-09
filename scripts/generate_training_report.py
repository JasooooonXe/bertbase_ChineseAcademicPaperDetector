from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an HTML training report from run logs.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing stdout.log and metrics.json.")
    parser.add_argument("--output-file", default=None, help="Optional output html file path.")
    return parser.parse_args()


def parse_log_records(stdout_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_logs: list[dict[str, Any]] = []
    eval_logs: list[dict[str, Any]] = []
    for raw_line in stdout_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            payload = ast.literal_eval(line)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        normalized: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, str):
                try:
                    normalized[key] = float(value)
                except ValueError:
                    normalized[key] = value
            else:
                normalized[key] = value
        if "eval_loss" in normalized or "test_loss" in normalized:
            eval_logs.append(normalized)
        elif "loss" in normalized:
            train_logs.append(normalized)
    return train_logs, eval_logs


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _format_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def build_html(
    run_dir: Path,
    metrics: dict[str, Any],
    train_logs: list[dict[str, Any]],
    eval_logs: list[dict[str, Any]],
) -> str:
    report_payload = {
        "train_logs": train_logs,
        "eval_logs": eval_logs,
        "metrics": metrics,
        "run_name": run_dir.name,
    }
    metrics_json = json.dumps(report_payload, ensure_ascii=False)
    validation = metrics.get("validation", {})
    test = metrics.get("test", {})
    train = metrics.get("train", {})
    best_checkpoint = metrics.get("best_checkpoint", "-")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Training Report · {run_dir.name}</title>
  <style>
    :root {{
      --bg: #f5efe4;
      --card: rgba(255, 251, 245, 0.88);
      --ink: #181714;
      --muted: #6b655c;
      --line: rgba(24, 23, 20, 0.12);
      --accent: #ab4b2f;
      --accent-2: #2e5a4d;
      --accent-3: #a38136;
      --shadow: 0 24px 80px rgba(62, 38, 20, 0.14);
      --sans: "Aptos", "Segoe UI Variable Display", "Microsoft YaHei UI", sans-serif;
      --serif: "Georgia", "Times New Roman", "Noto Serif SC", serif;
      --mono: "IBM Plex Mono", Consolas, monospace;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: var(--sans);
      background:
        radial-gradient(circle at top left, rgba(171, 75, 47, 0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(46, 90, 77, 0.16), transparent 24%),
        linear-gradient(180deg, #f8f3eb, #efe3cf 100%);
    }}
    .page {{
      width: min(1320px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 24px 0 60px;
    }}
    .hero, .card {{
      border: 1px solid var(--line);
      border-radius: 30px;
      background: var(--card);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .hero {{
      padding: 34px 36px;
      margin-bottom: 24px;
      position: relative;
      overflow: hidden;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      width: 260px;
      height: 260px;
      right: -60px;
      top: -80px;
      border-radius: 999px;
      background: radial-gradient(circle, rgba(171, 75, 47, 0.22), transparent 66%);
    }}
    .eyebrow {{
      display: inline-block;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(24, 23, 20, 0.06);
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 14px 0 8px;
      font: 700 clamp(34px, 6vw, 62px)/1 var(--serif);
      letter-spacing: -0.04em;
    }}
    .sub {{
      margin: 0;
      max-width: 760px;
      color: var(--muted);
      line-height: 1.8;
      font-size: 17px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 16px;
      margin-top: 24px;
    }}
    .metric {{
      padding: 18px;
      border-radius: 20px;
      background: rgba(255,255,255,0.6);
      border: 1px solid rgba(24,23,20,0.08);
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .metric .value {{
      margin-top: 8px;
      font-size: 30px;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 24px;
    }}
    .stack {{
      display: grid;
      gap: 24px;
    }}
    .card {{
      padding: 24px;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 21px;
    }}
    .desc {{
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.7;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
    }}
    .chart-card {{
      border: 1px solid rgba(24,23,20,0.08);
      border-radius: 22px;
      background: linear-gradient(180deg, rgba(255,255,255,0.72), rgba(255,255,255,0.52));
      padding: 16px;
    }}
    .chart-card h3 {{
      margin: 0 0 8px;
      font-size: 16px;
    }}
    .chart-svg {{
      width: 100%;
      height: 220px;
      display: block;
      border-radius: 16px;
      background: rgba(255,255,255,0.6);
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
      margin-right: 6px;
    }}
    .meta-list {{
      display: grid;
      gap: 14px;
    }}
    .meta-item {{
      padding: 16px;
      border-radius: 18px;
      border: 1px solid rgba(24,23,20,0.08);
      background: rgba(255,255,255,0.58);
    }}
    .meta-item .k {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.07em;
    }}
    .meta-item .v {{
      margin-top: 8px;
      word-break: break-word;
      font-family: var(--mono);
      font-size: 14px;
      line-height: 1.7;
    }}
    .footer-note {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.8;
    }}
    @media (max-width: 980px) {{
      .metrics, .grid, .chart-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <span class="eyebrow">Training Dashboard</span>
      <h1>{run_dir.name}</h1>
      <p class="sub">自动解析 stdout.log 与 metrics.json，展示训练损失、验证指标、学习率变化，以及最终最佳 checkpoint 位置。这个页面可以直接双击打开，也可以放到项目文档里做训练汇报。</p>
      <div class="metrics">
        <div class="metric"><div class="label">Validation F1</div><div class="value">{_format_metric(validation.get("eval_f1"))}</div></div>
        <div class="metric"><div class="label">Validation AUC</div><div class="value">{_format_metric(validation.get("eval_roc_auc"))}</div></div>
        <div class="metric"><div class="label">Test F1</div><div class="value">{_format_metric(test.get("test_f1"))}</div></div>
        <div class="metric"><div class="label">Train Loss</div><div class="value">{_format_metric(train.get("train_loss"))}</div></div>
        <div class="metric"><div class="label">Runtime (s)</div><div class="value">{_format_metric(train.get("train_runtime"), 1)}</div></div>
      </div>
    </section>

    <section class="grid">
      <div class="stack">
        <section class="card">
          <h2>训练过程曲线</h2>
          <p class="desc">左侧关注收敛，右侧关注泛化。默认从 stdout.log 提取每次训练打印与验证打印，不依赖外部服务。</p>
          <div class="chart-grid">
            <div class="chart-card">
              <h3>Train Loss</h3>
              <svg class="chart-svg" id="chart-train-loss"></svg>
            </div>
            <div class="chart-card">
              <h3>Learning Rate</h3>
              <svg class="chart-svg" id="chart-lr"></svg>
            </div>
            <div class="chart-card">
              <h3>Validation F1</h3>
              <svg class="chart-svg" id="chart-val-f1"></svg>
            </div>
            <div class="chart-card">
              <h3>Validation ROC-AUC</h3>
              <svg class="chart-svg" id="chart-val-auc"></svg>
            </div>
          </div>
        </section>
      </div>

      <aside class="stack">
        <section class="card">
          <h2>运行摘要</h2>
          <div class="meta-list">
            <div class="meta-item"><div class="k">Best Checkpoint</div><div class="v">{best_checkpoint}</div></div>
            <div class="meta-item"><div class="k">Train Samples / s</div><div class="v">{_format_metric(train.get("train_samples_per_second"), 3)}</div></div>
            <div class="meta-item"><div class="k">Validation Accuracy</div><div class="v">{_format_metric(validation.get("eval_accuracy"))}</div></div>
            <div class="meta-item"><div class="k">Test Accuracy</div><div class="v">{_format_metric(test.get("test_accuracy"))}</div></div>
            <div class="meta-item"><div class="k">Total Parsed Train Logs</div><div class="v">{len(train_logs)}</div></div>
            <div class="meta-item"><div class="k">Total Parsed Eval Logs</div><div class="v">{len(eval_logs)}</div></div>
          </div>
          <p class="footer-note">如果训练日志里持续打印 loss 且验证集 F1 / AUC 稳定上升，这通常说明训练过程健康。若 train loss 继续下降但验证指标掉头，优先考虑过拟合或评估分布偏移。</p>
        </section>
      </aside>
    </section>
  </main>

  <script>
    const REPORT = {metrics_json};

    function drawLineChart(svgId, rows, xKey, yKey, color) {{
      const svg = document.getElementById(svgId);
      const width = svg.clientWidth || 420;
      const height = svg.clientHeight || 220;
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      svg.innerHTML = "";

      if (!rows.length) {{
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", "24");
        text.setAttribute("y", "40");
        text.setAttribute("fill", "#6b655c");
        text.setAttribute("font-size", "14");
        text.textContent = "No data";
        svg.appendChild(text);
        return;
      }}

      const margin = {{ top: 18, right: 16, bottom: 28, left: 42 }};
      const innerW = width - margin.left - margin.right;
      const innerH = height - margin.top - margin.bottom;
      const xs = rows.map((row, index) => Number(row[xKey] ?? index));
      const ys = rows.map((row) => Number(row[yKey]));
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);

      function scaleX(value) {{
        if (maxX === minX) return margin.left + innerW / 2;
        return margin.left + ((value - minX) / (maxX - minX)) * innerW;
      }}

      function scaleY(value) {{
        if (maxY === minY) return margin.top + innerH / 2;
        return margin.top + innerH - ((value - minY) / (maxY - minY)) * innerH;
      }}

      for (let i = 0; i < 4; i++) {{
        const y = margin.top + (innerH / 3) * i;
        const grid = document.createElementNS("http://www.w3.org/2000/svg", "line");
        grid.setAttribute("x1", margin.left);
        grid.setAttribute("x2", width - margin.right);
        grid.setAttribute("y1", y);
        grid.setAttribute("y2", y);
        grid.setAttribute("stroke", "rgba(24,23,20,0.08)");
        svg.appendChild(grid);
      }}

      let d = "";
      rows.forEach((row, index) => {{
        const x = scaleX(xs[index]);
        const y = scaleY(ys[index]);
        d += index === 0 ? `M ${{x}} ${{y}}` : ` L ${{x}} ${{y}}`;
      }});

      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", d);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", color);
      path.setAttribute("stroke-width", "3");
      path.setAttribute("stroke-linecap", "round");
      path.setAttribute("stroke-linejoin", "round");
      svg.appendChild(path);

      const firstLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
      firstLabel.setAttribute("x", margin.left);
      firstLabel.setAttribute("y", height - 8);
      firstLabel.setAttribute("fill", "#6b655c");
      firstLabel.setAttribute("font-size", "11");
      firstLabel.textContent = minX.toFixed(2);
      svg.appendChild(firstLabel);

      const lastLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
      lastLabel.setAttribute("x", width - margin.right - 28);
      lastLabel.setAttribute("y", height - 8);
      lastLabel.setAttribute("fill", "#6b655c");
      lastLabel.setAttribute("font-size", "11");
      lastLabel.textContent = maxX.toFixed(2);
      svg.appendChild(lastLabel);

      const topLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
      topLabel.setAttribute("x", 8);
      topLabel.setAttribute("y", margin.top + 4);
      topLabel.setAttribute("fill", "#6b655c");
      topLabel.setAttribute("font-size", "11");
      topLabel.textContent = maxY.toFixed(4);
      svg.appendChild(topLabel);

      const bottomLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
      bottomLabel.setAttribute("x", 8);
      bottomLabel.setAttribute("y", margin.top + innerH);
      bottomLabel.setAttribute("fill", "#6b655c");
      bottomLabel.setAttribute("font-size", "11");
      bottomLabel.textContent = minY.toFixed(4);
      svg.appendChild(bottomLabel);
    }}

    drawLineChart("chart-train-loss", REPORT.train_logs, "epoch", "loss", "#ab4b2f");
    drawLineChart("chart-lr", REPORT.train_logs, "epoch", "learning_rate", "#2e5a4d");
    drawLineChart("chart-val-f1", REPORT.eval_logs.filter(row => row.eval_f1 !== undefined), "epoch", "eval_f1", "#a38136");
    drawLineChart("chart-val-auc", REPORT.eval_logs.filter(row => row.eval_roc_auc !== undefined), "epoch", "eval_roc_auc", "#5c3a82");
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    stdout_path = run_dir / "stdout.log"
    metrics_path = run_dir / "metrics.json"
    if not stdout_path.exists():
        raise FileNotFoundError(f"Missing stdout.log: {stdout_path}")

    train_logs, eval_logs = parse_log_records(stdout_path)
    metrics = _safe_read_json(metrics_path)
    output_path = Path(args.output_file) if args.output_file else run_dir / "training_report.html"
    output_path.write_text(
        build_html(run_dir, metrics, train_logs, eval_logs),
        encoding="utf-8",
    )
    print(f"Generated report: {output_path}")
    print(f"Parsed train logs: {len(train_logs)}")
    print(f"Parsed eval logs: {len(eval_logs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
