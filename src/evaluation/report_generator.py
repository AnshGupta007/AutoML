"""
HTML report generator.
Produces a comprehensive evaluation report with metrics, plots, and explanations.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoMLPro — Evaluation Report</title>
<style>
  body{{font-family:Inter,Segoe UI,sans-serif;margin:0;padding:24px;background:#0f172a;color:#e2e8f0}}
  h1{{color:#38bdf8;margin-bottom:4px}}
  h2{{color:#7dd3fc;border-bottom:1px solid #334155;padding-bottom:8px}}
  .meta{{color:#94a3b8;font-size:0.85rem;margin-bottom:32px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:16px;margin-bottom:32px}}
  .card{{background:#1e293b;border-radius:12px;padding:20px;text-align:center}}
  .card .value{{font-size:2rem;font-weight:700;color:#38bdf8}}
  .card .label{{font-size:0.8rem;color:#94a3b8;margin-top:4px}}
  table{{width:100%;border-collapse:collapse;margin-bottom:24px}}
  th{{background:#1e293b;padding:10px 14px;text-align:left;font-size:0.8rem;color:#94a3b8;text-transform:uppercase}}
  td{{padding:10px 14px;border-bottom:1px solid #1e293b;font-size:0.9rem}}
  pre{{background:#1e293b;padding:16px;border-radius:8px;overflow-x:auto;font-size:0.8rem;color:#a5f3fc}}
  .badge{{display:inline-block;padding:3px 10px;border-radius:20px;font-size:0.75rem;font-weight:600}}
  .badge-green{{background:#064e3b;color:#6ee7b7}}
  .badge-blue{{background:#0c4a6e;color:#7dd3fc}}
  img{{max-width:100%;border-radius:12px;margin:16px 0}}
</style>
</head>
<body>
<h1>🤖 AutoMLPro — Evaluation Report</h1>
<div class="meta">Generated: {timestamp} &nbsp;|&nbsp; Experiment: {experiment} &nbsp;|&nbsp; Model: <span class="badge badge-blue">{model_name}</span></div>

<h2>📊 Performance Metrics</h2>
<div class="grid">
{metric_cards}
</div>

<h2>🔍 Feature Importances</h2>
{feature_table}

<h2>⚙️ Configuration</h2>
<pre>{config_json}</pre>

{extra_sections}

</body></html>
"""


class ReportGenerator:
    """Generates an HTML evaluation report from training results.

    Example:
        >>> reporter = ReportGenerator(output_dir="reports")
        >>> path = reporter.generate(
        ...     metrics={"accuracy": 0.94, "roc_auc": 0.98},
        ...     feature_importances=fi_series,
        ...     model_name="xgboost",
        ...     config=cfg,
        ... )
    """

    def __init__(self, output_dir: Union[str, Path] = "reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        metrics: dict,
        model_name: str = "unknown",
        experiment_name: str = "automl",
        feature_importances: Optional[pd.Series] = None,
        config: Optional[dict] = None,
        extra_images: Optional[list[Path]] = None,
    ) -> Path:
        """Generate an HTML evaluation report.

        Args:
            metrics: Dict of metric_name → float.
            model_name: Name of the best model.
            experiment_name: Experiment identifier.
            feature_importances: Optional feature importance Series.
            config: Configuration dict.
            extra_images: Paths to additional images (SHAP plots etc.) to embed.

        Returns:
            Path to generated HTML report.
        """
        # Metric cards
        cards = []
        for name, value in metrics.items():
            formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
            cards.append(
                f'<div class="card"><div class="value">{formatted}</div>'
                f'<div class="label">{name}</div></div>'
            )
        metric_cards_html = "\n".join(cards)

        # Feature importance table
        if feature_importances is not None:
            top20 = feature_importances.head(20)
            rows = "".join(
                f"<tr><td>{feat}</td><td>{val:.6f}</td></tr>"
                for feat, val in top20.items()
            )
            feature_table_html = (
                "<table><thead><tr><th>Feature</th><th>Importance</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>"
            )
        else:
            feature_table_html = "<p>Feature importances not available.</p>"

        # Config
        config_json = json.dumps(config or {}, indent=2, default=str)

        # Extra images
        extra_sections = ""
        if extra_images:
            extra_sections += "<h2>📈 Visualizations</h2>"
            for img_path in extra_images:
                if Path(img_path).exists():
                    extra_sections += (
                        f'<img src="file://{img_path}" alt="{Path(img_path).name}">'
                    )

        html = HTML_TEMPLATE.format(
            timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            experiment=experiment_name,
            model_name=model_name,
            metric_cards=metric_cards_html,
            feature_table=feature_table_html,
            config_json=config_json,
            extra_sections=extra_sections,
        )

        report_path = self.output_dir / f"evaluation_report_{datetime.utcnow():%Y%m%d_%H%M%S}.html"
        report_path.write_text(html, encoding="utf-8")
        log.info(f"Evaluation report saved to {report_path}")
        return report_path
