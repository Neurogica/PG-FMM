"""FlowCast Table 3 metric summaries (SEVIR native protocol).

Uses FlowCast's ``MetricsAccumulator`` + ``calculate_metrics`` so numbers match
Table 3 in the FlowCast paper (CSI-M, CSI-P16-M, FSS-P16-M, HSS-M, FAR-M,
CRPS, and +65 min lead-time metrics).
"""

from __future__ import annotations

from typing import Any


def extract_table3_row(results: dict[str, Any], *, thresholds: list[float]) -> dict[str, float]:
    """Flatten ``calculate_metrics`` output into Table 3 column names."""
    last = -1
    fss_by_scale = results.get("fss_m_from_mean_by_scale") or {}
    fss_p16 = fss_by_scale.get(16)
    if fss_p16 is None and results.get("fss_m_from_mean") is not None:
        fss_p16 = results["fss_m_from_mean"]

    csi_m_lead = results.get("csi_m_from_mean_lead_time") or []
    csi_219_lead = results.get("csi_last_thresh_from_mean_lead_time") or []

    # FlowCast Table 3 reports CRPS scaled by max SEVIR VIL (255).
    crps_raw = float(results["crps_mean"])
    return {
        "crps": crps_raw / 255.0,
        "crps_raw": crps_raw,
        "csi_m": float(results["csi_from_mean_m"]),
        "csi_p16_m": float(results["csi_pool_from_mean_m"]),
        "fss_p16_m": float(fss_p16) if fss_p16 is not None else float("nan"),
        "hss_m": float(results["hss_from_mean_m"]),
        "far_m": float(results["far_from_mean_m"]),
        "csi_m_plus65": float(csi_m_lead[last]) if csi_m_lead else float("nan"),
        "csi_219_plus65": float(csi_219_lead[last]) if csi_219_lead else float("nan"),
    }


def format_table3_markdown(
    rows: list[tuple[str, dict[str, float]]],
    *,
    title: str = "Table 3 style comparison (SEVIR, 12-step forecast)",
) -> str:
    """Render a markdown table from ``(model_name, metrics)`` pairs."""
    cols = [
        ("CRPS ↓", "crps", 4),
        ("CSI-M ↑", "csi_m", 3),
        ("CSI-P16-M ↑", "csi_p16_m", 3),
        ("FSS-P16-M ↑", "fss_p16_m", 3),
        ("HSS-M ↑", "hss_m", 3),
        ("FAR-M ↓", "far_m", 3),
        ("+65m CSI-M ↑", "csi_m_plus65", 3),
        ("+65m CSI-219 ↑", "csi_219_plus65", 3),
    ]
    header = "| Model | " + " | ".join(c[0] for c in cols) + " |"
    sep = "|---|" + "|".join(["---:"] * len(cols)) + "|"
    lines = [f"## {title}", "", header, sep]
    for name, m in rows:
        cells = []
        for _, key, prec in cols:
            v = m.get(key, float("nan"))
            cells.append(f"{v:.{prec}f}" if v == v else "—")
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
