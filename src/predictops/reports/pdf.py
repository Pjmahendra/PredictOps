"""PDF maintenance report for one prediction run (reportlab).

Every number in the report comes from the stored run, its predictions, alerts and the
SHAP explanation of the worst minute — nothing is typed in by hand.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

NAVY = colors.HexColor("#111a2b"); BLUE = colors.HexColor("#2a78d6"); GREY = colors.HexColor("#52514e")
STATUS = {"Healthy": "#0ca30c", "Warning": "#c98500", "Critical": "#d03b3b", "Low": "#0ca30c", "Medium": "#c98500", "High": "#d03b3b"}


def _tbl(data, widths=None, header=True):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", 8.5), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8ccd4")), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5)]
    t.setStyle(TableStyle(st))
    return t


def build_run_report(run: dict, preds: pd.DataFrame, alerts: list[dict], sensors: pd.DataFrame,
                     explanation: dict | None, equipment: str) -> bytes:
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=NAVY, fontSize=17, alignment=0)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=BLUE, fontSize=12, spaceBefore=10)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, textColor=GREY)
    P = lambda t, s=body: Paragraph(t, s)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title=f"PredictOps report run {run['id']}", author="PredictOps")
    worst = preds.loc[preds.health_score.idxmin()]
    last = preds.iloc[-1]
    E = []
    E += [P("PredictOps — Equipment Health Report", h1),
          P(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · Run #{run['id']} ({run['source']}) · "
            f"<b>historical data analysis, not live monitoring</b>", small), Spacer(1, 6)]
    E += [P("1. Equipment and period", h2), _tbl([
        ["Equipment", equipment], ["Period assessed", f"{run['start_time']} → {run['end_time']} ({run['n_minutes']:,} minutes)"],
        ["Failure-signature model", run["model_version_detector"]], ["Anomaly model", run["model_version_anomaly"]]],
        widths=[45 * mm, 133 * mm], header=False)]

    sc = preds.health_status.value_counts()
    E += [P("2. Equipment health assessment", h2), _tbl([
        ["Indicator", "Value"],
        ["Latest minute", f"{last.timestamp:%Y-%m-%d %H:%M} — health {last.health_score:.0f} ({last.health_status}), risk {last.failure_risk}"],
        ["Worst minute", f"{worst.timestamp:%Y-%m-%d %H:%M} — health {worst.health_score:.0f} ({worst.health_status}), risk {worst.failure_risk}"],
        ["Mean health score", f"{preds.health_score.mean():.1f}"],
        ["Time per status", " · ".join(f"{k}: {sc.get(k, 0) / len(preds):.1%}" for k in ["Healthy", "Warning", "Critical"])],
        ["Anomalous minutes (persistent)", f"{int((preds.anomaly_status == 'Anomalous').sum())}"],
        ["High-risk minutes", f"{int((preds.failure_risk == 'High').sum())}"]], widths=[50 * mm, 128 * mm])]
    E += [P("Health score is a heuristic index (0–100), not a failure probability.", small)]

    E += [P("3. Sensor summary", h2)]
    if not sensors.empty:
        d = sensors.describe().T[["mean", "min", "max"]].round(2)
        E += [_tbl([["Sensor", "Mean", "Min", "Max"]] + [[i, *map(str, r)] for i, r in d.iterrows()], widths=[50 * mm, 40 * mm, 40 * mm, 40 * mm])]

    E += [P("4. Anomaly alerts and failure risk", h2)]
    if alerts:
        rows = [["Start", "End", "Min", "Severity", "Status", "Ack."]]
        rows += [[a["start_time"][:16].replace("T", " "), a["end_time"][11:16], str(a["minutes"]), a["severity"], a["status"],
                  "yes" if a["acknowledged"] else "no"] for a in alerts]
        E += [_tbl(rows, widths=[38 * mm, 20 * mm, 15 * mm, 25 * mm, 25 * mm, 15 * mm])]
    else:
        E += [P("No persistent anomaly alerts in this period.")]

    E += [P("5. Explanation of the worst minute", h2), P(f"<b>Reasons</b> ({worst.timestamp:%Y-%m-%d %H:%M}):")]
    E += [P("• " + r) for r in worst.reasons]
    if explanation:
        rows = [["Feature (failure-signature model)", "Value", "SHAP (log-odds)"]]
        rows += [[c["feature"], f"{c['value']:.3f}", f"{c['shap']:+.3f}"] for c in explanation["detector_shap"]["contributions"]]
        E += [Spacer(1, 4), _tbl(rows, widths=[90 * mm, 40 * mm, 40 * mm])]
        rows = [["Most deviating feature (anomaly model)", "Robust z"]]
        rows += [[d["feature"], f"{d['robust_z']:+.2f}"] for d in explanation["anomaly_top_deviations"]]
        E += [Spacer(1, 4), _tbl(rows, widths=[130 * mm, 40 * mm])]
        E += [P("SHAP shows what the model relied on; it is not proof of the physical cause.", small)]

    E += [P("6. Maintenance recommendations", h2)]
    recs = list(dict.fromkeys(r for lst in preds[preds.health_status != "Healthy"].recommendations for r in lst)) or \
        ["Continue normal monitoring."]
    E += [P("• " + r) for r in recs]
    E += [Spacer(1, 10), P("Limitations: single compressor; 4 reported failures (air leaks); the system detects ongoing abnormal "
                           "behaviour and does not predict failures hours in advance. See project documentation docs/05–07.", small)]
    doc.build(E)
    return buf.getvalue()
