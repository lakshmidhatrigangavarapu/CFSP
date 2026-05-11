"""
Professional PDF Report Generator using WeasyPrint
===================================================
Generates beautiful, modern clinical reports using HTML/CSS → PDF

Requirements:
    pip install weasyprint

Features:
- Modern card-based design
- Color-coded risk indicators
- Clean typography (system fonts)
- Professional tables with borders
- Responsive layout
- Easy to customize via CSS
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import html

log = logging.getLogger(__name__)

# WeasyPrint requires system libraries (pango, gobject, etc.)
# If not available, this module will raise ImportError when imported
HAS_WEASYPRINT = False
HTML = None
CSS = None

try:
    from weasyprint import HTML, CSS
    HAS_WEASYPRINT = True
except (ImportError, OSError) as e:
    # Re-raise as ImportError so the calling code can handle it
    raise ImportError(f"WeasyPrint not available: {e}") from e


# ============================================================================
# CSS STYLES
# ============================================================================

REPORT_CSS = """
@page {
    size: A4;
    margin: 1.5cm 1.5cm 2cm 1.5cm;
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9px;
        color: #666;
    }
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.5;
    color: #1a1a1a;
    background: white;
}

/* ===== HEADER ===== */
.report-header {
    text-align: center;
    padding-bottom: 20px;
    border-bottom: 3px solid #1a365d;
    margin-bottom: 25px;
}

.report-title {
    font-size: 26pt;
    font-weight: 700;
    color: #1a365d;
    margin-bottom: 5px;
    letter-spacing: -0.5px;
}

.report-subtitle {
    font-size: 11pt;
    color: #4a5568;
    font-weight: 400;
}

.report-meta {
    font-size: 9pt;
    color: #718096;
    margin-top: 12px;
}

/* ===== RISK TIER BOX ===== */
.risk-tier-section {
    text-align: center;
    margin: 25px 0;
}

.risk-tier-label {
    font-size: 11pt;
    color: #4a5568;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.risk-tier-box {
    display: inline-block;
    padding: 12px 40px;
    border-radius: 8px;
    font-size: 20pt;
    font-weight: 700;
    color: white;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.risk-high { background: linear-gradient(135deg, #dc3545, #c82333); }
.risk-moderate { background: linear-gradient(135deg, #fd7e14, #e8590c); }
.risk-low { background: linear-gradient(135deg, #28a745, #218838); }
.risk-unknown { background: linear-gradient(135deg, #6c757d, #5a6268); }

.risk-justification {
    font-size: 9pt;
    color: #4a5568;
    font-style: italic;
    margin-top: 10px;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}

/* ===== METRICS DASHBOARD ===== */
.metrics-grid {
    display: flex;
    justify-content: space-between;
    gap: 15px;
    margin: 25px 0;
}

.metric-card {
    flex: 1;
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 15px;
    text-align: center;
}

.metric-value {
    font-size: 22pt;
    font-weight: 700;
    color: #1a365d;
}

.metric-label {
    font-size: 8pt;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* ===== SECTION HEADERS ===== */
.section-header {
    font-size: 14pt;
    font-weight: 700;
    color: #1a365d;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 8px;
    margin: 30px 0 15px 0;
    page-break-after: avoid;
}

.section-header .section-number {
    color: #4299e1;
    margin-right: 8px;
}

.subsection-header {
    font-size: 11pt;
    font-weight: 600;
    color: #2d3748;
    margin: 20px 0 10px 0;
}

/* ===== ALERT BOX ===== */
.alert-box {
    background: #fff5f5;
    border: 2px solid #fc8181;
    border-left: 5px solid #e53e3e;
    border-radius: 6px;
    padding: 15px 20px;
    margin: 20px 0;
}

.alert-title {
    font-size: 11pt;
    font-weight: 700;
    color: #c53030;
    margin-bottom: 10px;
}

.alert-item {
    font-size: 9pt;
    color: #742a2a;
    margin: 6px 0;
    padding-left: 15px;
    position: relative;
}

.alert-item::before {
    content: "⚠";
    position: absolute;
    left: 0;
}

.alert-category {
    font-weight: 600;
    color: #c53030;
    margin-top: 10px;
    margin-bottom: 5px;
    font-size: 9pt;
}

/* ===== FACTORS LIST ===== */
.factors-list {
    list-style: none;
    padding: 0;
}

.factors-list li {
    padding: 8px 0;
    border-bottom: 1px solid #edf2f7;
    font-size: 10pt;
}

.factors-list li:last-child {
    border-bottom: none;
}

.severity-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 8pt;
    font-weight: 600;
    margin-right: 8px;
}

.severity-high { background: #fed7d7; color: #c53030; }
.severity-moderate { background: #feebc8; color: #c05621; }
.severity-low { background: #c6f6d5; color: #276749; }

.protective-icon {
    color: #38a169;
    margin-right: 6px;
}

/* ===== EVIDENCE TABLE ===== */
.evidence-table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    font-size: 9pt;
}

.evidence-table th {
    background: #1a365d;
    color: white;
    padding: 10px 8px;
    text-align: left;
    font-weight: 600;
    font-size: 8pt;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.evidence-table td {
    padding: 8px;
    border-bottom: 1px solid #e2e8f0;
    vertical-align: top;
}

.evidence-table tr:nth-child(even) {
    background: #f7fafc;
}

.evidence-table tr:hover {
    background: #edf2f7;
}

.branch-indicator {
    display: inline-block;
    width: 24px;
    height: 24px;
    border-radius: 4px;
    color: white;
    font-weight: 700;
    text-align: center;
    line-height: 24px;
    font-size: 10pt;
}

.branch-a { background: #dc3545; }
.branch-b { background: #fd7e14; }
.branch-c { background: #0d6efd; }
.branch-nexus { background: #ffc107; color: #1a1a1a; }

.evidence-text {
    font-style: italic;
    color: #4a5568;
}

/* ===== SCENARIO CARDS ===== */
.scenario-card {
    border: 2px solid #e2e8f0;
    border-radius: 10px;
    margin: 20px 0;
    overflow: hidden;
    page-break-inside: avoid;
}

.scenario-header {
    padding: 12px 18px;
    color: white;
    font-weight: 700;
    font-size: 11pt;
}

.scenario-header-a { background: linear-gradient(135deg, #dc3545, #c82333); }
.scenario-header-b { background: linear-gradient(135deg, #fd7e14, #e8590c); }
.scenario-header-c { background: linear-gradient(135deg, #0d6efd, #0b5ed7); }

.scenario-body {
    padding: 18px;
}

.scenario-body-a { background: #fff5f5; }
.scenario-body-b { background: #fffaf0; }
.scenario-body-c { background: #f0f7ff; }

.scenario-gated {
    opacity: 0.6;
}

.scenario-gated .scenario-header {
    background: #6c757d !important;
}

.scenario-gated .scenario-body {
    background: #f8f9fa !important;
}

.plausibility-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 9pt;
    font-weight: 600;
    margin-bottom: 12px;
}

.plausibility-high { background: #fed7d7; color: #c53030; }
.plausibility-moderate { background: #feebc8; color: #c05621; }
.plausibility-low { background: #c6f6d5; color: #276749; }

.scenario-section-title {
    font-weight: 600;
    font-size: 9pt;
    color: #2d3748;
    margin: 12px 0 6px 0;
}

.scenario-section-title:first-child {
    margin-top: 0;
}

.warning-signs-list, .actions-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.warning-signs-list li, .actions-list li {
    font-size: 9pt;
    padding: 4px 0;
    padding-left: 18px;
    position: relative;
}

.warning-signs-list li::before {
    content: "⚡";
    position: absolute;
    left: 0;
}

.actions-list li::before {
    content: "→";
    position: absolute;
    left: 0;
    color: #4299e1;
    font-weight: 700;
}

.crisis-endpoint {
    font-size: 9pt;
    background: #fff5f5;
    border-left: 3px solid #e53e3e;
    padding: 8px 12px;
    margin: 10px 0;
    color: #742a2a;
}

.narrative-text {
    font-size: 9pt;
    color: #4a5568;
    font-style: italic;
    border-left: 3px solid #cbd5e0;
    padding-left: 12px;
    margin-top: 12px;
}

/* ===== PRIORITY ACTIONS ===== */
.priority-actions {
    background: #f0fff4;
    border: 2px solid #9ae6b4;
    border-radius: 8px;
    padding: 18px;
    margin: 20px 0;
}

.priority-actions-title {
    font-size: 11pt;
    font-weight: 700;
    color: #276749;
    margin-bottom: 12px;
}

.priority-actions ol {
    padding-left: 25px;
    margin: 0;
}

.priority-actions li {
    font-size: 10pt;
    padding: 6px 0;
    color: #1a1a1a;
}

.priority-actions li:nth-child(-n+3) {
    font-weight: 600;
}

/* ===== CLINICAL NOTE ===== */
.clinical-note {
    background: #fafafa;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 18px;
    margin: 15px 0;
    font-size: 9pt;
    line-height: 1.6;
    white-space: pre-wrap;
    word-wrap: break-word;
}

.highlight-a { background: rgba(220, 53, 69, 0.25); border-radius: 2px; }
.highlight-b { background: rgba(253, 126, 20, 0.25); border-radius: 2px; }
.highlight-c { background: rgba(13, 110, 253, 0.25); border-radius: 2px; }
.highlight-nexus { background: rgba(255, 193, 7, 0.35); border-radius: 2px; }

.branch-tag {
    font-size: 7pt;
    font-weight: 700;
    vertical-align: super;
    color: #666;
}

/* ===== COLOR LEGEND ===== */
.color-legend {
    display: flex;
    justify-content: center;
    gap: 20px;
    padding: 12px;
    background: #f7fafc;
    border-radius: 6px;
    margin: 15px 0;
    font-size: 8pt;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
}

.legend-color {
    width: 16px;
    height: 16px;
    border-radius: 3px;
}

.legend-color-a { background: #dc3545; }
.legend-color-b { background: #fd7e14; }
.legend-color-c { background: #0d6efd; }
.legend-color-nexus { background: #ffc107; }

/* ===== PAGE BREAKS ===== */
.page-break {
    page-break-before: always;
}

/* ===== FOOTER ===== */
.report-footer {
    margin-top: 40px;
    padding-top: 15px;
    border-top: 1px solid #e2e8f0;
    text-align: center;
    font-size: 8pt;
    color: #718096;
}
"""


# ============================================================================
# HTML TEMPLATE BUILDERS
# ============================================================================

def _escape(text: str) -> str:
    """Escape HTML special characters."""
    if not text:
        return ""
    return html.escape(str(text))


def _get_risk_class(risk_tier: str) -> str:
    """Get CSS class for risk tier."""
    risk = risk_tier.upper().strip()
    if "HIGH" in risk:
        return "risk-high"
    elif "MODERATE" in risk or "MEDIUM" in risk:
        return "risk-moderate"
    elif "LOW" in risk:
        return "risk-low"
    return "risk-unknown"


def _get_severity(text: str) -> str:
    """Estimate severity from text content."""
    text_lower = text.lower()
    high_keywords = ["suicid", "homicid", "overdose", "acute", "severe", "psychosis", "mania", "imminent"]
    moderate_keywords = ["substance", "alcohol", "drug", "depression", "anxiety", "history", "chronic"]
    
    if any(kw in text_lower for kw in high_keywords):
        return "high"
    elif any(kw in text_lower for kw in moderate_keywords):
        return "moderate"
    return "low"


def _build_header(silver_labels: dict, report: dict) -> str:
    """Build report header HTML."""
    dx = silver_labels.get("primary_mh_diagnosis", {})
    dx_title = dx.get("title", "Unknown") if isinstance(dx, dict) else "Unknown"
    generated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    risk_tier = report.get("overall_risk_tier", "UNKNOWN").upper().strip()
    risk_class = _get_risk_class(risk_tier)
    justification = report.get("risk_tier_justification", "")
    
    return f"""
    <div class="report-header">
        <h1 class="report-title">Clinical Preparedness Report</h1>
        <p class="report-subtitle">Counterfactual Risk Assessment with Evidence Attribution</p>
        <p class="report-meta">Generated: {generated} • Primary Dx: {_escape(dx_title)}</p>
    </div>
    
    <div class="risk-tier-section">
        <div class="risk-tier-label">Overall Risk Assessment</div>
        <div class="risk-tier-box {risk_class}">{_escape(risk_tier)}</div>
        {f'<p class="risk-justification">{_escape(justification)}</p>' if justification else ''}
    </div>
    """


def _build_metrics(evidence_result: dict, scenarios_result: dict, report: dict) -> str:
    """Build metrics dashboard HTML."""
    spans = evidence_result.get("evidence_spans", [])
    coverage = evidence_result.get("coverage_stats", {})
    nexus_count = evidence_result.get("nexus_count", 0)
    
    scenarios = scenarios_result.get("scenarios", [])
    active = sum(1 for s in scenarios if not s.get("gated"))
    
    return f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{len(spans)}</div>
            <div class="metric-label">Evidence Spans</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{nexus_count}</div>
            <div class="metric-label">Nexus Factors</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{coverage.get('coverage_pct', 0)}%</div>
            <div class="metric-label">Note Coverage</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{active}/{len(scenarios)}</div>
            <div class="metric-label">Active Scenarios</div>
        </div>
    </div>
    """


def _build_alerts(critical_signals: dict, report: dict) -> str:
    """Build critical alerts HTML."""
    all_alerts = []
    if critical_signals and critical_signals.get("signals_found"):
        all_alerts.extend(critical_signals.get("priority_alerts", []))
    all_alerts.extend(report.get("priority_alerts", []))
    
    # Deduplicate
    seen = set()
    unique_alerts = []
    for a in all_alerts:
        a_lower = a.lower().strip()
        if a_lower not in seen:
            seen.add(a_lower)
            unique_alerts.append(a)
    
    if not unique_alerts:
        return ""
    
    # Group alerts
    suicide, substance, violence, other = [], [], [], []
    for alert in unique_alerts:
        alert_lower = alert.lower()
        if any(kw in alert_lower for kw in ["suicid", "self-harm", "si/hi"]):
            suicide.append(alert)
        elif any(kw in alert_lower for kw in ["substance", "drug", "alcohol", "overdose"]):
            substance.append(alert)
        elif any(kw in alert_lower for kw in ["violen", "homicid", "assault"]):
            violence.append(alert)
        else:
            other.append(alert)
    
    html_parts = ['<div class="alert-box">', '<div class="alert-title">⚠ Critical Safety Alerts</div>']
    
    if suicide:
        html_parts.append('<div class="alert-category">Suicide/Self-Harm Risk:</div>')
        for a in suicide[:3]:
            html_parts.append(f'<div class="alert-item">{_escape(a)}</div>')
    
    if substance:
        html_parts.append('<div class="alert-category">Substance Use Concern:</div>')
        for a in substance[:3]:
            html_parts.append(f'<div class="alert-item">{_escape(a)}</div>')
    
    if violence:
        html_parts.append('<div class="alert-category">Violence/Aggression Risk:</div>')
        for a in violence[:3]:
            html_parts.append(f'<div class="alert-item">{_escape(a)}</div>')
    
    if other:
        html_parts.append('<div class="alert-category">Other Concerns:</div>')
        for a in other[:3]:
            html_parts.append(f'<div class="alert-item">{_escape(a)}</div>')
    
    html_parts.append('</div>')
    return '\n'.join(html_parts)


def _build_factors(report: dict) -> str:
    """Build risk and protective factors HTML."""
    risk_factors = report.get("key_risk_factors", [])
    protective = report.get("protective_factors", [])
    
    html_parts = []
    
    if risk_factors:
        html_parts.append('<h2 class="section-header"><span class="section-number">1.</span>Key Risk Factors</h2>')
        html_parts.append('<ul class="factors-list">')
        for rf in risk_factors:
            severity = _get_severity(rf)
            html_parts.append(f'''
                <li>
                    <span class="severity-badge severity-{severity}">{severity.upper()}</span>
                    {_escape(rf)}
                </li>
            ''')
        html_parts.append('</ul>')
    
    if protective:
        html_parts.append('<h2 class="section-header"><span class="section-number">2.</span>Protective Factors</h2>')
        html_parts.append('<ul class="factors-list">')
        for pf in protective:
            html_parts.append(f'<li><span class="protective-icon">✓</span>{_escape(pf)}</li>')
        html_parts.append('</ul>')
    
    return '\n'.join(html_parts)


def _build_evidence_table(evidence_result: dict) -> str:
    """Build evidence attribution table HTML."""
    spans = evidence_result.get("evidence_spans", [])
    
    if not spans:
        return ""
    
    html_parts = [
        '<h2 class="section-header"><span class="section-number">3.</span>Evidence Attribution</h2>',
        '<div class="color-legend">',
        '<div class="legend-item"><div class="legend-color legend-color-a"></div>Branch A: Psychiatric</div>',
        '<div class="legend-item"><div class="legend-color legend-color-b"></div>Branch B: Substance Use</div>',
        '<div class="legend-item"><div class="legend-color legend-color-c"></div>Branch C: Social</div>',
        '<div class="legend-item"><div class="legend-color legend-color-nexus"></div>Nexus: Multi-Branch</div>',
        '</div>',
        '<table class="evidence-table">',
        '<tr><th>ID</th><th>Risk Factor</th><th>Evidence from Note</th><th>Branch</th><th>Severity</th></tr>'
    ]
    
    for idx, span in enumerate(spans[:20], 1):
        factors = span.get("factors", [])
        factors_str = "; ".join(factors[:2])[:45]
        if len(factors_str) > 45:
            factors_str += "..."
        
        evidence = span.get("text", "")[:75]
        if len(evidence) > 75:
            evidence += "..."
        
        branches = span.get("branches", [])
        branch_str = ", ".join(branches)
        color_key = span.get("color_key", "A").lower()
        if color_key not in ["a", "b", "c", "nexus"]:
            color_key = "a"
        
        severity = _get_severity(" ".join(factors))
        
        html_parts.append(f'''
            <tr>
                <td><span class="branch-indicator branch-{color_key}">{idx}</span></td>
                <td>{_escape(factors_str)}</td>
                <td class="evidence-text">"{_escape(evidence)}"</td>
                <td><strong>{_escape(branch_str)}</strong></td>
                <td><span class="severity-badge severity-{severity}">{severity.upper()}</span></td>
            </tr>
        ''')
    
    html_parts.append('</table>')
    
    if len(spans) > 20:
        html_parts.append(f'<p style="font-size: 8pt; color: #666; margin-top: 8px;">Showing 20 of {len(spans)} evidence spans.</p>')
    
    return '\n'.join(html_parts)


def _build_scenario_card(scenario: dict) -> str:
    """Build a single scenario card HTML."""
    branch_id = scenario.get("scenario_id", "?")
    branch_label = scenario.get("branch_label", "Unknown")
    is_gated = scenario.get("gated", False)
    
    branch_key = branch_id.lower() if branch_id in ["A", "B", "C"] else "a"
    gated_class = " scenario-gated" if is_gated else ""
    
    if is_gated:
        return f'''
        <div class="scenario-card{gated_class}">
            <div class="scenario-header scenario-header-{branch_key}">
                [{branch_id}] {_escape(branch_label)}
            </div>
            <div class="scenario-body scenario-body-{branch_key}">
                <p><strong>STATUS: NOT APPLICABLE</strong></p>
                <p style="font-style: italic; margin-top: 8px;">{_escape(scenario.get('gate_reason', 'Insufficient evidence'))}</p>
            </div>
        </div>
        '''
    
    scenario_obj = scenario.get("scenario", {})
    plausibility = scenario_obj.get("plausibility", "?").upper()
    plaus_class = "high" if plausibility == "HIGH" else ("moderate" if plausibility == "MODERATE" else "low")
    
    warning_signs = scenario_obj.get("warning_signs", [])
    crisis = scenario_obj.get("crisis_endpoint", "")
    actions = scenario_obj.get("preparedness_actions", [])
    narrative = scenario_obj.get("narrative", "")
    plaus_rationale = scenario_obj.get("plausibility_rationale", "")
    
    html_parts = [f'''
        <div class="scenario-card">
            <div class="scenario-header scenario-header-{branch_key}">
                [{branch_id}] {_escape(branch_label)}
            </div>
            <div class="scenario-body scenario-body-{branch_key}">
                <span class="plausibility-badge plausibility-{plaus_class}">Plausibility: {plausibility}</span>
    ''']
    
    if plaus_rationale:
        html_parts.append(f'<p style="font-size: 9pt; color: #4a5568;">{_escape(plaus_rationale)}</p>')
    
    if warning_signs:
        html_parts.append('<div class="scenario-section-title">Warning Signs</div>')
        html_parts.append('<ul class="warning-signs-list">')
        for ws in warning_signs[:5]:
            html_parts.append(f'<li>{_escape(ws)}</li>')
        html_parts.append('</ul>')
    
    if crisis:
        html_parts.append(f'<div class="scenario-section-title">Crisis Endpoint</div>')
        html_parts.append(f'<div class="crisis-endpoint">{_escape(crisis)}</div>')
    
    if actions:
        html_parts.append('<div class="scenario-section-title">Preparedness Actions</div>')
        html_parts.append('<ul class="actions-list">')
        for a in actions[:5]:
            html_parts.append(f'<li>{_escape(a)}</li>')
        html_parts.append('</ul>')
    
    if narrative:
        narrative_preview = narrative[:300] + "..." if len(narrative) > 300 else narrative
        html_parts.append(f'<div class="narrative-text">{_escape(narrative_preview)}</div>')
    
    html_parts.append('</div></div>')
    
    return '\n'.join(html_parts)


def _build_scenarios(scenarios_result: dict) -> str:
    """Build all scenario cards HTML."""
    scenarios = scenarios_result.get("scenarios", [])
    
    if not scenarios:
        return ""
    
    html_parts = [
        '<div class="page-break"></div>',
        '<h2 class="section-header"><span class="section-number">4.</span>Counterfactual Scenario Briefing</h2>',
        '<p style="font-size: 9pt; color: #666; margin-bottom: 15px;">Each card represents a potential deterioration pathway. Content is preserved exactly as generated.</p>'
    ]
    
    for scenario in scenarios:
        html_parts.append(_build_scenario_card(scenario))
    
    return '\n'.join(html_parts)


def _build_priority_actions(report: dict) -> str:
    """Build priority actions HTML."""
    actions = report.get("priority_actions", [])
    
    if not actions:
        return ""
    
    html_parts = [
        '<h2 class="section-header"><span class="section-number">5.</span>Priority Preparedness Actions</h2>',
        '<div class="priority-actions">',
        '<div class="priority-actions-title">Immediate Clinical Interventions</div>',
        '<ol>'
    ]
    
    for action in actions:
        html_parts.append(f'<li>{_escape(action)}</li>')
    
    html_parts.extend(['</ol>', '</div>'])
    
    return '\n'.join(html_parts)


def _build_patient_overview(report: dict) -> str:
    """Build patient overview and mental health status HTML."""
    overview = report.get("patient_overview", "")
    mh_status = report.get("mental_health_status", "")
    
    html_parts = []
    
    if overview:
        html_parts.append('<h2 class="section-header"><span class="section-number">6.</span>Patient Overview</h2>')
        html_parts.append(f'<p style="font-size: 10pt; line-height: 1.6;">{_escape(overview)}</p>')
    
    if mh_status:
        html_parts.append('<h2 class="section-header"><span class="section-number">7.</span>Mental Health Status</h2>')
        html_parts.append(f'<p style="font-size: 10pt; line-height: 1.6;">{_escape(mh_status)}</p>')
    
    return '\n'.join(html_parts)


def _build_clinical_note(note_text: str, evidence_result: dict) -> str:
    """Build highlighted clinical note HTML."""
    if not note_text:
        return ""
    
    spans = evidence_result.get("evidence_spans", [])
    
    # Build highlight regions
    regions = []
    for s in spans:
        start = s.get("start", 0)
        end = s.get("end", 0)
        color_key = s.get("color_key", "A").lower()
        branches = s.get("branches", [])
        if start < end <= len(note_text):
            regions.append((start, end, color_key, branches))
    
    # Sort and clean overlaps
    regions.sort(key=lambda x: x[0])
    cleaned = []
    last_end = 0
    for start, end, color_key, branches in regions:
        if start >= last_end:
            cleaned.append((start, end, color_key, branches))
            last_end = end
    
    # Build highlighted text
    if not cleaned:
        highlighted = _escape(note_text)
    else:
        parts = []
        pos = 0
        for start, end, color_key, branches in cleaned:
            if pos < start:
                parts.append(_escape(note_text[pos:start]))
            
            hl_text = _escape(note_text[start:end])
            branch_str = ",".join(branches)
            parts.append(f'<span class="highlight-{color_key}">{hl_text}</span><span class="branch-tag">[{branch_str}]</span>')
            pos = end
        
        if pos < len(note_text):
            parts.append(_escape(note_text[pos:]))
        
        highlighted = ''.join(parts)
    
    return f'''
    <div class="page-break"></div>
    <h2 class="section-header"><span class="section-number">8.</span>Clinical Note with Evidence Highlights</h2>
    <div class="color-legend">
        <div class="legend-item"><div class="legend-color legend-color-a"></div>Branch A</div>
        <div class="legend-item"><div class="legend-color legend-color-b"></div>Branch B</div>
        <div class="legend-item"><div class="legend-color legend-color-c"></div>Branch C</div>
        <div class="legend-item"><div class="legend-color legend-color-nexus"></div>Nexus</div>
    </div>
    <div class="clinical-note">{highlighted}</div>
    '''


def _build_footer() -> str:
    """Build report footer HTML."""
    return f'''
    <div class="report-footer">
        Clinical Preparedness Pipeline V4 • Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    '''


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_pdf_report_html(
    note_text: str,
    evidence_result: dict,
    report: dict,
    silver_labels: dict,
    scenarios_result: dict,
    critical_signals: dict = None,
    output_path: Optional[str] = None,
    subject_id: str = "unknown",
    hadm_id: str = "unknown",
) -> Optional[str]:
    """
    Generate a professional PDF report using HTML/CSS and WeasyPrint.
    
    Args:
        note_text: Original clinical note text.
        evidence_result: Phase 4 output (evidence attribution).
        report: Phase 3 output (preparedness report).
        silver_labels: Phase 1 output (extracted factors).
        scenarios_result: Phase 2 output (scenarios).
        critical_signals: Phase 1.6 output (critical signals).
        output_path: Where to save the .pdf file.
        subject_id: Patient subject ID for filename.
        hadm_id: Admission ID for filename.
    
    Returns:
        Path to the generated .pdf file, or None if WeasyPrint is missing.
    """
    if not HAS_WEASYPRINT:
        log.warning("WeasyPrint not installed. Install with: pip install weasyprint")
        return None
    
    if critical_signals is None:
        critical_signals = {}
    
    log.info("Generating professional PDF report using WeasyPrint...")
    
    # Build HTML document
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Clinical Preparedness Report</title>
    </head>
    <body>
        {_build_header(silver_labels, report)}
        {_build_metrics(evidence_result, scenarios_result, report)}
        {_build_alerts(critical_signals, report)}
        {_build_factors(report)}
        {_build_evidence_table(evidence_result)}
        {_build_scenarios(scenarios_result)}
        {_build_priority_actions(report)}
        {_build_patient_overview(report)}
        {_build_clinical_note(note_text, evidence_result)}
        {_build_footer()}
    </body>
    </html>
    """
    
    # Output path
    if output_path is None:
        output_path = f"Clinical_Report_{subject_id}_{hadm_id}.pdf"
    
    if not output_path.endswith('.pdf'):
        output_path = output_path.rsplit('.', 1)[0] + '.pdf'
    
    # Generate PDF
    try:
        html_doc = HTML(string=html_content)
        css = CSS(string=REPORT_CSS)
        html_doc.write_pdf(output_path, stylesheets=[css])
        log.info(f"PDF report saved → {output_path}")
        return output_path
    except Exception as e:
        log.error(f"PDF generation failed: {e}")
        return None


# For backwards compatibility
generate_pdf_report = generate_pdf_report_html
