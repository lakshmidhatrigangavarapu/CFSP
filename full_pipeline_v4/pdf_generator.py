"""
Phase 4.5: Professional PDF Report Generator (V2)
===================================================
Generates a professional, scannable clinical preparedness PDF with:

  DOCUMENT STRUCTURE:
  ==================
  Page 1:   Executive Summary (Risk Tier, Key Metrics, Top 3 Risks)
            - Prominent boxed risk tier (HIGH/MODERATE/LOW)
            - Evidence summary mini-table
            - Critical safety alerts (grouped & deduplicated)

  Page 2:   Evidence Attribution Table
            - Clean table: ID | Factor | Evidence | Branch | Severity
            - Color-coded by branch with legend

  Page 3-4: Clinical Note with Highlighted Evidence
            - Original text preserved
            - Branch markers [A], [B], [C] with subtle highlighting

  Page 5+:  Counterfactual Scenario Briefing (Card Layout)
            - Each scenario as a visual "card" block
            - Plausibility, Warning Signs, Crisis Endpoint, Actions
            - Color-coded headers

  Final:    Priority Actions & Risk Justification
            - Numbered action list
            - Risk tier justification

  COLOR SCHEME:
  ============
  Red    (#DC3545) — Branch A: Psychiatric/Clinical Deterioration (Suicide Risk)
  Orange (#FD7E14) — Branch B: Substance Use Escalation
  Blue   (#0D6EFD) — Branch C: Social/Environmental Collapse
  Yellow (#FFC107) — Nexus Factor: supports 2+ branches (highest priority)
  Green  (#198754) — Protective factors / Low risk indicators

  DESIGN PRINCIPLES:
  ==================
  - Scannable within 30-60 seconds
  - Clear visual hierarchy (Title > Section > Content)
  - Card-based layouts for scenarios
  - Minimal clutter, maximum readability
  - Professional clinical document style

Font: Poppins (auto-downloaded from Google Fonts if not present)
Requires: reportlab (pip install reportlab)
"""

import logging
import os
import re
import urllib.request
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
    from reportlab.lib.pagesizes import A4, LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch, cm, mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, ListFlowable, ListItem, KeepTogether, Flowable
    )
    from reportlab.platypus.flowables import HRFlowable
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.graphics import renderPDF
    HAS_REPORTLAB = True
except Exception as _reportlab_err:
    HAS_REPORTLAB = False
    log.warning(f"reportlab import failed: {_reportlab_err}")


# ============================================================================
# COLOR DEFINITIONS (Modern Clinical Palette)
# ============================================================================

if HAS_REPORTLAB:
    # Branch colors with semantic meaning
    BRANCH_COLORS = {
        "A": {
            "color": colors.HexColor("#DC3545"),
            "hex": "#DC3545",
            "bg_light": "#FFEBEE",
            "name": "Red",
            "label": "Psychiatric / Clinical Deterioration",
            "icon": "⚠",
            "risk_type": "Suicide Risk"
        },
        "B": {
            "color": colors.HexColor("#FD7E14"),
            "hex": "#FD7E14",
            "bg_light": "#FFF3E0",
            "name": "Orange",
            "label": "Substance Use Escalation",
            "icon": "⚗",
            "risk_type": "Substance Use"
        },
        "C": {
            "color": colors.HexColor("#0D6EFD"),
            "hex": "#0D6EFD",
            "bg_light": "#E3F2FD",
            "name": "Blue",
            "label": "Social / Environmental Collapse",
            "icon": "🏠",
            "risk_type": "Social Risk"
        },
        "nexus": {
            "color": colors.HexColor("#FFC107"),
            "hex": "#FFC107",
            "bg_light": "#FFF8E1",
            "name": "Yellow",
            "label": "Nexus Factor (Multi-Branch)",
            "icon": "★",
            "risk_type": "Multi-Factor"
        },
    }
    
    # Severity colors
    SEVERITY_COLORS = {
        "HIGH": colors.HexColor("#DC3545"),
        "MODERATE": colors.HexColor("#FD7E14"),
        "LOW": colors.HexColor("#198754"),
    }
    
    # UI Colors
    HEADER_BLUE = colors.HexColor("#1A365D")
    HEADER_LIGHT = colors.HexColor("#2C5282")
    ACCENT_BLUE = colors.HexColor("#3182CE")
    
    LIGHT_GRAY = colors.HexColor("#F7FAFC")
    MEDIUM_GRAY = colors.HexColor("#E2E8F0")
    DARK_GRAY = colors.HexColor("#4A5568")
    
    ALERT_RED = colors.HexColor("#C53030")
    ALERT_BG = colors.HexColor("#FED7D7")
    
    SUCCESS_GREEN = colors.HexColor("#198754")
    SUCCESS_BG = colors.HexColor("#D4EDDA")
    
    WARNING_ORANGE = colors.HexColor("#ED8936")
    WARNING_BG = colors.HexColor("#FEEBC8")
    
    # Card backgrounds
    CARD_BG = colors.HexColor("#FFFFFF")
    CARD_BORDER = colors.HexColor("#CBD5E0")
    
else:
    BRANCH_COLORS = {}
    SEVERITY_COLORS = {}
    HEADER_BLUE = None
    HEADER_LIGHT = None
    ACCENT_BLUE = None
    LIGHT_GRAY = None
    MEDIUM_GRAY = None
    DARK_GRAY = None
    ALERT_RED = None
    ALERT_BG = None
    SUCCESS_GREEN = None
    SUCCESS_BG = None
    WARNING_ORANGE = None
    WARNING_BG = None
    CARD_BG = None
    CARD_BORDER = None


# ============================================================================
# POPPINS FONT SETUP
# ============================================================================

POPPINS_FONT_DIR = Path(__file__).parent / "fonts"
POPPINS_URL = "https://fonts.google.com/download?family=Poppins"

POPPINS_VARIANTS = {
    "Poppins": "Poppins-Regular.ttf",
    "Poppins-Bold": "Poppins-Bold.ttf",
    "Poppins-Italic": "Poppins-Italic.ttf",
    "Poppins-BoldItalic": "Poppins-BoldItalic.ttf",
    "Poppins-Light": "Poppins-Light.ttf",
    "Poppins-Medium": "Poppins-Medium.ttf",
    "Poppins-SemiBold": "Poppins-SemiBold.ttf",
}

_fonts_registered = False


def _download_poppins_fonts():
    """Download Poppins font from Google Fonts if not present."""
    POPPINS_FONT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if fonts already exist
    required_file = POPPINS_FONT_DIR / "Poppins-Regular.ttf"
    if required_file.exists():
        log.debug("Poppins fonts already present.")
        return True
    
    log.info("Downloading Poppins font from Google Fonts...")
    try:
        # Google Fonts download link
        response = urllib.request.urlopen(POPPINS_URL, timeout=30)
        zip_data = BytesIO(response.read())
        
        with zipfile.ZipFile(zip_data, 'r') as zf:
            for name in zf.namelist():
                if name.endswith('.ttf'):
                    # Extract just the filename
                    font_name = os.path.basename(name)
                    target_path = POPPINS_FONT_DIR / font_name
                    with zf.open(name) as src, open(target_path, 'wb') as dst:
                        dst.write(src.read())
        
        log.info(f"Poppins fonts downloaded to {POPPINS_FONT_DIR}")
        return True
    except Exception as e:
        log.warning(f"Failed to download Poppins fonts: {e}")
        log.warning("Will use Helvetica as fallback.")
        return False


def _register_poppins_fonts():
    """Register Poppins fonts with reportlab."""
    global _fonts_registered
    if _fonts_registered:
        return True
    
    if not HAS_REPORTLAB:
        return False
    
    # Try to download fonts if not present
    _download_poppins_fonts()
    
    registered_count = 0
    for font_name, filename in POPPINS_VARIANTS.items():
        font_path = POPPINS_FONT_DIR / filename
        if font_path.exists():
            try:
                pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
                registered_count += 1
            except Exception as e:
                log.debug(f"Could not register {font_name}: {e}")
    
    if registered_count > 0:
        log.info(f"Registered {registered_count} Poppins font variants.")
        _fonts_registered = True
        return True
    else:
        log.warning("No Poppins fonts registered. Using Helvetica fallback.")
        return False


def _get_font_name(bold: bool = False, italic: bool = False) -> str:
    """Get appropriate font name based on style."""
    if _fonts_registered:
        if bold and italic:
            return "Poppins-BoldItalic"
        elif bold:
            return "Poppins-Bold"
        elif italic:
            return "Poppins-Italic"
        return "Poppins"
    else:
        # Fallback to Helvetica
        if bold and italic:
            return "Helvetica-BoldOblique"
        elif bold:
            return "Helvetica-Bold"
        elif italic:
            return "Helvetica-Oblique"
        return "Helvetica"


# ============================================================================
# STYLES (Enhanced Professional Design)
# ============================================================================

def _create_styles() -> dict:
    """Create comprehensive paragraph styles for professional PDF layout."""
    _register_poppins_fonts()
    base_font = _get_font_name()
    bold_font = _get_font_name(bold=True)
    semibold_font = "Poppins-SemiBold" if _fonts_registered else bold_font
    light_font = "Poppins-Light" if _fonts_registered else base_font
    
    styles = {
        # Document Title
        "title": ParagraphStyle(
            "Title",
            fontName=bold_font,
            fontSize=28,
            leading=34,
            alignment=TA_CENTER,
            spaceAfter=4,
            textColor=HEADER_BLUE,
        ),
        # Document Subtitle
        "subtitle": ParagraphStyle(
            "Subtitle",
            fontName=light_font,
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
            spaceAfter=20,
            textColor=DARK_GRAY,
        ),
        # Section Headers (H1)
        "heading1": ParagraphStyle(
            "Heading1",
            fontName=bold_font,
            fontSize=18,
            leading=24,
            spaceBefore=24,
            spaceAfter=12,
            textColor=HEADER_BLUE,
            borderPadding=(0, 0, 4, 0),
        ),
        # Subsection Headers (H2)
        "heading2": ParagraphStyle(
            "Heading2",
            fontName=semibold_font,
            fontSize=14,
            leading=18,
            spaceBefore=16,
            spaceAfter=8,
            textColor=HEADER_LIGHT,
        ),
        # Minor Headers (H3)
        "heading3": ParagraphStyle(
            "Heading3",
            fontName=semibold_font,
            fontSize=11,
            leading=14,
            spaceBefore=10,
            spaceAfter=4,
            textColor=DARK_GRAY,
        ),
        # Body Text
        "normal": ParagraphStyle(
            "Normal",
            fontName=base_font,
            fontSize=10,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=6,
            textColor=colors.black,
        ),
        # Justified Body Text
        "body_justified": ParagraphStyle(
            "BodyJustified",
            fontName=base_font,
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
            textColor=colors.black,
        ),
        # Small Text / Captions
        "small": ParagraphStyle(
            "Small",
            fontName=base_font,
            fontSize=8,
            leading=11,
            spaceAfter=4,
            textColor=DARK_GRAY,
        ),
        # Metadata Text
        "meta": ParagraphStyle(
            "Meta",
            fontName=base_font,
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            spaceAfter=16,
            textColor=DARK_GRAY,
        ),
        # Alert / Warning Text
        "alert": ParagraphStyle(
            "Alert",
            fontName=bold_font,
            fontSize=10,
            leading=14,
            textColor=ALERT_RED,
            spaceAfter=4,
            leftIndent=8,
        ),
        # Alert Header
        "alert_header": ParagraphStyle(
            "AlertHeader",
            fontName=bold_font,
            fontSize=12,
            leading=16,
            textColor=ALERT_RED,
            spaceBefore=8,
            spaceAfter=6,
        ),
        # Bullet List Item
        "bullet": ParagraphStyle(
            "Bullet",
            fontName=base_font,
            fontSize=10,
            leading=14,
            leftIndent=20,
            bulletIndent=8,
            spaceAfter=4,
        ),
        # Numbered List Item
        "numbered": ParagraphStyle(
            "Numbered",
            fontName=base_font,
            fontSize=10,
            leading=14,
            leftIndent=24,
            spaceAfter=6,
        ),
        # Clinical Note Text
        "note_text": ParagraphStyle(
            "NoteText",
            fontName=base_font,
            fontSize=9,
            leading=13,
            alignment=TA_LEFT,
            spaceAfter=2,
        ),
        # Executive Summary Key Metrics
        "metric_label": ParagraphStyle(
            "MetricLabel",
            fontName=base_font,
            fontSize=9,
            leading=12,
            textColor=DARK_GRAY,
            alignment=TA_CENTER,
        ),
        "metric_value": ParagraphStyle(
            "MetricValue",
            fontName=bold_font,
            fontSize=14,
            leading=18,
            textColor=HEADER_BLUE,
            alignment=TA_CENTER,
        ),
        # Risk Tier (Large, Prominent)
        "risk_tier_high": ParagraphStyle(
            "RiskTierHigh",
            fontName=bold_font,
            fontSize=24,
            leading=28,
            textColor=colors.white,
            alignment=TA_CENTER,
            backColor=SEVERITY_COLORS.get("HIGH", colors.red),
        ),
        "risk_tier_moderate": ParagraphStyle(
            "RiskTierModerate",
            fontName=bold_font,
            fontSize=24,
            leading=28,
            textColor=colors.white,
            alignment=TA_CENTER,
            backColor=SEVERITY_COLORS.get("MODERATE", colors.orange),
        ),
        "risk_tier_low": ParagraphStyle(
            "RiskTierLow",
            fontName=bold_font,
            fontSize=24,
            leading=28,
            textColor=colors.white,
            alignment=TA_CENTER,
            backColor=SEVERITY_COLORS.get("LOW", colors.green),
        ),
        # Card Title (for scenario cards)
        "card_title": ParagraphStyle(
            "CardTitle",
            fontName=bold_font,
            fontSize=12,
            leading=16,
            spaceBefore=0,
            spaceAfter=6,
        ),
        # Card Body
        "card_body": ParagraphStyle(
            "CardBody",
            fontName=base_font,
            fontSize=9,
            leading=13,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        # Scenario Label (Branch A/B/C)
        "scenario_label": ParagraphStyle(
            "ScenarioLabel",
            fontName=bold_font,
            fontSize=10,
            leading=14,
            textColor=colors.white,
        ),
        # Evidence Quote
        "evidence_quote": ParagraphStyle(
            "EvidenceQuote",
            fontName=_get_font_name(italic=True),
            fontSize=9,
            leading=12,
            leftIndent=12,
            rightIndent=12,
            textColor=DARK_GRAY,
            borderColor=MEDIUM_GRAY,
            borderPadding=4,
        ),
        # Table Header
        "table_header": ParagraphStyle(
            "TableHeader",
            fontName=bold_font,
            fontSize=9,
            leading=12,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        # Table Cell
        "table_cell": ParagraphStyle(
            "TableCell",
            fontName=base_font,
            fontSize=8,
            leading=11,
            alignment=TA_LEFT,
        ),
    }
    return styles


# ============================================================================
# CUSTOM FLOWABLES (Visual Components)
# ============================================================================

class RiskTierBox(Flowable):
    """A prominent boxed risk tier indicator."""
    
    def __init__(self, risk_tier: str, width: float = 4*cm, height: float = 1.5*cm):
        Flowable.__init__(self)
        self.risk_tier = risk_tier.upper().strip()
        self.width = width
        self.height = height
        
    def draw(self):
        # Determine color based on risk level
        if "HIGH" in self.risk_tier:
            bg_color = SEVERITY_COLORS.get("HIGH", colors.red)
        elif "MODERATE" in self.risk_tier or "MEDIUM" in self.risk_tier:
            bg_color = SEVERITY_COLORS.get("MODERATE", colors.orange)
        else:
            bg_color = SEVERITY_COLORS.get("LOW", colors.green)
        
        # Draw rounded rectangle background
        self.canv.setFillColor(bg_color)
        self.canv.setStrokeColor(bg_color)
        self.canv.roundRect(0, 0, self.width, self.height, 8, fill=1, stroke=0)
        
        # Draw text
        self.canv.setFillColor(colors.white)
        self.canv.setFont(_get_font_name(bold=True), 18)
        text_width = self.canv.stringWidth(self.risk_tier, _get_font_name(bold=True), 18)
        self.canv.drawString((self.width - text_width) / 2, self.height / 2 - 6, self.risk_tier)


class MetricCard(Flowable):
    """A small metric card showing label and value."""
    
    def __init__(self, label: str, value: str, width: float = 3.5*cm, height: float = 1.8*cm):
        Flowable.__init__(self)
        self.label = label
        self.value = value
        self.width = width
        self.height = height
        
    def draw(self):
        # Draw card background
        self.canv.setFillColor(LIGHT_GRAY)
        self.canv.setStrokeColor(MEDIUM_GRAY)
        self.canv.roundRect(0, 0, self.width, self.height, 4, fill=1, stroke=1)
        
        # Draw value (larger, bold)
        self.canv.setFillColor(HEADER_BLUE)
        self.canv.setFont(_get_font_name(bold=True), 14)
        value_width = self.canv.stringWidth(str(self.value), _get_font_name(bold=True), 14)
        self.canv.drawString((self.width - value_width) / 2, self.height / 2 + 2, str(self.value))
        
        # Draw label (smaller, below)
        self.canv.setFillColor(DARK_GRAY)
        self.canv.setFont(_get_font_name(), 8)
        label_width = self.canv.stringWidth(self.label, _get_font_name(), 8)
        self.canv.drawString((self.width - label_width) / 2, self.height / 2 - 14, self.label)


class SectionDivider(Flowable):
    """A styled section divider with optional label."""
    
    def __init__(self, width: float = 16*cm, label: str = None):
        Flowable.__init__(self)
        self.width = width
        self.height = 24 if label else 12
        self.label = label
        
    def draw(self):
        if self.label:
            # Draw line with label in center
            self.canv.setStrokeColor(MEDIUM_GRAY)
            self.canv.setLineWidth(1)
            
            # Calculate label width
            self.canv.setFont(_get_font_name(bold=True), 10)
            label_width = self.canv.stringWidth(self.label, _get_font_name(bold=True), 10)
            
            # Draw left line
            self.canv.line(0, self.height/2, (self.width - label_width) / 2 - 10, self.height/2)
            # Draw right line
            self.canv.line((self.width + label_width) / 2 + 10, self.height/2, self.width, self.height/2)
            
            # Draw label
            self.canv.setFillColor(HEADER_LIGHT)
            self.canv.drawString((self.width - label_width) / 2, self.height/2 - 4, self.label)
        else:
            # Simple line
            self.canv.setStrokeColor(MEDIUM_GRAY)
            self.canv.setLineWidth(0.5)
            self.canv.line(0, self.height/2, self.width, self.height/2)


# ============================================================================
# PAGE 1: EXECUTIVE SUMMARY
# ============================================================================

def _build_executive_summary(
    styles: dict,
    evidence_result: dict,
    silver_labels: dict,
    report: dict,
    critical_signals: dict,
    scenarios_result: dict,
) -> list:
    """Build the executive summary page with prominent risk tier and key metrics."""
    elements = []
    
    # ─────────────────────────────────────────────────────────────────────────
    # TITLE & METADATA
    # ─────────────────────────────────────────────────────────────────────────
    elements.append(Paragraph("Clinical Preparedness Report", styles["title"]))
    elements.append(Paragraph(
        "Counterfactual Risk Assessment with Evidence Attribution",
        styles["subtitle"]
    ))
    
    # Metadata row
    dx = silver_labels.get("primary_mh_diagnosis", {})
    dx_title = dx.get("title", "Unknown") if isinstance(dx, dict) else "Unknown"
    generated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    meta_text = f"Generated: {generated}  •  Primary Dx: {dx_title}"
    elements.append(Paragraph(meta_text, styles["meta"]))
    
    elements.append(HRFlowable(width="100%", thickness=2, color=HEADER_BLUE, spaceAfter=20))
    
    # ─────────────────────────────────────────────────────────────────────────
    # RISK TIER (PROMINENT BOX)
    # ─────────────────────────────────────────────────────────────────────────
    risk_tier = report.get("overall_risk_tier", "UNKNOWN").upper().strip()
    if not risk_tier or risk_tier == "?":
        risk_tier = "UNKNOWN"
    
    # Create centered risk tier box
    risk_box = RiskTierBox(risk_tier, width=5*cm, height=1.8*cm)
    
    # Title above the box
    elements.append(Paragraph("OVERALL RISK ASSESSMENT", styles["heading2"]))
    elements.append(Spacer(1, 6))
    
    # Center the risk box in a table
    risk_table = Table([[risk_box]], colWidths=[17*cm])
    risk_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(risk_table)
    elements.append(Spacer(1, 8))
    
    # Risk justification
    justification = report.get("risk_tier_justification", "")
    if justification:
        elements.append(Paragraph(
            f"<i>{_escape_xml(justification)}</i>",
            styles["small"]
        ))
    elements.append(Spacer(1, 16))
    
    # ─────────────────────────────────────────────────────────────────────────
    # KEY METRICS DASHBOARD
    # ─────────────────────────────────────────────────────────────────────────
    spans = evidence_result.get("evidence_spans", [])
    coverage = evidence_result.get("coverage_stats", {})
    nexus_count = evidence_result.get("nexus_count", 0)
    
    # Count active vs gated scenarios
    scenarios = scenarios_result.get("scenarios", [])
    active_scenarios = sum(1 for s in scenarios if not s.get("gated"))
    gated_scenarios = sum(1 for s in scenarios if s.get("gated"))
    
    # Risk factor count
    risk_factors = report.get("key_risk_factors", [])
    protective_factors = report.get("protective_factors", [])
    
    metrics_data = [
        [
            MetricCard("Evidence Spans", str(len(spans))),
            MetricCard("Nexus Factors", str(nexus_count)),
            MetricCard("Note Coverage", f"{coverage.get('coverage_pct', 0)}%"),
            MetricCard("Active Scenarios", f"{active_scenarios}/{len(scenarios)}"),
        ]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[4.3*cm] * 4)
    metrics_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 20))
    
    # ─────────────────────────────────────────────────────────────────────────
    # CRITICAL SAFETY ALERTS (Grouped & Deduplicated)
    # ─────────────────────────────────────────────────────────────────────────
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
    
    if unique_alerts:
        # Group alerts by type
        suicide_alerts = []
        substance_alerts = []
        violence_alerts = []
        other_alerts = []
        
        for alert in unique_alerts:
            alert_lower = alert.lower()
            if any(kw in alert_lower for kw in ["suicid", "self-harm", "self harm", "si/hi"]):
                suicide_alerts.append(alert)
            elif any(kw in alert_lower for kw in ["substance", "drug", "alcohol", "overdose", "intox"]):
                substance_alerts.append(alert)
            elif any(kw in alert_lower for kw in ["violen", "homicid", "assault", "aggress"]):
                violence_alerts.append(alert)
            else:
                other_alerts.append(alert)
        
        elements.append(Paragraph("CRITICAL SAFETY ALERTS", styles["alert_header"]))
        
        # Create alert box with red border
        alert_items = []
        
        if suicide_alerts:
            alert_items.append(Paragraph("<b>Suicide/Self-Harm Risk:</b>", styles["alert"]))
            for a in suicide_alerts[:3]:
                cleaned = _clean_alert_text(a)
                if cleaned:
                    alert_items.append(Paragraph(f"  • {_escape_xml(cleaned)}", styles["bullet"]))
        
        if substance_alerts:
            alert_items.append(Paragraph("<b>Substance Use Concern:</b>", styles["alert"]))
            for a in substance_alerts[:3]:
                cleaned = _clean_alert_text(a)
                if cleaned:
                    alert_items.append(Paragraph(f"  • {_escape_xml(cleaned)}", styles["bullet"]))
        
        if violence_alerts:
            alert_items.append(Paragraph("<b>Violence/Aggression Risk:</b>", styles["alert"]))
            for a in violence_alerts[:3]:
                cleaned = _clean_alert_text(a)
                if cleaned:
                    alert_items.append(Paragraph(f"  • {_escape_xml(cleaned)}", styles["bullet"]))
        
        if other_alerts:
            alert_items.append(Paragraph("<b>Other Concerns:</b>", styles["alert"]))
            for a in other_alerts[:3]:
                cleaned = _clean_alert_text(a)
                if cleaned:
                    alert_items.append(Paragraph(f"  • {_escape_xml(cleaned)}", styles["bullet"]))
        
        if alert_items:
            alert_table = Table([[alert_items]], colWidths=[16*cm])
            alert_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), ALERT_BG),
                ('BOX', (0, 0), (-1, -1), 2, ALERT_RED),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            elements.append(alert_table)
        elements.append(Spacer(1, 16))
    
    # ─────────────────────────────────────────────────────────────────────────
    # TOP 3 IMMEDIATE RISKS
    # ─────────────────────────────────────────────────────────────────────────
    if risk_factors:
        elements.append(Paragraph("TOP RISK FACTORS", styles["heading2"]))
        
        for i, rf in enumerate(risk_factors[:5], 1):
            # Determine severity based on keywords
            rf_lower = rf.lower()
            if any(kw in rf_lower for kw in ["suicid", "homicid", "overdose", "acute", "severe", "imminent"]):
                severity = "HIGH"
                severity_color = SEVERITY_COLORS["HIGH"]
            elif any(kw in rf_lower for kw in ["substance", "history", "prior", "chronic"]):
                severity = "MODERATE"
                severity_color = SEVERITY_COLORS["MODERATE"]
            else:
                severity = "LOW"
                severity_color = SEVERITY_COLORS["LOW"]
            
            # Create risk factor entry with severity indicator
            rf_html = f'<font color="#{severity_color.hexval()[2:]}">[{severity}]</font> {_escape_xml(rf)}'
            elements.append(Paragraph(f"  {i}. {rf_html}", styles["numbered"]))
        
        elements.append(Spacer(1, 12))
    
    # ─────────────────────────────────────────────────────────────────────────
    # PROTECTIVE FACTORS (Brief)
    # ─────────────────────────────────────────────────────────────────────────
    if protective_factors:
        elements.append(Paragraph("PROTECTIVE FACTORS", styles["heading2"]))
        for pf in protective_factors[:3]:
            pf_html = f'<font color="#{SUCCESS_GREEN.hexval()[2:]}">✓</font> {_escape_xml(pf)}'
            elements.append(Paragraph(f"  {pf_html}", styles["bullet"]))
        elements.append(Spacer(1, 12))
    
    # ─────────────────────────────────────────────────────────────────────────
    # COLOR LEGEND (Compact)
    # ─────────────────────────────────────────────────────────────────────────
    elements.append(HRFlowable(width="100%", thickness=0.5, color=MEDIUM_GRAY, spaceBefore=12, spaceAfter=12))
    elements.append(Paragraph("Evidence Color Legend", styles["heading3"]))
    
    legend_data = [
        ["", "Branch A — Psychiatric", "", "Branch B — Substance Use", "", "Branch C — Social", "", "Nexus — Multi-Branch"],
    ]
    
    legend_table = Table(legend_data, colWidths=[0.6*cm, 4*cm, 0.6*cm, 4*cm, 0.6*cm, 3.5*cm, 0.6*cm, 3*cm])
    legend_style = [
        ('FONTNAME', (0, 0), (-1, -1), _get_font_name()),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 0), (0, 0), BRANCH_COLORS["A"]["color"]),
        ('BACKGROUND', (2, 0), (2, 0), BRANCH_COLORS["B"]["color"]),
        ('BACKGROUND', (4, 0), (4, 0), BRANCH_COLORS["C"]["color"]),
        ('BACKGROUND', (6, 0), (6, 0), BRANCH_COLORS["nexus"]["color"]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]
    legend_table.setStyle(TableStyle(legend_style))
    elements.append(legend_table)
    
    return elements


# ============================================================================
# PAGE 2: EVIDENCE ATTRIBUTION TABLE
# ============================================================================

def _estimate_severity(factors: List[str], branches: List[str]) -> str:
    """Estimate severity based on factors and branch coverage."""
    factors_str = " ".join(factors).lower()
    
    # High severity indicators
    high_keywords = ["suicid", "homicid", "overdose", "acute", "severe", "imminent", "crisis", "psychosis", "mania"]
    if any(kw in factors_str for kw in high_keywords):
        return "HIGH"
    
    # Nexus factors (multi-branch) are at least moderate
    if len(branches) > 1 or "nexus" in str(branches).lower():
        return "MODERATE"
    
    # Moderate indicators
    moderate_keywords = ["substance", "alcohol", "drug", "depression", "anxiety", "history", "prior"]
    if any(kw in factors_str for kw in moderate_keywords):
        return "MODERATE"
    
    return "LOW"


def _build_evidence_table_section(styles: dict, evidence_result: dict) -> list:
    """Build a clean, structured evidence attribution table."""
    elements = []
    
    elements.append(PageBreak())
    elements.append(Paragraph("Evidence Attribution Table", styles["heading1"]))
    elements.append(Paragraph(
        "Detailed breakdown of clinical evidence supporting each risk scenario branch.",
        styles["small"]
    ))
    elements.append(Spacer(1, 12))
    
    spans = evidence_result.get("evidence_spans", [])
    
    if not spans:
        elements.append(Paragraph("No evidence spans identified.", styles["normal"]))
        return elements
    
    # Build table with columns: ID | Factor | Evidence | Branch | Severity
    table_data = [[
        Paragraph("<b>ID</b>", styles["table_header"]),
        Paragraph("<b>Risk Factor</b>", styles["table_header"]),
        Paragraph("<b>Evidence from Clinical Note</b>", styles["table_header"]),
        Paragraph("<b>Branch</b>", styles["table_header"]),
        Paragraph("<b>Severity</b>", styles["table_header"]),
    ]]
    
    for idx, span in enumerate(spans[:25], 1):
        factors = span.get("factors", [])
        factors_str = "; ".join(factors[:2])
        if len(factors_str) > 40:
            factors_str = factors_str[:37] + "..."
        
        evidence_text = span.get("text", "")
        # Clean and truncate evidence
        evidence_text = evidence_text.replace("\n", " ").strip()
        if len(evidence_text) > 80:
            evidence_text = evidence_text[:77] + "..."
        
        branches = span.get("branches", [])
        branch_str = ", ".join(branches)
        color_key = span.get("color_key", "A")
        
        # Estimate severity
        severity = _estimate_severity(factors, branches)
        severity_color = SEVERITY_COLORS.get(severity, DARK_GRAY)
        
        row = [
            Paragraph(str(idx), styles["table_cell"]),
            Paragraph(_escape_xml(factors_str), styles["table_cell"]),
            Paragraph(f'<i>"{_escape_xml(evidence_text)}"</i>', styles["table_cell"]),
            Paragraph(f"<b>{branch_str}</b>", styles["table_cell"]),
            Paragraph(f"<font color='#{severity_color.hexval()[2:]}'><b>{severity}</b></font>", styles["table_cell"]),
        ]
        table_data.append(row)
    
    # Create table with proper column widths
    evidence_table = Table(
        table_data, 
        colWidths=[0.8*cm, 3*cm, 7.5*cm, 1.5*cm, 1.8*cm],
        repeatRows=1  # Repeat header on new pages
    )
    
    # Table styling
    table_style = [
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), HEADER_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), _get_font_name(bold=True)),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
        
        # Body rows
        ('FONTNAME', (0, 1), (-1, -1), _get_font_name()),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ALIGN', (0, 1), (0, -1), 'CENTER'),  # ID column
        ('ALIGN', (3, 1), (4, -1), 'CENTER'),  # Branch and Severity columns
        ('VALIGN', (0, 1), (-1, -1), 'TOP'),
        
        # Grid lines
        ('GRID', (0, 0), (-1, -1), 0.5, MEDIUM_GRAY),
        ('LINEBELOW', (0, 0), (-1, 0), 2, HEADER_BLUE),
        
        # Padding
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]
    
    # Add alternating row colors and branch color indicators
    for idx, span in enumerate(spans[:25], 1):
        row_idx = idx  # Account for header row
        
        # Alternating row background
        if idx % 2 == 0:
            table_style.append(('BACKGROUND', (0, row_idx), (-1, row_idx), LIGHT_GRAY))
        
        # Color indicator in ID column based on branch
        color_key = span.get("color_key", "A")
        branch_color = BRANCH_COLORS.get(color_key, BRANCH_COLORS["A"])["color"]
        table_style.append(('BACKGROUND', (0, row_idx), (0, row_idx), branch_color))
        table_style.append(('TEXTCOLOR', (0, row_idx), (0, row_idx), colors.white))
    
    evidence_table.setStyle(TableStyle(table_style))
    elements.append(evidence_table)
    
    # Show if there are more spans
    if len(spans) > 25:
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(
            f"<i>Showing top 25 of {len(spans)} evidence spans.</i>",
            styles["small"]
        ))
    
    # Ungrounded branches warning
    ungrounded = evidence_result.get("ungrounded_branches", [])
    if ungrounded:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(
            f"⚠ <b>Ungrounded Branches:</b> {', '.join(ungrounded)} — no supporting evidence found in clinical note.",
            styles["alert"]
        ))
    
    return elements


# ============================================================================
# PAGE 3-4: HIGHLIGHTED CLINICAL NOTE (Structured)
# ============================================================================

# Common clinical note section headers to detect and format
CLINICAL_HEADERS = [
    "Chief Complaint", "CC",
    "History of Present Illness", "HPI",
    "Past Medical History", "PMH", "PMHx",
    "Past Psychiatric History", "PPH", "Psychiatric History",
    "Social History", "SH", "SHx",
    "Family History", "FH", "FHx",
    "Medications", "Meds", "Current Medications", "Home Medications",
    "Allergies", "ALLERGIES",
    "Review of Systems", "ROS",
    "Physical Exam", "Physical Examination", "PE",
    "Mental Status Exam", "MSE", "Mental Status Examination",
    "Laboratory Data", "Labs", "Laboratory",
    "Imaging", "Radiology",
    "Assessment", "Assessment and Plan", "A/P", "A&P",
    "Plan", "Discharge Plan",
    "Diagnosis", "Diagnoses", "Dx",
    "Discharge Diagnosis", "Discharge Diagnoses",
    "Discharge Condition",
    "Discharge Instructions",
    "Disposition",
    "Service",
    "Attending",
    "Major Surgical or Invasive Procedure",
    "Admission Date", "Discharge Date",
    "Date of Birth", "DOB",
    "Sex", "Gender",
    "Unit No",
    "Name",
]


def _clean_clinical_text(text: str) -> str:
    """Clean up clinical note text for better readability."""
    import re
    
    # Normalize whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\r', '\n', text)
    
    # Remove excessive blank lines (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Clean up lines with only whitespace
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep empty lines but limit them
        if not stripped:
            if cleaned_lines and cleaned_lines[-1] != '':
                cleaned_lines.append('')
        else:
            cleaned_lines.append(stripped)
    
    return '\n'.join(cleaned_lines)


def _is_section_header(line: str) -> bool:
    """Check if a line is a clinical note section header."""
    stripped = line.strip().rstrip(':').strip()
    
    # Check against known headers
    for header in CLINICAL_HEADERS:
        if stripped.lower() == header.lower():
            return True
        if stripped.lower().startswith(header.lower() + ":"):
            return True
    
    # Also detect patterns like "SECTION_NAME:" at start of line
    if stripped.endswith(':') and len(stripped) < 50 and stripped.isupper():
        return True
    
    return False


def _format_section_header(line: str) -> str:
    """Format a section header line with styling."""
    stripped = line.strip()
    return f'<b><font color="#1A365D">{_escape_xml(stripped)}</font></b>'


def _build_highlighted_note_section(styles: dict, note_text: str, evidence_result: dict) -> list:
    """Build structured clinical note section with highlighted evidence spans."""
    elements = []
    
    elements.append(PageBreak())
    elements.append(Paragraph("Clinical Note with Evidence Highlights", styles["heading1"]))
    
    # Color legend
    legend_data = [[
        Paragraph('<font color="#DC3545"><b>■</b></font> Branch A: Psychiatric', styles["small"]),
        Paragraph('<font color="#FD7E14"><b>■</b></font> Branch B: Substance', styles["small"]),
        Paragraph('<font color="#0D6EFD"><b>■</b></font> Branch C: Social', styles["small"]),
        Paragraph('<font color="#FFC107"><b>■</b></font> Nexus: Multi-Branch', styles["small"]),
    ]]
    legend_table = Table(legend_data, colWidths=[4.2*cm, 4.2*cm, 3.8*cm, 4.2*cm])
    legend_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_GRAY),
        ('BOX', (0, 0), (-1, -1), 1, MEDIUM_GRAY),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    elements.append(legend_table)
    elements.append(Spacer(1, 12))
    
    # Clean the clinical note text
    cleaned_note = _clean_clinical_text(note_text)
    
    # Build highlight regions from evidence spans
    spans = evidence_result.get("evidence_spans", [])
    highlight_regions = []
    for s in spans:
        start = s.get("start", 0)
        end = s.get("end", 0)
        color_key = s.get("color_key", "A")
        color_hex = BRANCH_COLORS.get(color_key, BRANCH_COLORS["A"])["hex"]
        branches = s.get("branches", [])
        if start < end <= len(note_text):
            highlight_regions.append((start, end, color_hex, branches))
    
    # Sort and clean overlaps
    highlight_regions.sort(key=lambda x: x[0])
    cleaned_regions = []
    last_end = 0
    for start, end, color_hex, branches in highlight_regions:
        if start >= last_end:
            cleaned_regions.append((start, end, color_hex, branches))
            last_end = end
        elif start < last_end < end:
            cleaned_regions.append((last_end, end, color_hex, branches))
            last_end = end
    
    # Create a styled clinical note section header style
    header_style = ParagraphStyle(
        "NoteHeader",
        fontName=_get_font_name(bold=True),
        fontSize=10,
        leading=14,
        spaceBefore=12,
        spaceAfter=4,
        textColor=HEADER_BLUE,
    )
    
    body_style = ParagraphStyle(
        "NoteBody",
        fontName=_get_font_name(),
        fontSize=9,
        leading=13,
        alignment=TA_LEFT,
        spaceAfter=3,
        leftIndent=8,
    )
    
    # Process lines - identify sections and format
    lines = cleaned_note.split('\n')
    current_pos = 0
    current_section_content = []
    
    for line in lines:
        # Find position in original text for highlighting
        line_start = note_text.find(line, max(0, current_pos - 50))
        if line_start == -1:
            line_start = current_pos
        line_end = line_start + len(line)
        
        stripped = line.strip()
        
        # Handle empty lines - create paragraph break
        if not stripped:
            if current_section_content:
                elements.append(Spacer(1, 6))
            current_pos = line_end + 1
            continue
        
        # Check if this is a section header
        is_header = _is_section_header(stripped)
        
        # Find highlights in this line (use original positions)
        line_highlights = [
            (max(s, line_start) - line_start, min(e, line_end) - line_start, c, b)
            for s, e, c, b in cleaned_regions
            if s < line_end and e > line_start
        ]
        
        # Build the line content with highlights
        if not line_highlights:
            if is_header:
                html_content = _format_section_header(stripped)
                elements.append(Paragraph(html_content, header_style))
            else:
                elements.append(Paragraph(_escape_xml(stripped), body_style))
        else:
            # Build HTML with highlights
            html_parts = []
            pos = 0
            line_text = line if len(line) > 0 else stripped
            
            for hl_start, hl_end, color_hex, branches in line_highlights:
                # Clamp to line bounds
                hl_start = max(0, min(hl_start, len(line_text)))
                hl_end = max(0, min(hl_end, len(line_text)))
                
                if hl_start >= hl_end:
                    continue
                
                # Text before highlight
                if pos < hl_start:
                    html_parts.append(_escape_xml(line_text[pos:hl_start]))
                
                # Highlighted text with branch marker
                hl_text = _escape_xml(line_text[hl_start:hl_end])
                branch_str = ",".join(branches)
                html_parts.append(
                    f'<font backColor="{color_hex}"><b>{hl_text}</b></font>'
                    f'<super><font size="5" color="#666666">[{branch_str}]</font></super>'
                )
                pos = hl_end
            
            # Remaining text
            if pos < len(line_text):
                html_parts.append(_escape_xml(line_text[pos:]))
            
            html_content = "".join(html_parts)
            
            if is_header:
                # Wrap header in bold/color
                elements.append(Paragraph(f'<b><font color="#1A365D">{html_content}</font></b>', header_style))
            else:
                elements.append(Paragraph(html_content, body_style))
        
        current_pos = line_end + 1
    
    return elements


def _escape_xml(text: str) -> str:
    """Escape XML special characters for Paragraph text."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;"))


def _clean_alert_text(text: str) -> str:
    """Clean up alert text by removing special characters and formatting."""
    import re
    
    if not text:
        return ""
    
    # Remove ■ characters and similar box drawing chars
    text = re.sub(r'[■□▪▫●○◆◇★☆▲△▼▽]', '', text)
    
    # Remove "SIGNAL:" type prefixes to avoid duplication
    text = re.sub(r'(SUICIDE|SELF-HARM|VIOLENCE|SUBSTANCE)[/-]?\s*(SIGNAL|ALERT|RISK):\s*', '', text, flags=re.IGNORECASE)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Clean up quotes and dashes
    text = text.strip().strip('-').strip()
    
    # Truncate long alerts and add ellipsis
    if len(text) > 200:
        text = text[:197] + "..."
    
    return text


# ============================================================================
# PAGE 5+: SCENARIO BRIEFING (CARD-BASED DESIGN)
# ============================================================================

def _build_scenario_card(
    styles: dict,
    scenario: dict,
    branch_id: str,
    branch_label: str,
    color_info: dict,
) -> list:
    """Build a visual card for a single scenario."""
    card_elements = []
    
    # Card header with branch color
    header_color = color_info["color"]
    header_text = f"{color_info.get('icon', '●')} [{branch_id}] {branch_label}"
    
    # Create colored header
    header_style = ParagraphStyle(
        f"CardHeader{branch_id}",
        fontName=_get_font_name(bold=True),
        fontSize=11,
        leading=14,
        textColor=colors.white,
    )
    
    # Check if scenario is gated (not applicable)
    if scenario.get("gated"):
        # Muted card for gated scenario
        card_content = [
            [Paragraph(header_text, header_style)],
            [Paragraph(
                f"<b>STATUS: NOT APPLICABLE</b><br/>"
                f"<i>{_escape_xml(scenario.get('gate_reason', 'Insufficient evidence for this pathway'))}</i>",
                styles["card_body"]
            )],
        ]
        
        card_table = Table(card_content, colWidths=[16*cm])
        card_table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), MEDIUM_GRAY),
            ('TEXTCOLOR', (0, 0), (-1, 0), DARK_GRAY),
            ('TOPPADDING', (0, 0), (0, 0), 8),
            ('BOTTOMPADDING', (0, 0), (0, 0), 8),
            ('LEFTPADDING', (0, 0), (0, 0), 10),
            # Body
            ('BACKGROUND', (0, 1), (-1, -1), LIGHT_GRAY),
            ('BOX', (0, 0), (-1, -1), 1, MEDIUM_GRAY),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('LEFTPADDING', (0, 1), (-1, -1), 10),
            ('RIGHTPADDING', (0, 1), (-1, -1), 10),
        ]))
        card_elements.append(card_table)
        card_elements.append(Spacer(1, 12))
        return card_elements
    
    # Active scenario card
    scenario_obj = scenario.get("scenario", {})
    
    # Build card body content
    body_parts = []
    
    # Plausibility (prominent)
    plaus = scenario_obj.get("plausibility", "?").upper()
    p_rat = scenario_obj.get("plausibility_rationale", "")
    
    plaus_color = SEVERITY_COLORS.get("HIGH" if plaus == "HIGH" else ("MODERATE" if plaus == "MODERATE" else "LOW"), DARK_GRAY)
    
    body_parts.append(Paragraph(
        f"<b>Plausibility:</b> <font color='#{plaus_color.hexval()[2:]}'><b>{plaus}</b></font>",
        styles["card_body"]
    ))
    if p_rat:
        body_parts.append(Paragraph(f"<i>{_escape_xml(p_rat)}</i>", styles["card_body"]))
    
    body_parts.append(Spacer(1, 6))
    
    # Warning Signs (bullet list)
    wsigns = scenario_obj.get("warning_signs", [])
    if wsigns:
        body_parts.append(Paragraph("<b>Warning Signs:</b>", styles["card_body"]))
        for ws in wsigns[:5]:
            body_parts.append(Paragraph(f"  • {_escape_xml(ws)}", styles["card_body"]))
        body_parts.append(Spacer(1, 4))
    
    # Crisis Endpoint (highlighted)
    crisis = scenario_obj.get("crisis_endpoint", "")
    if crisis:
        body_parts.append(Paragraph(
            f"<b>Crisis Endpoint:</b> <font color='#{ALERT_RED.hexval()[2:]}'>{_escape_xml(crisis)}</font>",
            styles["card_body"]
        ))
        body_parts.append(Spacer(1, 4))
    
    # Preparedness Actions (numbered)
    actions = scenario_obj.get("preparedness_actions", [])
    if actions:
        body_parts.append(Paragraph("<b>Preparedness Actions:</b>", styles["card_body"]))
        for i, a in enumerate(actions[:5], 1):
            body_parts.append(Paragraph(f"  {i}. {_escape_xml(a)}", styles["card_body"]))
    
    # Narrative (brief, at end)
    narrative = scenario_obj.get("narrative", "")
    if narrative:
        body_parts.append(Spacer(1, 6))
        narrative_preview = narrative[:300] + "..." if len(narrative) > 300 else narrative
        body_parts.append(Paragraph(
            f"<i><font color='#{DARK_GRAY.hexval()[2:]}'>{_escape_xml(narrative_preview)}</font></i>",
            styles["card_body"]
        ))
    
    # Build the card
    card_content = [
        [Paragraph(header_text, header_style)],
        [body_parts],
    ]
    
    card_table = Table(card_content, colWidths=[16*cm])
    card_table.setStyle(TableStyle([
        # Header row with branch color
        ('BACKGROUND', (0, 0), (-1, 0), header_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('TOPPADDING', (0, 0), (0, 0), 10),
        ('BOTTOMPADDING', (0, 0), (0, 0), 10),
        ('LEFTPADDING', (0, 0), (0, 0), 12),
        # Body with light colored background
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor(color_info.get("bg_light", "#FFFFFF"))),
        ('BOX', (0, 0), (-1, -1), 2, header_color),
        ('TOPPADDING', (0, 1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
        ('LEFTPADDING', (0, 1), (-1, -1), 12),
        ('RIGHTPADDING', (0, 1), (-1, -1), 12),
    ]))
    
    card_elements.append(KeepTogether(card_table))
    card_elements.append(Spacer(1, 16))
    
    return card_elements


def _build_report_section(
    styles: dict,
    report: dict,
    silver_labels: dict,
    scenarios_result: dict,
    critical_signals: dict,
) -> list:
    """Build the full preparedness report with card-based scenario design."""
    elements = []
    
    elements.append(PageBreak())
    elements.append(Paragraph("Detailed Clinical Assessment", styles["heading1"]))
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. PATIENT OVERVIEW
    # ─────────────────────────────────────────────────────────────────────────
    elements.append(Paragraph("1. Patient Overview", styles["heading2"]))
    overview = report.get("patient_overview", "Not available.")
    elements.append(Paragraph(_escape_xml(overview), styles["body_justified"]))
    elements.append(Spacer(1, 8))
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. MENTAL HEALTH STATUS
    # ─────────────────────────────────────────────────────────────────────────
    elements.append(Paragraph("2. Mental Health Status", styles["heading2"]))
    mh_status = report.get("mental_health_status", "Not available.")
    elements.append(Paragraph(_escape_xml(mh_status), styles["body_justified"]))
    elements.append(Spacer(1, 8))
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3. KEY RISK FACTORS (Enhanced)
    # ─────────────────────────────────────────────────────────────────────────
    risk_factors = report.get("key_risk_factors", [])
    if risk_factors:
        elements.append(Paragraph("3. Key Risk Factors", styles["heading2"]))
        
        for rf in risk_factors:
            rf_lower = rf.lower()
            # Determine severity color
            if any(kw in rf_lower for kw in ["suicid", "homicid", "overdose", "acute", "severe", "psychosis"]):
                sev_color = SEVERITY_COLORS["HIGH"]
                sev_label = "HIGH"
            elif any(kw in rf_lower for kw in ["substance", "history", "chronic", "depression"]):
                sev_color = SEVERITY_COLORS["MODERATE"]
                sev_label = "MOD"
            else:
                sev_color = SEVERITY_COLORS["LOW"]
                sev_label = "LOW"
            
            rf_html = f"<font color='#{sev_color.hexval()[2:]}'><b>[{sev_label}]</b></font> {_escape_xml(rf)}"
            elements.append(Paragraph(f"  • {rf_html}", styles["bullet"]))
        
        elements.append(Spacer(1, 8))
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. PROTECTIVE FACTORS
    # ─────────────────────────────────────────────────────────────────────────
    protective = report.get("protective_factors", [])
    if protective:
        elements.append(Paragraph("4. Protective Factors", styles["heading2"]))
        for pf in protective:
            pf_html = f"<font color='#{SUCCESS_GREEN.hexval()[2:]}'>✓</font> {_escape_xml(pf)}"
            elements.append(Paragraph(f"  {pf_html}", styles["bullet"]))
        elements.append(Spacer(1, 8))
    
    # ─────────────────────────────────────────────────────────────────────────
    # 5. COUNTERFACTUAL SCENARIO BRIEFING (Card Layout)
    # ─────────────────────────────────────────────────────────────────────────
    elements.append(PageBreak())
    elements.append(Paragraph("5. Counterfactual Scenario Briefing", styles["heading1"]))
    elements.append(Paragraph(
        "Each card represents a potential deterioration pathway. "
        "Content is preserved exactly as generated — only formatting is enhanced.",
        styles["small"]
    ))
    elements.append(Spacer(1, 16))
    
    for scenario in scenarios_result.get("scenarios", []):
        branch_id = scenario.get("scenario_id", "?")
        branch_label = scenario.get("branch_label", "Unknown Pathway")
        color_info = BRANCH_COLORS.get(branch_id, BRANCH_COLORS["A"])
        
        card_elements = _build_scenario_card(
            styles, scenario, branch_id, branch_label, color_info
        )
        elements.extend(card_elements)
    
    # ─────────────────────────────────────────────────────────────────────────
    # 6. PRIORITY PREPAREDNESS ACTIONS
    # ─────────────────────────────────────────────────────────────────────────
    elements.append(Paragraph("6. Priority Preparedness Actions", styles["heading1"]))
    elements.append(Paragraph(
        "Immediate clinical interventions recommended based on risk assessment.",
        styles["small"]
    ))
    elements.append(Spacer(1, 8))
    
    priority_actions = report.get("priority_actions", [])
    if priority_actions:
        action_data = []
        for i, action in enumerate(priority_actions, 1):
            # Create numbered action with emphasis on first few
            if i <= 3:
                action_html = f"<b>{i}.</b> <b>{_escape_xml(action)}</b>"
            else:
                action_html = f"<b>{i}.</b> {_escape_xml(action)}"
            action_data.append([Paragraph(action_html, styles["numbered"])])
        
        actions_table = Table(action_data, colWidths=[15.5*cm])
        actions_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), SUCCESS_BG),  # Green background for ALL rows
            ('BOX', (0, 0), (-1, -1), 2, SUCCESS_GREEN),   # Thicker green border
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('LINEBELOW', (0, 0), (-1, -2), 0.5, SUCCESS_GREEN),  # Separator lines between items
        ]))
        elements.append(actions_table)
    else:
        elements.append(Paragraph("No priority actions specified.", styles["normal"]))
    
    elements.append(Spacer(1, 16))
    
    # ─────────────────────────────────────────────────────────────────────────
    # 7. RISK TIER SUMMARY (Final)
    # ─────────────────────────────────────────────────────────────────────────
    risk_tier = report.get("overall_risk_tier", "?").upper().strip()
    justification = report.get("risk_tier_justification", "")
    
    elements.append(HRFlowable(width="100%", thickness=1, color=HEADER_BLUE, spaceBefore=8, spaceAfter=16))
    
    elements.append(Paragraph("7. Risk Assessment Summary", styles["heading2"]))
    
    # Create final risk tier display
    if "HIGH" in risk_tier:
        tier_color = SEVERITY_COLORS["HIGH"]
        tier_bg = ALERT_BG
    elif "MODERATE" in risk_tier or "MEDIUM" in risk_tier:
        tier_color = SEVERITY_COLORS["MODERATE"]
        tier_bg = WARNING_BG
    else:
        tier_color = SEVERITY_COLORS["LOW"]
        tier_bg = SUCCESS_BG
    
    tier_content = [
        [Paragraph(
            f"<font size='14'><b>OVERALL RISK TIER: </b></font>"
            f"<font color='#{tier_color.hexval()[2:]}' size='16'><b>{risk_tier}</b></font>",
            styles["normal"]
        )],
    ]
    if justification:
        tier_content.append([Paragraph(f"<i>{_escape_xml(justification)}</i>", styles["small"])])
    
    tier_table = Table(tier_content, colWidths=[15.5*cm])
    tier_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), tier_bg),
        ('BOX', (0, 0), (-1, -1), 2, tier_color),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    elements.append(tier_table)
    
    # Document end footer
    elements.append(Spacer(1, 24))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=MEDIUM_GRAY))
    elements.append(Paragraph(
        f"<i>Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Clinical Preparedness Pipeline V4</i>",
        styles["small"]
    ))
    
    return elements


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_pdf_report(
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
    Generate a professional PDF report with:
    - Executive summary with prominent risk tier
    - Evidence attribution table
    - Color-highlighted clinical note
    - Card-based scenario briefing
    - Priority actions and risk justification

    Args:
        note_text: Original clinical note text.
        evidence_result: Phase 4 output (evidence attribution).
        report: Phase 3 output (preparedness report).
        silver_labels: Phase 1 output (extracted factors).
        scenarios_result: Phase 2 output (scenarios).
        critical_signals: Phase 1.6 output (critical signals).
        output_path: Where to save the .pdf file. Auto-generated if None.
        subject_id: Patient subject ID for filename.
        hadm_id: Admission ID for filename.

    Returns:
        Path to the generated .pdf file, or None if reportlab is missing.
    """
    if not HAS_REPORTLAB:
        log.warning(
            "Phase 4.5: reportlab not installed. "
            "Install with: pip install reportlab"
        )
        return None
    
    # Ensure critical_signals is a dict
    if critical_signals is None:
        critical_signals = {}
    
    log.info("Phase 4.5: Generating professional PDF report (V2) ...")
    
    # Register fonts
    _register_poppins_fonts()
    
    # Create styles
    styles = _create_styles()
    
    # Output path
    if output_path is None:
        output_path = f"report_{subject_id}_{hadm_id}.pdf"
    
    # Ensure .pdf extension
    if not output_path.endswith('.pdf'):
        output_path = output_path.rsplit('.', 1)[0] + '.pdf'
    
    # Create document with professional margins
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=1.8*cm,
        rightMargin=1.8*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm,
    )
    
    # Build all elements
    elements = []
    
    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 1: EXECUTIVE SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    elements.extend(_build_executive_summary(
        styles, evidence_result, silver_labels, report, critical_signals, scenarios_result
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 2: EVIDENCE ATTRIBUTION TABLE
    # ─────────────────────────────────────────────────────────────────────────
    elements.extend(_build_evidence_table_section(styles, evidence_result))
    
    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 3-4: HIGHLIGHTED CLINICAL NOTE
    # ─────────────────────────────────────────────────────────────────────────
    elements.extend(_build_highlighted_note_section(styles, note_text, evidence_result))
    
    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 5+: DETAILED REPORT WITH SCENARIO CARDS
    # ─────────────────────────────────────────────────────────────────────────
    elements.extend(_build_report_section(styles, report, silver_labels, scenarios_result, critical_signals))
    
    # Build PDF
    try:
        doc.build(elements)
        log.info(f"Phase 4.5: Professional PDF report saved → {output_path}")
        return output_path
    except Exception as e:
        log.error(f"Phase 4.5: PDF generation failed: {e}")
        return None
