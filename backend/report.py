from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from datetime import datetime
import os
import json

# ─────────────────────────────────────────────────────────────────────────────
# Groq AI integration
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY = "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"
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"

# ── Defect severity weights ───────────────────────────────────────────────────
# base_penalty is deducted at 100% confidence; scaled linearly by actual confidence.
# safety=True means the part directly affects driving safety.
DEFECT_SEVERITY = {
    "windshield": {"base_penalty": 30, "label": "Windshield Damage",  "safety": True},
    "bonnet":     {"base_penalty": 20, "label": "Bonnet Damage",      "safety": True},
    "light":      {"base_penalty": 15, "label": "Light Damage",       "safety": True},
    "bumper":     {"base_penalty": 14, "label": "Bumper Damage",      "safety": False},
    "fender":     {"base_penalty": 10, "label": "Fender Damage",      "safety": False},
    "door":       {"base_penalty": 9,  "label": "Door Damage",        "safety": False},
    "dickey":     {"base_penalty": 7,  "label": "Boot/Dickey Damage", "safety": False},
}

DEFECT_ADVICE = {
    "windshield": "Windshield cracks or chips impair driver visibility and compromise cabin structural integrity — repair or replace immediately before driving.",
    "bonnet":     "Bonnet damage can indicate impact near the engine bay — inspect for underlying mechanical damage and repair to prevent water ingress.",
    "light":      "Damaged lights reduce visibility and road legality — replace affected light units before driving at night or in poor conditions.",
    "bumper":     "Bumper damage is primarily cosmetic but reduces impact absorption in future collisions — repair recommended to restore safety rating.",
    "fender":     "Fender damage is cosmetic but exposed metal will oxidise over time — clean, treat, and repair at nearest opportunity.",
    "door":       "Door damage may affect weather sealing and interior noise — inspect door seals and hinges, then repair to prevent further wear.",
    "dickey":     "Boot/dickey damage can compromise weather sealing and rear-compartment security — repair as needed.",
}

# ── Defect type labels: part + confidence → specific damage descriptor ────────
# Minor  = <55% confidence   (surface marks, light scuffs)
# Moderate = 55-79%          (visible dents, cracks)
# Severe = >=80%             (deep damage, shattered, crumpled)
DEFECT_TYPE_LABELS = {
    "windshield": {
        "minor":    "Surface Chip / Stress Crack",
        "moderate": "Windshield Crack",
        "severe":   "Shattered / Major Crack",
    },
    "bonnet": {
        "minor":    "Surface Scratch / Paint Chip",
        "moderate": "Bonnet Dent",
        "severe":   "Crumple / Deep Impact Damage",
    },
    "bumper": {
        "minor":    "Scuff / Paint Scratch",
        "moderate": "Bumper Dent / Crack",
        "severe":   "Bumper Collapse / Severe Impact",
    },
    "fender": {
        "minor":    "Surface Scratch",
        "moderate": "Fender Dent",
        "severe":   "Deep Dent / Crease",
    },
    "door": {
        "minor":    "Paint Scratch / Scuff",
        "moderate": "Door Dent",
        "severe":   "Deep Dent / Panel Damage",
    },
    "dickey": {
        "minor":    "Surface Scratch",
        "moderate": "Boot Dent",
        "severe":   "Deep Dent / Structural Damage",
    },
    "light": {
        "minor":    "Light Cover Scratch",
        "moderate": "Cracked Light Cover",
        "severe":   "Broken / Shattered Light Unit",
    },
}


def _get_defect_type_label(part_key: str, confidence: float) -> str:
    """
    Returns a human-readable defect TYPE label based on the part and confidence.
    e.g. part='door', conf=42 -> 'Paint Scratch / Scuff'
         part='windshield', conf=88 -> 'Shattered / Major Crack'
    Falls back to 'Part Damage' if not mapped.
    """
    tier = "severe" if confidence >= 80 else "moderate" if confidence >= 55 else "minor"
    mapping = DEFECT_TYPE_LABELS.get(part_key, {})
    return mapping.get(tier, f"{part_key.capitalize()} Damage")


def call_groq(prompt: str, system: str = "You are an expert vehicle inspector and automotive AI assistant.") -> str:
    try:
        import urllib.request
        payload = json.dumps({
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt}
            ],
            "max_tokens": 1400,
            "temperature": 0.4
        }).encode("utf-8")
        req = urllib.request.Request(
            GROQ_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[Groq] call failed: {e}")
        return ""


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _build_defect_detail(defects: list) -> list:
    """Collapse raw defect list into per-type dicts with max confidence."""
    grouped = {}
    for d in defects:
        if isinstance(d, (list, tuple)):
            name, conf = str(d[0]), float(d[1]) if len(d) > 1 else 0.0
        elif isinstance(d, dict):
            name = d.get("label") or d.get("class") or d.get("name") or "Unknown"
            conf = float(d.get("confidence", 0))
        else:
            name, conf = str(d), 0.0
        key = name.lower().strip()
        sev = DEFECT_SEVERITY.get(key, {"base_penalty": 10, "label": name.capitalize(), "safety": False})
        if key not in grouped or grouped[key]["confidence"] < conf:
            grouped[key] = {"key": key, "label": sev["label"], "confidence": conf, "sev": sev}
    return list(grouped.values())


def _compute_health_score(defect_details: list, engine_result) -> int:
    score = 100
    for d in defect_details:
        score -= d["sev"]["base_penalty"] * (d["confidence"] / 100)
    if engine_result and engine_result.get("is_knock"):
        score -= 25 * (engine_result.get("confidence", 100) / 100)
    return max(5, min(100, round(score)))


def _compute_risk(defect_details: list, engine_result) -> str:
    has_safety = any(d["sev"]["safety"] for d in defect_details)
    high_conf  = any(d["confidence"] >= 75 for d in defect_details)
    count      = len(defect_details)
    knock      = engine_result.get("is_knock", False) if engine_result else False

    if knock and (count >= 2 or has_safety):
        return "Critical"
    if knock or (has_safety and high_conf) or count >= 4:
        return "High"
    if count >= 2 or (has_safety and not high_conf) or (count == 1 and high_conf):
        return "Medium"
    return "Low"


# ── Smart fallbacks ───────────────────────────────────────────────────────────

def _fallback_summary(status: str, defect_details: list) -> str:
    if not defect_details:
        return ("The vehicle passed all visual inspection checks with no detectable exterior damage. "
                "It appears to be in excellent condition with no immediate repair needs.")
    parts        = [d["label"] for d in defect_details]
    safety_parts = [d["label"] for d in defect_details if d["sev"]["safety"]]
    parts_str    = ", ".join(parts)
    if safety_parts:
        return (f"The inspection detected damage to: {parts_str}. "
                f"Safety-critical damage was identified on {', '.join(safety_parts)}, requiring immediate attention. "
                f"Professional repair is strongly recommended before continued vehicle use.")
    return (f"The inspection identified damage to {parts_str}. "
            f"The affected components are primarily cosmetic in nature. "
            f"Timely repair will prevent further deterioration and maintain resale value.")


def _fallback_recommendations(defect_details: list, engine_result) -> list:
    recs = []
    for d in defect_details:
        advice = DEFECT_ADVICE.get(d["key"])
        if advice:
            recs.append(advice)
    if engine_result and engine_result.get("is_knock"):
        recs.append("Engine knock detected — consult a mechanic to inspect fuel system, ignition timing, and knock sensors.")
    if not recs:
        recs = ["Continue routine maintenance schedule.", "Keep records of this clean inspection for future resale."]
    return recs[:5]


def _fallback_risk_factors(defect_details: list, engine_result) -> list:
    factors = []
    safety = [d["label"] for d in defect_details if d["sev"]["safety"]]
    if safety:
        factors.append(f"Safety-critical components affected: {', '.join(safety)}")
    high_conf = [d["label"] for d in defect_details if d["confidence"] >= 75]
    if high_conf:
        factors.append(f"High-confidence damage on: {', '.join(high_conf)}")
    if engine_result and engine_result.get("is_knock"):
        factors.append(f"Engine knock confirmed at {engine_result.get('confidence', 0):.0f}% confidence")
    cosmetic = [d["label"] for d in defect_details if not d["sev"]["safety"] and d["confidence"] < 75]
    if cosmetic:
        factors.append(f"Minor cosmetic damage on: {', '.join(cosmetic)}")
    return factors[:4]


# ── Main AI analysis ──────────────────────────────────────────────────────────

def generate_ai_analysis(
    defects: list,
    vehicle_info: dict,
    engine_result,
    overall_status: str
) -> dict:
    make    = vehicle_info.get("make",    "Unknown")
    model   = vehicle_info.get("model",   "Unknown")
    year    = vehicle_info.get("year",    "Unknown")
    mileage = vehicle_info.get("mileage", "Unknown")

    defect_details = _build_defect_detail(defects)
    fallback_score = _compute_health_score(defect_details, engine_result)
    fallback_risk  = _compute_risk(defect_details, engine_result)

    # Rich defect description for the prompt
    if defect_details:
        defect_lines = []
        for d in defect_details:
            sev_word   = "severe" if d["confidence"] >= 80 else "moderate" if d["confidence"] >= 55 else "minor"
            dtype      = _get_defect_type_label(d["key"], d["confidence"])
            safety_tag = " [SAFETY-CRITICAL]" if d["sev"]["safety"] else ""
            defect_lines.append(
                f"  - {d['label']} → likely defect type: {dtype} | {sev_word} ({d['confidence']:.1f}% confidence){safety_tag}"
            )
        defect_str = "\n".join(defect_lines)
    else:
        defect_str = "  - None detected"

    engine_str = "Not tested"
    if engine_result and engine_result.get("verdict"):
        engine_str = (
            f"{'KNOCK DETECTED' if engine_result.get('is_knock') else 'HEALTHY'} "
            f"(verdict: {engine_result.get('verdict','?')}, "
            f"confidence: {engine_result.get('confidence', 0):.1f}%)"
        )

    prompt = f"""You are an expert automotive inspector writing a professional vehicle condition report.

Vehicle: {year} {make} {model}  |  Mileage: {mileage}
Overall status: {overall_status}
Engine audio: {engine_str}

Detected body defects:
{defect_str}

SCORING RULES (very important — follow exactly):
- Start at 100. Deduct ONLY for what is actually detected.
- Minor defect <55% confidence (cosmetic only): deduct 5-8 pts
- Moderate defect 55-79% confidence: deduct 10-15 pts
- Severe defect >=80% confidence: deduct 18-28 pts
- Safety-critical parts (windshield, lights, bonnet): add 5 extra pts to deduction
- Engine knock: deduct 20-25 pts
- A single low-confidence cosmetic defect (door, fender, bumper, dickey) must NOT drop below 75
- Only reach below 50 with multiple high-confidence or safety-critical defects
- Suggested score based on these exact defects: {fallback_score}

Respond ONLY with valid JSON, no markdown:
{{
  "health_score": <integer 0-100>,
  "risk_level": "<Low|Medium|High|Critical>",
  "summary": "<2-3 sentences naming the SPECIFIC defects found, their severity level, and practical impact on the vehicle>",
  "defect_explanations": {{
    "<exact defect key in lowercase e.g. door, fender, windshield>": "<1 sentence: what this damage means for the vehicle and what specific action is needed>"
  }},
  "recommendations": [
    "<specific action tied to actual defects found — not generic>",
    "<another specific recommendation>",
    "<another>"
  ],
  "risk_factors": [
    "<specific risk from actual defects>",
    "<another>"
  ]
}}

Risk level guide:
- Low: 1 minor cosmetic defect, no safety parts, no knock
- Medium: 1-2 defects, none safety-critical, no knock
- High: safety-critical part damaged OR knock OR 3+ defects
- Critical: safety-critical damage AND knock, or multiple severe defects
"""

    raw = call_groq(prompt)

    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = "\n".join(clean.split("\n")[:-1])
        result = json.loads(clean)

        result["health_score"] = max(5, min(100, int(result.get("health_score", fallback_score))))
        result.setdefault("risk_level",          fallback_risk)
        result.setdefault("summary",             _fallback_summary(overall_status, defect_details))
        result.setdefault("recommendations",     _fallback_recommendations(defect_details, engine_result))
        result.setdefault("risk_factors",        _fallback_risk_factors(defect_details, engine_result))
        result.setdefault("defect_explanations", {})
        return result

    except Exception as e:
        print(f"[Groq] JSON parse failed: {e}\nRaw: {raw}")
        return {
            "health_score":        fallback_score,
            "risk_level":          fallback_risk,
            "summary":             _fallback_summary(overall_status, defect_details),
            "recommendations":     _fallback_recommendations(defect_details, engine_result),
            "risk_factors":        _fallback_risk_factors(defect_details, engine_result),
            "defect_explanations": {},
        }


# ─────────────────────────────────────────────────────────────────────────────
# PDF constants
# ─────────────────────────────────────────────────────────────────────────────
ALL_DEFECTS = ["Bonnet", "Bumper", "Dickey", "Door", "Fender", "Light", "Windshield"]


# ─────────────────────────────────────────────────────────────────────────────
# PDF helpers
# ─────────────────────────────────────────────────────────────────────────────
def _health_bar_table(score: int, styles):
    if score >= 75:   bar_color = colors.HexColor("#16a34a")
    elif score >= 50: bar_color = colors.HexColor("#d97706")
    else:             bar_color = colors.HexColor("#dc2626")

    filled    = round(score / 10)
    empty     = 10 - filled
    hex_color = "#16a34a" if score >= 75 else "#d97706" if score >= 50 else "#dc2626"

    t = Table(
        [[Paragraph(f"<b>{score}/100</b>", ParagraphStyle("ScoreVal", parent=styles["Normal"],
                    fontSize=22, textColor=bar_color, alignment=1))],
         [Paragraph(f'<font color="{hex_color}">{"█" * filled}</font><font color="#e5e7eb">{"█" * empty}</font>',
                    ParagraphStyle("Bar", parent=styles["Normal"], fontSize=18, alignment=1))],
         [Paragraph("Vehicle Health Score", ParagraphStyle("ScoreLabel", parent=styles["Normal"],
                    fontSize=9, textColor=colors.grey, alignment=1))]],
        colWidths=[17 * cm]
    )
    t.setStyle(TableStyle([
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND',    (0, 0), (-1, -1), colors.HexColor("#f9fafb")),
        ('BOX',           (0, 0), (-1, -1), 1, colors.HexColor("#e5e7eb")),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    return t


def _risk_badge_color(risk: str):
    return {
        "Low":      ("#166534", "#dcfce7"),
        "Medium":   ("#92400e", "#fef3c7"),
        "High":     ("#9a3412", "#ffedd5"),
        "Critical": ("#991b1b", "#fee2e2"),
    }.get(risk, ("#374151", "#f3f4f6"))


# ─────────────────────────────────────────────────────────────────────────────
# Main report generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_report(
    defects: list,
    image_paths: list,
    output_path: str,
    vehicle_info: dict = None,
    engine_result: dict = None,
    ai_analysis: dict = None
):
    if vehicle_info is None:
        vehicle_info = {}

    def _norm(d):
        if isinstance(d, (list, tuple)):
            return [str(d[0]) if len(d) > 0 else "Unknown", float(d[1]) if len(d) > 1 else 0.0]
        if isinstance(d, dict):
            label = d.get("label") or d.get("class") or d.get("name") or "Unknown"
            return [str(label), float(d.get("confidence", 0))]
        return [str(d), 0.0]

    defects = [_norm(d) for d in (defects or [])]

    detected_set         = {d[0].lower() for d in defects if d[0]}
    total_detected_types = len(detected_set)

    if total_detected_types == 0:
        overall_status = "PASS";  status_color = "#166534"
    elif total_detected_types <= 2:
        overall_status = "ATTENTION"; status_color = "#d97706"
    else:
        overall_status = "FAIL";  status_color = "#991b1b"

    if ai_analysis is None:
        ai_analysis = generate_ai_analysis(defects, vehicle_info, engine_result, overall_status)

    health_score        = ai_analysis.get("health_score", 50)
    risk_level          = ai_analysis.get("risk_level",   "Medium")
    ai_summary          = ai_analysis.get("summary",      "")
    recommendations     = ai_analysis.get("recommendations", [])
    risk_factors        = ai_analysis.get("risk_factors",  [])
    defect_explanations = ai_analysis.get("defect_explanations", {})

    doc    = SimpleDocTemplate(output_path, pagesize=A4,
                               rightMargin=2*cm, leftMargin=2*cm,
                               topMargin=2*cm,   bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    # ── Title ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("VEHICLE INSPECTION REPORT", ParagraphStyle(
        "BigTitle", parent=styles["Title"], fontSize=24, spaceAfter=6,
        alignment=1, textColor=colors.HexColor("#1e3a8a")
    )))
    story.append(Paragraph(
        f"AI-Powered · {datetime.now().strftime('%B %d, %Y at %H:%M')}",
        ParagraphStyle("Sub", parent=styles["Normal"], fontSize=10,
                       alignment=1, textColor=colors.grey, spaceAfter=20)
    ))

    # ── Vehicle Info ──────────────────────────────────────────────────────────
    vin        = vehicle_info.get("vin",     "Not Provided")
    make       = vehicle_info.get("make",    "Not Provided")
    model      = vehicle_info.get("model",   "Not Provided")
    year       = vehicle_info.get("year",    "Not Provided")
    mileage    = vehicle_info.get("mileage", "Not Provided")
    make_model = f"{make} {model}" if make != "Not Provided" else "Not Provided"

    info_table = Table(
        [["VIN / Registration", vin,        "Mileage", mileage],
         ["Make / Model",       make_model, "Year",    year]],
        colWidths=[5*cm, 5*cm, 4*cm, 4*cm]
    )
    info_table.setStyle(TableStyle([
        ('GRID',          (0, 0), (-1, -1), 1,   colors.grey),
        ('BACKGROUND',    (0, 0), (-1,  0), colors.lightgrey),
        ('FONTNAME',      (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE',      (0, 0), (-1, -1), 10),
        ('ALIGN',         (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 8),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 24))

    # ── AI Health Assessment ──────────────────────────────────────────────────
    story.append(Paragraph("<b>AI Vehicle Health Assessment</b>", styles["Heading2"]))
    story.append(Spacer(1, 8))
    story.append(_health_bar_table(health_score, styles))
    story.append(Spacer(1, 12))

    fg, bg = _risk_badge_color(risk_level)
    risk_table = Table([[
        Paragraph("<b>Risk Level</b>", styles["Normal"]),
        Paragraph(f"<font color='{fg}'><b>{risk_level}</b></font>",
                  ParagraphStyle("RiskVal", parent=styles["Normal"], fontSize=14)),
        Paragraph("<b>Overall Status</b>", styles["Normal"]),
        Paragraph(f"<font color='{status_color}'><b>{overall_status}</b></font>",
                  ParagraphStyle("StatusVal", parent=styles["Normal"], fontSize=14)),
    ]], colWidths=[4*cm, 4.5*cm, 4*cm, 4.5*cm])
    risk_table.setStyle(TableStyle([
        ('BOX',           (0, 0), (-1, -1), 1,   colors.HexColor("#e5e7eb")),
        ('LINEAFTER',     (1, 0), (1,  0),  0.5, colors.HexColor("#e5e7eb")),
        ('BACKGROUND',    (0, 0), (1,  0),  colors.HexColor(bg)),
        ('BACKGROUND',    (2, 0), (3,  0),  colors.HexColor("#f0f9ff")),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 16))

    # ── Risk Factors ──────────────────────────────────────────────────────────
    if risk_factors:
        story.append(Paragraph("<b>Key Risk Factors:</b>", styles["Heading3"]))
        for rf in risk_factors:
            story.append(Paragraph(f"• {rf}", ParagraphStyle(
                "RF", parent=styles["Normal"], fontSize=10,
                leftIndent=14, spaceBefore=3, textColor=colors.HexColor("#7f1d1d")
            )))
        story.append(Spacer(1, 12))

    # ── AI Summary ────────────────────────────────────────────────────────────
    story.append(Paragraph("<b>AI Inspection Summary</b>", styles["Heading3"]))
    story.append(Paragraph(ai_summary, ParagraphStyle(
        "AISummary", parent=styles["Normal"],
        fontSize=10.5, leading=16, leftIndent=10, rightIndent=10,
        spaceBefore=6, spaceAfter=6,
        backColor=colors.HexColor("#f0f9ff"),
        textColor=colors.HexColor("#1e3a5f")
    )))
    story.append(Spacer(1, 12))

    # ── Defect-by-Defect Analysis ─────────────────────────────────────────────
    if detected_set:
        story.append(Paragraph("<b>Defect Analysis</b>", styles["Heading3"]))
        story.append(Spacer(1, 4))

        # Build per-defect rows — use Groq explanations if available, else fallback
        for d_label, d_conf in defects:
            key       = d_label.lower().strip()
            sev_info  = DEFECT_SEVERITY.get(key, {"label": d_label, "safety": False, "base_penalty": 10})
            is_safety = sev_info["safety"]
            dtype     = _get_defect_type_label(key, d_conf)
            sev_word  = "Severe" if d_conf >= 80 else "Moderate" if d_conf >= 55 else "Minor"
            color_hex = "#991b1b" if is_safety else "#92400e"
            tag       = " ⚠ Safety-Critical" if is_safety else ""
            expl      = (defect_explanations.get(key)
                         or defect_explanations.get(d_label.lower())
                         or DEFECT_ADVICE.get(key,
                            "Consult a qualified technician for assessment and repair."))

            story.append(Paragraph(
                f"<font color='{color_hex}'><b>{d_label} — {dtype}{tag}</b></font>  "
                f"<font color='#6b7280' size=9>({sev_word}, {d_conf:.1f}% confidence)</font>",
                ParagraphStyle("DH", parent=styles["Normal"],
                               fontSize=10, spaceBefore=7, leftIndent=10)
            ))
            story.append(Paragraph(expl, ParagraphStyle(
                "DE", parent=styles["Normal"],
                fontSize=9.5, leftIndent=22, spaceBefore=2, spaceAfter=5,
                textColor=colors.HexColor("#374151")
            )))
        story.append(Spacer(1, 12))

    # ── Recommendations ───────────────────────────────────────────────────────
    story.append(Paragraph("<b>AI Recommendations</b>", styles["Heading3"]))
    for i, rec in enumerate(recommendations, 1):
        story.append(Paragraph(f"<b>{i}.</b> {rec}", ParagraphStyle(
            "Rec", parent=styles["Normal"],
            fontSize=10, leftIndent=14, spaceBefore=4, leading=14
        )))
    story.append(Spacer(1, 24))

    # ── Engine Audio Analysis ─────────────────────────────────────────────────
    if engine_result and engine_result.get("verdict"):
        is_knock = engine_result.get("is_knock", False)
        verdict  = engine_result.get("verdict", "")
        conf     = engine_result.get("confidence", 0)
        duration = engine_result.get("duration_s", 0)
        eng_remark = (
            "Pre-detonation knock detected. Possible causes: low-octane fuel, carbon buildup, "
            "faulty knock sensor, or advanced ignition timing. Immediate mechanic inspection recommended."
            if is_knock else
            "No knock detected. Engine sounds healthy. Continue regular maintenance schedule."
        )
        story.append(Paragraph("<b>Engine Sound Analysis</b>", styles["Heading2"]))
        story.append(Spacer(1, 8))
        eng_table = Table(
            [["Engine Status", "KNOCK DETECTED" if is_knock else "ENGINE HEALTHY", "Confidence", f"{conf}%"],
             ["Verdict",       verdict,                                              "Audio Duration", f"{duration}s"],
             ["Remarks",       Paragraph(eng_remark, styles["Normal"]),             "",               ""]],
            colWidths=[4*cm, 6*cm, 4*cm, 4*cm]
        )
        eng_table.setStyle(TableStyle([
            ('GRID',          (0, 0), (-1, -1), 0.8, colors.grey),
            ('BACKGROUND',    (0, 0), (-1,  0),
             colors.HexColor("#fee2e2") if is_knock else colors.HexColor("#dcfce7")),
            ('FONTNAME',      (0, 0), (-1,  0), 'Helvetica-Bold'),
            ('FONTSIZE',      (0, 0), (-1, -1), 10),
            ('ALIGN',         (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING',   (0, 0), (-1, -1), 8),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING',    (0, 0), (-1, -1), 8),
            ('SPAN',          (1, 2), (3,  2)),
        ]))
        story.append(eng_table)
        story.append(Spacer(1, 24))

    # ── Damage Checklist ──────────────────────────────────────────────────────
    story.append(Paragraph("<b>Damage Inspection Checklist</b>", styles["Heading2"]))
    story.append(Spacer(1, 10))

    table_data = [["Component", "Defect Type", "Confidence", "Severity"]]
    for defect_name in ALL_DEFECTS:
        key = defect_name.lower()
        if key in detected_set:
            confs      = [c for d, c in defects if d.lower() == key]
            max_conf   = max(confs) if confs else 0
            sev_word   = "Severe" if max_conf >= 80 else "Moderate" if max_conf >= 55 else "Minor"
            dtype      = _get_defect_type_label(key, max_conf)
            if DEFECT_SEVERITY.get(key, {}).get("safety"):
                sev_word += " ⚠"
            table_data.append([
                Paragraph(f"<b>{defect_name}</b>", styles["Normal"]),
                Paragraph(f"<font color='#991b1b'>{dtype}</font>", styles["Normal"]),
                Paragraph(f"{max_conf:.1f}%", styles["Normal"]),
                Paragraph(sev_word, styles["Normal"]),
            ])
        else:
            table_data.append([
                Paragraph(f"<b>{defect_name}</b>", styles["Normal"]),
                Paragraph("No Damage Detected", styles["Normal"]),
                Paragraph("—", styles["Normal"]),
                Paragraph("—", styles["Normal"]),
            ])

    table_data.append([
        Paragraph("<b>Total Unique Defects</b>", styles["Normal"]),
        Paragraph(f"<font color='{status_color}'><b>{total_detected_types}</b></font>", styles["Normal"]),
        Paragraph("<b>Overall Status:</b>", styles["Normal"]),
        Paragraph(f"<font color='{status_color}'><b>{overall_status}</b></font>", styles["Normal"]),
    ])

    defect_table = Table(table_data, colWidths=[5.5*cm, 3.8*cm, 3.5*cm, 5.7*cm])
    defect_table.setStyle(TableStyle([
        ('GRID',          (0, 0), (-1, -1), 0.8, colors.grey),
        ('BACKGROUND',    (0, 0), (-1,  0), colors.lightgrey),
        ('FONTNAME',      (0, 0), (-1,  0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1,  0), 11),
        ('ALIGN',         (0, 0), (-1,  0), 'CENTER'),
        ('VALIGN',        (0, 0), (-1,  0), 'MIDDLE'),
        ('FONTSIZE',      (0, 1), (-1, -1), 10),
        ('ALIGN',         (0, 1), (-1, -1), 'LEFT'),
        ('VALIGN',        (0, 1), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING',    (0, 0), (-1, -1), 8),
    ]))
    story.append(defect_table)
    story.append(Spacer(1, 24))

    # ── Annotated Images ──────────────────────────────────────────────────────
    if defects and image_paths:
        story.append(Paragraph("<b>Defects Identified — Annotated Images</b>", styles["Heading2"]))
        story.append(Spacer(1, 12))
        image_grid_data = []
        row = []
        for idx, img_path in enumerate(image_paths):
            try:
                row.append(Image(img_path, width=8*cm, height=5*cm))
                if len(row) == 2:
                    image_grid_data.append(row); row = []
            except Exception as e:
                print(f"Error loading image {idx}: {e}")
        if row:
            image_grid_data.append(row)
        if image_grid_data:
            image_table = Table(image_grid_data, colWidths=[8.5*cm, 8.5*cm])
            image_table.setStyle(TableStyle([
                ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING',   (0, 0), (-1, -1), 5),
                ('RIGHTPADDING',  (0, 0), (-1, -1), 5),
                ('TOPPADDING',    (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(image_table)
        story.append(Spacer(1, 24))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(Paragraph(
        "<i>Note: This report was generated using AI-powered computer vision (Roboflow) "
        "and large language model analysis (Groq LLaMA 3.3). Results are based on provided images "
        "and should be verified by a qualified technician for critical decisions or insurance purposes.</i>",
        ParagraphStyle("Disclaimer", parent=styles["Normal"],
                       fontSize=9, textColor=colors.grey, alignment=1, spaceBefore=20)
    ))

    try:
        doc.build(story)
    except Exception as e:
        raise Exception(f"Failed to build PDF: {e}")
