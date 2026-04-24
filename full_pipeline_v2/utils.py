"""
Shared utilities V2: JSON parsing, text cleaning, schema loading.

Same as V1 core utilities, with config path updated.
"""

import json
import logging
import re
from pathlib import Path

from .config import SCHEMA_PATH

log = logging.getLogger(__name__)


def strip_js_comments(text: str) -> str:
    """Remove JavaScript-style // line comments from text."""
    return re.sub(r'//[^\n]*', '', text)


def parse_marked_sections(text: str, markers: list) -> dict:
    """
    Parse plain text split by known section markers into a dict.
    """
    marker_pattern = '|'.join(re.escape(m) for m in markers)
    pattern = re.compile(
        rf'(?:^|\n)\s*(?:#{{1,3}}\s*)?({marker_pattern})\s*:?\s*(?:\n|$)',
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    result = {}
    for i, match in enumerate(matches):
        key = match.group(1).strip().lower().replace(' ', '_')
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        result[key] = content
    return result


def parse_bullet_list(text: str) -> list:
    """Parse bullet-point or numbered list text into a list of strings."""
    items = []
    for line in text.split('\n'):
        stripped = line.strip()
        cleaned = re.sub(r'^[-•*]\s*', '', stripped)
        cleaned = re.sub(r'^\d+[.)\-]\s*', '', cleaned)
        if cleaned:
            items.append(cleaned)
    return items


def load_schema() -> dict:
    """Load the extraction schema from disk."""
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def get_schema_string() -> str:
    """Return the schema as a compact JSON string for prompt injection."""
    schema = load_schema()
    fields = schema.get("fields", schema)
    return json.dumps(fields)


def clean_report_text(text: str) -> str:
    """
    Clean a clinical report by handling redacted/blank fields.
    """
    text = re.sub(r'\b_{2,}\b', '[REDACTED]', text)
    text = re.sub(r'_{3,}', '[REDACTED]', text)
    text = re.sub(r'-{3,}', '[REDACTED]', text)
    text = re.sub(r'(\[REDACTED\]\s*){2,}', '[REDACTED] ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _try_repair_json_object(text: str):
    """Attempt to repair a truncated/corrupted JSON object."""
    start = text.find("{")
    if start == -1:
        return None

    obj_text = text[start:]

    try:
        return json.loads(obj_text)
    except json.JSONDecodeError:
        pass

    best_result = None
    best_len = 0

    for i in range(len(obj_text) - 1, 0, -1):
        ch = obj_text[i]
        if ch not in (",", "}", "]", '"', "e", "l"):
            continue

        snippet = obj_text[:i + 1].rstrip().rstrip(",")

        open_braces = snippet.count("{") - snippet.count("}")
        open_brackets = snippet.count("[") - snippet.count("]")

        if open_braces < 0 or open_brackets < 0:
            continue

        candidate = snippet + "]" * open_brackets + "}" * open_braces

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and len(str(candidate)) > best_len:
                best_result = parsed
                best_len = len(str(candidate))
                if len(parsed) >= 3:
                    log.info(f"JSON repair succeeded — recovered {len(parsed)} top-level keys")
                    return parsed
        except json.JSONDecodeError:
            continue

    if best_result is not None:
        log.info(f"JSON repair succeeded (partial) — recovered {len(best_result)} top-level keys")
    return best_result


def safe_parse_json(text: str):
    """
    Extract and parse the first valid JSON object or array from model output.
    """
    for tag in [
        "[INST]", "[/INST]", "</s>", "<s>",
        "<|end|>", "<|assistant|>", "<|user|>", "<|system|>",
        "```json", "```", "<|endoftext|>",
    ]:
        text = text.replace(tag, "")
    text = text.strip()

    text = strip_js_comments(text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    if start != -1:
        end = text.rfind("}")
        if end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

    if start != -1:
        repaired = _try_repair_json_object(text)
        if repaired is not None:
            return repaired

    arr_start = text.find("[")
    if arr_start != -1:
        depth = 0
        for i in range(arr_start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[arr_start:i + 1]
                    if len(candidate) > 5:
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break

    log.warning(f"JSON parse failed ({len(text)} chars). Preview: {text[:200]}")
    return {"_parse_failed": True, "_raw_text": text[:1000]}


def compact_factors(extracted: dict) -> str:
    """
    Build a concise, readable patient summary from extracted silver labels.
    V2: includes disease_category, progression, cognitive_status, violence indicators.
    """
    if not extracted or extracted.get("_parse_failed"):
        return "Clinical data extraction was incomplete. Limited data available."

    dx = extracted.get("primary_mh_diagnosis") or {}
    if not isinstance(dx, dict):
        dx = {}
    comorbid = extracted.get("comorbid_mh_diagnoses") or []
    med_comorb = extracted.get("medical_comorbidities") or []
    sui = extracted.get("suicide_risk_indicators") or {}
    if not isinstance(sui, dict):
        sui = {}
    vio = extracted.get("violence_risk_indicators") or {}
    if not isinstance(vio, dict):
        vio = {}
    sub = extracted.get("substance_use") or {}
    if not isinstance(sub, dict):
        sub = {}
    meds = extracted.get("medication_profile") or {}
    if not isinstance(meds, dict):
        meds = {}
    soc = extracted.get("social_factors") or {}
    if not isinstance(soc, dict):
        soc = {}
    adm = extracted.get("admission_context") or {}
    if not isinstance(adm, dict):
        adm = {}
    traj = extracted.get("trajectory_signals") or {}
    if not isinstance(traj, dict):
        traj = {}
    icu = extracted.get("icu_admission") or {}
    if not isinstance(icu, dict):
        icu = {}

    lines = []

    # Primary diagnosis
    dx_title = dx.get("title", "Unknown")
    dx_code = dx.get("icd_code", "")
    dx_cat = dx.get("dsm5_category", "")
    lines.append(f"Primary MH Diagnosis: {dx_title} (ICD: {dx_code}, DSM-5: {dx_cat})")

    # V2: Disease category and progression
    disease_cat = extracted.get("disease_category", "unknown")
    progression = extracted.get("progression", "unknown")
    lines.append(f"Disease Category: {disease_cat}")
    lines.append(f"Progression: {progression}")

    # V2: Cognitive status and functional impairment
    cog = extracted.get("cognitive_status", "unknown")
    func = extracted.get("functional_impairment", "unknown")
    if cog != "unknown":
        lines.append(f"Cognitive Status: {cog}")
    if func != "unknown":
        lines.append(f"Functional Impairment: {func}")

    # Severity
    lines.append(f"Overall Severity: {extracted.get('severity_level', 'unknown')}")

    # Discharge diagnoses (V2)
    discharge_dx = extracted.get("discharge_diagnoses") or []
    if discharge_dx:
        lines.append(f"Discharge Diagnoses: {'; '.join(discharge_dx)}")

    # Comorbid MH
    if comorbid:
        comorb_str = ", ".join(c.get("title", "?") for c in comorbid if c.get("title"))
        lines.append(f"Comorbid MH Diagnoses: {comorb_str}")
    else:
        lines.append("Comorbid MH Diagnoses: None identified")

    # Medical comorbidities
    if med_comorb:
        lines.append(f"Medical Comorbidities: {', '.join(med_comorb[:5])}")

    # Suicide risk
    sui_parts = []
    if sui.get("ideation_mentioned"):
        sui_parts.append("suicidal ideation mentioned")
    if sui.get("attempt_history"):
        sui_parts.append("prior attempt history")
    if sui.get("self_harm"):
        sui_parts.append("self-harm present")
    if sui.get("precautions_ordered"):
        sui_parts.append("precautions ordered")
    risk_lvl = sui.get("risk_level", "")
    if risk_lvl:
        sui_parts.append(f"risk level: {risk_lvl}")
    specific_threats = sui.get("specific_threats") or []
    if specific_threats:
        sui_parts.append(f"THREATS: {'; '.join(specific_threats)}")
    lines.append(f"Suicide Risk: {', '.join(sui_parts) if sui_parts else 'No active indicators'}")

    # V2: Violence risk
    vio_parts = []
    if vio.get("aggression_present"):
        vio_parts.append("aggression present")
    if vio.get("violence_history"):
        vio_parts.append("violence history")
    if vio.get("threats_made"):
        vio_parts.append("threats made")
    vio_lvl = vio.get("risk_level", "")
    if vio_lvl:
        vio_parts.append(f"risk level: {vio_lvl}")
    specific_incidents = vio.get("specific_incidents") or []
    if specific_incidents:
        vio_parts.append(f"INCIDENTS: {'; '.join(specific_incidents)}")
    lines.append(f"Violence Risk: {', '.join(vio_parts) if vio_parts else 'No active indicators'}")

    # Substance use
    sub_parts = []
    for key in ["alcohol", "drugs", "tobacco"]:
        val = sub.get(key, "none")
        if val and val != "none":
            sub_parts.append(f"{key}: {val}")
    if sub.get("positive_tox_screen"):
        detected = sub.get("substances_detected", [])
        sub_parts.append(f"positive tox screen ({', '.join(detected) if detected else 'unspecified'})")
    lines.append(f"Substance Use: {', '.join(sub_parts) if sub_parts else 'No active substance use'}")

    # Medication
    lines.append(f"Medication Adherence: {extracted.get('medication_adherence', 'unknown')}")
    psych_meds = meds.get("psychotropic_medications", [])
    if psych_meds:
        med_names = sorted(set(m.get("drug", "?") for m in psych_meds if m.get("drug")))
        lines.append(f"Psychotropic Medications: {', '.join(med_names)}")
    polypharm = meds.get("polypharmacy", False)
    if polypharm:
        lines.append(f"Polypharmacy: Yes (total meds: {meds.get('medication_count', '?')})")

    # Social
    lines.append(f"Marital Status: {soc.get('marital_status', 'unknown')}")
    lines.append(f"Insurance: {soc.get('insurance_type', 'unknown')}")
    lines.append(f"Social Isolation Risk: {'Yes' if soc.get('social_isolation_risk') else 'No'}")
    housing = soc.get("housing_status", "unknown")
    support = soc.get("support_system", "unknown")
    if housing != "unknown":
        lines.append(f"Housing Status: {housing}")
    if support != "unknown":
        lines.append(f"Support System: {support}")

    # Admission
    lines.append(
        f"Admission: {adm.get('admission_type', '?')}, "
        f"Emergency: {'Yes' if adm.get('is_emergency') else 'No'}, "
        f"LOS: {adm.get('los_days', '?')} days, "
        f"Discharge: {adm.get('discharge_disposition', '?')}"
    )

    # ICU
    if icu.get("admitted_to_icu"):
        lines.append(
            f"ICU: Yes (unit: {icu.get('careunit', '?')}, "
            f"LOS: {icu.get('los_days', '?')} days, "
            f"restraints: {'Yes' if icu.get('restraints_used') else 'No'})"
        )

    # Trajectory
    lines.append(
        f"Prior Admissions: {traj.get('prior_admission_count', 0)}, "
        f"Readmit <30d: {'Yes' if traj.get('readmission_within_30d') else 'No'}, "
        f"Dx Escalation: {'Yes' if traj.get('diagnosis_escalation') else 'No'}, "
        f"Severity Change: {traj.get('severity_change', 'unknown')}"
    )

    if extracted.get("psychiatric_service_involvement"):
        lines.append("Psychiatric Service: Involved during admission")

    labs = extracted.get("lab_abnormalities", [])
    if labs:
        lab_strs = [f"{l.get('test', '?')}: {l.get('finding', '?')}" for l in labs[:5]]
        lines.append(f"Lab Abnormalities: {', '.join(lab_strs)}")

    return "\n".join(lines)
