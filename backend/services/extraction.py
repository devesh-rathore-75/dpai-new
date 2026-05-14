"""
DPAI — Data Extraction Service (v2.0)
Parses structured invoice fields from raw OCR text.
Multi-pattern regex with confidence scoring.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  PATTERN MATCHING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def _find_best(patterns: list[tuple[str, float]], text: str) -> tuple[Optional[str], float]:
    """
    Try multiple regex patterns with assigned confidence weights.
    Returns (value, confidence) for the best match.
    patterns: list of (regex_pattern, base_confidence)
    """
    for pattern, conf in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            value = match.group(1).strip()
            if value:
                return value, conf
    return None, 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  FIELD EXTRACTORS
# ═══════════════════════════════════════════════════════════════════════════

def extract_invoice_number(text: str) -> tuple[Optional[str], float]:
    """Extract invoice number with confidence score."""
    patterns = [
        # High confidence: explicit label + value
        (r"invoice\s*(?:no|number|#|num)[.:;\s]*([A-Z0-9][\w\-/]{2,20})", 0.95),
        (r"\binv\b[.\-#]?\s*(?:no|number|#)?[.:;\s]*([A-Z0-9][\w\-/]{2,20})", 0.90),
        (r"bill\s*(?:no|number|#)[.:;\s]*([A-Z0-9][\w\-/]{2,20})", 0.85),
        (r"receipt\s*(?:no|number|#)[.:;\s]*([A-Z0-9][\w\-/]{2,20})", 0.80),
        (r"reference\s*(?:no|number|#)?[.:;\s]*([A-Z0-9][\w\-/]{2,20})", 0.70),
        # Lower confidence: standalone patterns
        (r"(?:^|\n)\s*(INV[\-/]?\d{3,}[\w\-]*)", 0.75),
        (r"(?:^|\n)\s*#\s*([A-Z0-9][\w\-/]{3,15})", 0.60),
    ]
    return _find_best(patterns, text)


def extract_date(text: str) -> tuple[Optional[str], float]:
    """Extract date with confidence score."""
    patterns = [
        # High confidence: labeled dates
        (r"(?:invoice\s*date|inv\.?\s*date)[.:;\s]*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})", 0.95),
        (r"(?:invoice\s*date|inv\.?\s*date)[.:;\s]*(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})", 0.95),
        (r"(?:invoice\s*date|inv\.?\s*date)[.:;\s]*([A-Za-z]+\.?\s+\d{1,2},?\s+\d{4})", 0.95),
        # Labeled: DD-Mon-YYYY (Indian format)
        (r"(?:invoice\s*date|inv\.?\s*date)[.:;\s]*(\d{1,2}[\-/][A-Za-z]{3,9}[\-/]\d{4})", 0.95),
        (r"\bdate[.:;\s]*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})", 0.85),
        (r"\bdate[.:;\s]*(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})", 0.85),
        (r"\bdate[.:;\s]*([A-Za-z]+\.?\s+\d{1,2},?\s+\d{4})", 0.85),
        # Labeled: DD-Mon-YYYY
        (r"\bdate[.:;\s]*(\d{1,2}[\-/][A-Za-z]{3,9}[\-/]\d{4})", 0.85),
        (r"\bdated[.:;\s]*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})", 0.80),
        # Lower: standalone date patterns
        (r"(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})", 0.50),
        (r"(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})", 0.50),
        (r"(\d{1,2}[\-/][A-Za-z]{3,9}[\-/]\d{4})", 0.50),
        (r"([A-Za-z]+\s+\d{1,2},?\s+\d{4})", 0.45),
    ]
    return _find_best(patterns, text)


def extract_vendor_name(text: str) -> tuple[Optional[str], float]:
    """Extract vendor/seller name with confidence score."""
    # Strategy 1: Labeled patterns
    labeled_patterns = [
        (r"(?:from|seller|vendor|supplier|company)\s*[.:;\s]*\n?\s*(.+?)(?:\n|$)", 0.90),
        (r"(?:billed?\s*by|sold\s*by)\s*[.:;\s]*\n?\s*(.+?)(?:\n|$)", 0.90),
        (r"(?:issued\s*by)\s*[.:;\s]*\n?\s*(.+?)(?:\n|$)", 0.85),
    ]

    for pattern, conf in labeled_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            name = match.group(1).strip()
            name = re.sub(r"[,:;]+$", "", name).strip()
            if 3 < len(name) < 100:
                return name, conf

    # Strategy 2: Business suffix detection in first 15 lines
    business_suffixes = r"\b(pvt\.?\s*ltd|ltd|llp|inc|corp|co\.|solutions|enterprises|services|technologies|consulting|systems|infosys|tech|global|india)\b"
    lines = text.split("\n")[:15]
    for line in lines:
        line = line.strip()
        if re.search(business_suffixes, line, re.IGNORECASE):
            clean = re.sub(r"[,:;]+$", "", line).strip()
            # Skip lines that are obviously not company names
            if 5 < len(clean) < 80 and not re.match(r"^\d", clean):
                return clean, 0.70

    # Strategy 3: First non-empty substantial line (very low confidence)
    skip_patterns = r"^(---|invoice|tax|bill|receipt|date|to|from|page|\d+$|\s*$)"
    for line in lines[:5]:
        line = line.strip()
        if len(line) > 10 and not re.match(skip_patterns, line, re.IGNORECASE):
            return line[:60], 0.30

    return None, 0.0


def extract_total_amount(text: str) -> tuple[Optional[str], float]:
    """Extract total amount with intelligent multi-strategy approach."""
    import re

    def _parse_amount(raw: str) -> float:
        """Parse amount string handling Indian format, OCR errors."""
        s = raw.replace(',', '').replace(' ', '')
        s = re.sub(r'^[₹$]', '', s)
        s = s.replace('I', '1').replace('O', '0').replace('o', '0').replace('l', '1')
        try:
            return float(s)
        except ValueError:
            return 0.0

    candidates = []  # (display_str, numeric_val, confidence)

    # Strategy 1: Labeled patterns (highest confidence)
    labeled = [
        (r"(?:grand\s*total|total\s*amount\s*(?:due|payable)?)\s*[.:;=\s]*(?:₹|rs\.?|inr)?\s*([\d,]+\.?\d*)", 0.95),
        (r"(?:amount\s*(?:due|payable|total))\s*[.:;=\s]*(?:₹|rs\.?|inr)?\s*([\d,]+\.?\d*)", 0.92),
        (r"(?:net\s*(?:amount|payable|total))\s*[.:;=\s]*(?:₹|rs\.?|inr)?\s*([\d,]+\.?\d*)", 0.90),
        (r"(?:invoice\s*total)\s*[.:;=\s]*(?:₹|rs\.?|inr)?\s*([\d,]+\.?\d*)", 0.90),
        (r"(?:balance\s*due)\s*[.:;=\s]*(?:₹|rs\.?|inr)?\s*([\d,]+\.?\d*)", 0.88),
    ]
    for pat, conf in labeled:
        for m in re.finditer(pat, text, re.IGNORECASE):
            val = _parse_amount(m.group(1))
            if val > 0:
                candidates.append((m.group(1).strip(), val, conf))

    # Strategy 2: "Total:" on same line
    for m in re.finditer(r"\bTotal\s*[.:;=\s]*(?:₹|rs\.?|inr)?\s*([\d,]+\.?\d+)", text, re.IGNORECASE):
        if not re.search(r"sub\s*total", text[max(0, m.start()-10):m.start()], re.IGNORECASE):
            val = _parse_amount(m.group(1))
            if val > 0:
                candidates.append((m.group(1).strip(), val, 0.80))

    # Strategy 3: Multi-line scan around TOTAL/GRAND TOTAL labels
    lines = text.split("\n")
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        is_total_line = bool(re.search(r"\b(grand\s*total|total\s*amount|net\s*total|amount\s*payable)\b", line_lower))
        is_simple_total = bool(re.search(r"\btotal\b", line_lower)) and not re.search(r"sub\s*total", line_lower)

        if is_total_line or is_simple_total:
            conf = 0.85 if is_total_line else 0.70
            # Scan the total line + 2 lines before/after
            for j in range(max(0, i - 2), min(i + 3, len(lines))):
                for raw in re.findall(r"([\d,]+\.\d{2})", lines[j]):
                    val = _parse_amount(raw)
                    if val > 10:
                        candidates.append((raw, val, conf))

    # Strategy 4: Rs. / ₹ prefixed amounts anywhere
    for m in re.finditer(r"(?:₹|Rs\.?|INR)\s*([\d,]+\.?\d*)", text, re.IGNORECASE):
        val = _parse_amount(m.group(1))
        if val > 100:
            candidates.append((m.group(1).strip(), val, 0.45))

    # Strategy 5: Largest decimal amount as last resort
    if not candidates:
        all_nums = re.findall(r"([\d,]+\.\d{2})", text)
        if all_nums:
            best = max(all_nums, key=lambda x: _parse_amount(x))
            val = _parse_amount(best)
            if val > 50:
                candidates.append((best, val, 0.30))

    if candidates:
        # Prefer highest confidence, then highest value
        candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
        return candidates[0][0], candidates[0][2]

    return None, 0.0


def extract_gst(text: str) -> tuple[dict, float]:
    """Extract GST with intelligent multi-line and fuzzy matching."""
    result = {}
    confidences = []

    # Fix common OCR errors in text for GST parsing
    gst_text = text.replace('|', 'I')

    # GSTIN (15-char alphanumeric)
    gstin_match = re.search(
        r"(?:GSTIN|GST\s*(?:No|Number|IN|Reg))[.:;\s]*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][A-Z0-9][Z][A-Z0-9])",
        gst_text, re.IGNORECASE
    )
    if gstin_match:
        result["gstin"] = gstin_match.group(1).upper()
        confidences.append(0.95)
    else:
        gstin_match2 = re.search(r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][A-Z0-9]Z[A-Z0-9])\b", gst_text)
        if gstin_match2:
            result["gstin"] = gstin_match2.group(1).upper()
            confidences.append(0.75)

    # Tax components - multi-strategy
    tax_labels = {
        "cgst": [r"CGST", r"C\.?G\.?S\.?T", r"Central\s*GST"],
        "sgst": [r"SGST", r"S\.?G\.?S\.?T", r"State\s*GST"],
        "igst": [r"IGST", r"I\.?G\.?S\.?T", r"Integrated\s*GST"],
    }

    lines = gst_text.split("\n")
    for key, label_patterns in tax_labels.items():
        found = False
        for label_pat in label_patterns:
            # Same-line: CGST @9%: 1,234.00
            pat = label_pat + r"\s*(?:@\s*\d+\.?\d*\s*%)?[.:;=\s]*(?:₹|rs\.?)?\s*([\d,]+\.?\d*)"
            m = re.search(pat, gst_text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if val and float(val.replace(',', '') or 0) > 0:
                    result[key] = val
                    confidences.append(0.90)
                    found = True
                    break

            # Multi-line: label on one line, amount on next
            if not found:
                for i, line in enumerate(lines):
                    if re.search(label_pat, line, re.IGNORECASE):
                        # Check same line for amount
                        amounts = re.findall(r"([\d,]+\.\d{2})", line)
                        if amounts:
                            result[key] = amounts[-1]
                            confidences.append(0.85)
                            found = True
                            break
                        # Check next line
                        if i + 1 < len(lines):
                            amounts = re.findall(r"([\d,]+\.\d{2})", lines[i + 1])
                            if amounts:
                                result[key] = amounts[0]
                                confidences.append(0.75)
                                found = True
                                break
            if found:
                break

    # Total tax/GST
    total_tax_m = re.search(r"(?:total\s*)?(?:tax|gst)\s*(?:amount)?[.:;=\s]*(?:₹|rs\.?)?\s*([\d,]+\.?\d*)", gst_text, re.IGNORECASE)
    if total_tax_m:
        val = total_tax_m.group(1).strip()
        if val and float(val.replace(',', '') or 0) > 0:
            result["total_gst"] = val
            confidences.append(0.80)

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return result, round(avg_conf, 2)


# ═══════════════════════════════════════════════════════════════════════════
#  DOCUMENT TYPE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_invoice(text: str) -> tuple[bool, str]:
    """Check if text looks like an invoice. Returns (is_invoice, reason)."""
    if not text:
        return False, "No text extracted from document"

    text_lower = text.lower()
    invoice_keywords = [
        "invoice", "bill", "receipt", "tax", "total", "amount",
        "subtotal", "gst", "cgst", "sgst", "igst", "payment",
        "due", "qty", "rate", "price", "discount",
    ]
    hits = sum(1 for kw in invoice_keywords if kw in text_lower)

    if hits >= 3:
        return True, f"Invoice detected ({hits} keyword matches)"
    elif hits >= 1:
        return True, f"Possible invoice ({hits} keyword matches, low confidence)"
    else:
        return False, "Document does not appear to be an invoice (no invoice keywords found)"


# ═══════════════════════════════════════════════════════════════════════════
#  MASTER EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_all(text: str) -> dict:
    """
    Master extraction function.
    Returns dict with all extracted fields, confidence scores, and document info.
    """
    empty_result = {
        "invoice_number": None,
        "date": None,
        "vendor_name": None,
        "total_amount": None,
        "gst": None,
        "confidence": {},
        "overall_confidence": 0.0,
        "is_invoice": False,
        "detection_message": "No text to analyze",
        "raw_text_preview": "",
        "raw_text_length": 0,
    }

    if not text or not text.strip():
        return empty_result

    # Document type detection
    is_invoice, detection_msg = detect_invoice(text)

    inv_num, inv_conf = extract_invoice_number(text)
    date_val, date_conf = extract_date(text)
    vendor, vendor_conf = extract_vendor_name(text)
    amount, amount_conf = extract_total_amount(text)
    gst_info, gst_conf = extract_gst(text)

    confidences = {
        "invoice_number": round(inv_conf, 2),
        "date": round(date_conf, 2),
        "vendor_name": round(vendor_conf, 2),
        "total_amount": round(amount_conf, 2),
        "gst": round(gst_conf, 2),
    }

    non_zero = [c for c in confidences.values() if c > 0]
    overall = round(sum(non_zero) / len(non_zero), 2) if non_zero else 0.0

    # Build preview (first 500 chars, cleaned)
    preview_lines = [l.strip() for l in text.split("\n") if l.strip()][:15]
    preview = "\n".join(preview_lines)

    return {
        "invoice_number": inv_num,
        "date": date_val,
        "vendor_name": vendor,
        "total_amount": amount,
        "gst": gst_info if gst_info else None,
        "confidence": confidences,
        "overall_confidence": overall,
        "is_invoice": is_invoice,
        "detection_message": detection_msg,
        "raw_text_preview": preview[:500],
        "raw_text_length": len(text),
    }
