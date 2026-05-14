"""
DPAI — AI Intelligence Service (v3.0)
Smart analysis: summaries, vendor memory, categories, anomalies, fraud detection.
"""

import re
import math
import logging
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  VENDOR MEMORY (in-memory, persists across requests within session)
# ═══════════════════════════════════════════════════════════════════════════

_vendor_db: dict[str, dict] = {}  # normalized_name -> profile


def _normalize_vendor(name: str) -> str:
    if not name:
        return ""
    n = re.sub(r'\s+', ' ', name.strip().upper())
    n = re.sub(r'[._]+$', '', n)  # trailing dots/underscores from OCR
    return n


def update_vendor_profile(vendor_name: str, amount_str: str, date_str: str, doc_id: str):
    """Track vendor across uploads."""
    if not vendor_name:
        return
    key = _normalize_vendor(vendor_name)
    if not key or len(key) < 3:
        return

    amount = 0.0
    if amount_str:
        try:
            amount = float(amount_str.replace(',', '').replace('₹', '').strip())
        except ValueError:
            pass

    if key not in _vendor_db:
        _vendor_db[key] = {
            "name": vendor_name,
            "normalized": key,
            "invoices": [],
            "total_spend": 0.0,
            "invoice_count": 0,
            "first_seen": date_str or "",
            "last_seen": date_str or "",
            "amounts": [],
        }

    profile = _vendor_db[key]
    profile["invoices"].append(doc_id)
    profile["invoice_count"] += 1
    profile["last_seen"] = date_str or profile["last_seen"]
    if amount > 0:
        profile["total_spend"] += amount
        profile["amounts"].append(amount)


def get_vendor_profile(vendor_name: str) -> dict | None:
    key = _normalize_vendor(vendor_name)
    return _vendor_db.get(key)


def get_all_vendors() -> list[dict]:
    vendors = []
    for v in _vendor_db.values():
        avg = v["total_spend"] / v["invoice_count"] if v["invoice_count"] > 0 else 0
        vendors.append({
            "name": v["name"],
            "invoice_count": v["invoice_count"],
            "total_spend": round(v["total_spend"], 2),
            "average_invoice": round(avg, 2),
            "first_seen": v["first_seen"],
            "last_seen": v["last_seen"],
            "is_recurring": v["invoice_count"] >= 2,
        })
    return sorted(vendors, key=lambda x: x["total_spend"], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════
#  CATEGORY DETECTION
# ═══════════════════════════════════════════════════════════════════════════

CATEGORY_KEYWORDS = {
    "Software & IT": ["software", "license", "saas", "cloud", "hosting", "domain", "api", "subscription", "aws", "azure", "server", "database", "app"],
    "Hardware": ["hardware", "laptop", "computer", "monitor", "printer", "keyboard", "mouse", "cable", "router", "equipment"],
    "Consulting": ["consulting", "consultancy", "advisory", "professional fees", "retainer", "strategy"],
    "Travel": ["travel", "flight", "hotel", "cab", "taxi", "airfare", "accommodation", "railway", "booking"],
    "Office Expenses": ["stationery", "furniture", "office", "desk", "chair", "maintenance", "cleaning", "rent"],
    "Utilities": ["electricity", "water", "internet", "broadband", "telecom", "mobile", "phone"],
    "Marketing": ["marketing", "advertising", "promotion", "campaign", "social media", "seo", "branding"],
    "Legal & Compliance": ["legal", "compliance", "audit", "registration", "trademark", "patent"],
}


def detect_category(raw_text: str, vendor_name: str = "") -> tuple[str, float]:
    """Auto-classify invoice by category. Returns (category, confidence)."""
    if not raw_text:
        return "Uncategorized", 0.0

    text_lower = (raw_text + " " + (vendor_name or "")).lower()
    scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits > 0:
            scores[category] = hits

    if not scores:
        return "General Expenses", 0.3

    best = max(scores, key=scores.get)
    confidence = min(0.95, 0.5 + scores[best] * 0.15)
    return best, round(confidence, 2)


# ═══════════════════════════════════════════════════════════════════════════
#  AI INVOICE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def generate_summary(extracted: dict, category: str = "") -> str:
    """Generate intelligent natural language summary."""
    parts = []

    vendor = extracted.get("vendor_name")
    inv_num = extracted.get("invoice_number")
    date = extracted.get("date")
    amount = extracted.get("total_amount")
    gst = extracted.get("gst")

    if vendor:
        parts.append(f"Invoice from **{vendor}**")
    else:
        parts.append("Invoice")

    if inv_num:
        parts[0] += f" (#{inv_num})"

    if date:
        parts.append(f"dated **{date}**")

    if amount:
        parts.append(f"for **₹{amount}**")

    if gst:
        gst_parts = []
        if gst.get("cgst"):
            gst_parts.append(f"CGST ₹{gst['cgst']}")
        if gst.get("sgst"):
            gst_parts.append(f"SGST ₹{gst['sgst']}")
        if gst.get("igst"):
            gst_parts.append(f"IGST ₹{gst['igst']}")
        if gst_parts:
            parts.append(f"including {', '.join(gst_parts)}")

    if category and category != "Uncategorized":
        parts.append(f"— categorized as **{category}**")

    summary = " ".join(parts) + "."

    # Add vendor context
    if vendor:
        profile = get_vendor_profile(vendor)
        if profile and profile["invoice_count"] > 1:
            summary += f" This is invoice #{profile['invoice_count']} from this vendor (total spend: ₹{profile['total_spend']:,.2f})."

    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  ANOMALY & FRAUD DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_anomalies(extracted: dict, all_documents: list[dict]) -> list[dict]:
    """Flag unusual patterns in the invoice."""
    flags = []
    amount_str = extracted.get("total_amount")
    amount = 0.0
    if amount_str:
        try:
            amount = float(amount_str.replace(',', '').replace('₹', '').strip())
        except ValueError:
            pass

    # 1. Unusually high amount
    if amount > 0:
        all_amounts = []
        for doc in all_documents:
            ext = doc.get("extracted_data", {})
            if ext and ext.get("total_amount"):
                try:
                    a = float(ext["total_amount"].replace(',', '').replace('₹', '').strip())
                    if a > 0:
                        all_amounts.append(a)
                except ValueError:
                    pass

        if len(all_amounts) >= 3:
            mean = sum(all_amounts) / len(all_amounts)
            variance = sum((x - mean) ** 2 for x in all_amounts) / len(all_amounts)
            std = math.sqrt(variance) if variance > 0 else 0
            if std > 0 and amount > mean + 2 * std:
                flags.append({
                    "type": "high_amount",
                    "severity": "warning",
                    "message": f"Amount ₹{amount:,.2f} is unusually high (avg: ₹{mean:,.2f})",
                    "icon": "📈",
                })

    # 2. Duplicate invoice number
    inv_num = extracted.get("invoice_number")
    if inv_num:
        dupes = []
        for doc in all_documents:
            ext = doc.get("extracted_data", {})
            if ext and ext.get("invoice_number") == inv_num:
                dupes.append(doc.get("document_id", ""))
        if len(dupes) > 1:
            flags.append({
                "type": "duplicate_invoice",
                "severity": "critical",
                "message": f"Invoice #{inv_num} appears {len(dupes)} times",
                "icon": "🔴",
            })

    # 3. GST mismatch
    gst = extracted.get("gst")
    if gst and amount > 0:
        cgst = 0
        sgst = 0
        try:
            if gst.get("cgst"):
                cgst = float(gst["cgst"].replace(',', '').strip())
            if gst.get("sgst"):
                sgst = float(gst["sgst"].replace(',', '').strip())
        except ValueError:
            pass
        if cgst > 0 and sgst > 0:
            if abs(cgst - sgst) > 1:
                flags.append({
                    "type": "gst_mismatch",
                    "severity": "warning",
                    "message": f"CGST (₹{cgst:,.2f}) ≠ SGST (₹{sgst:,.2f})",
                    "icon": "⚠️",
                })

    # 4. Missing tax details on high-value invoice
    if amount > 10000 and not gst:
        flags.append({
            "type": "missing_gst",
            "severity": "info",
            "message": "High-value invoice without GST details",
            "icon": "ℹ️",
        })

    # 5. Unknown vendor (first time)
    vendor = extracted.get("vendor_name")
    if vendor:
        profile = get_vendor_profile(vendor)
        if profile and profile["invoice_count"] == 1:
            flags.append({
                "type": "new_vendor",
                "severity": "info",
                "message": f"First invoice from {vendor}",
                "icon": "🆕",
            })

    return flags


def compute_risk_score(anomalies: list[dict]) -> tuple[int, str]:
    """Compute fraud risk score 0-100."""
    score = 0
    for a in anomalies:
        if a["severity"] == "critical":
            score += 35
        elif a["severity"] == "warning":
            score += 15
        elif a["severity"] == "info":
            score += 5

    score = min(score, 100)
    if score >= 50:
        level = "High Risk"
    elif score >= 20:
        level = "Medium Risk"
    else:
        level = "Low Risk"
    return score, level


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_analytics(documents: list[dict]) -> dict:
    """Compute business analytics from all processed documents."""
    total_spend = 0.0
    monthly_spend = defaultdict(float)
    vendor_spend = defaultdict(float)
    category_spend = defaultdict(float)
    gst_total = {"cgst": 0.0, "sgst": 0.0, "igst": 0.0}
    valid_count = 0
    warning_count = 0
    invalid_count = 0

    for doc in documents:
        ext = doc.get("extracted_data", {})
        val = doc.get("validation", {})
        if not ext:
            continue

        # Amount
        amount = 0.0
        if ext.get("total_amount"):
            try:
                amount = float(ext["total_amount"].replace(',', '').replace('₹', '').strip())
            except ValueError:
                pass

        total_spend += amount

        # Monthly
        date_str = ext.get("date", "")
        month_key = "Unknown"
        if date_str:
            for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d-%b-%Y"]:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    month_key = dt.strftime("%Y-%m")
                    break
                except ValueError:
                    continue
        monthly_spend[month_key] += amount

        # Vendor
        vendor = ext.get("vendor_name", "Unknown")
        vendor_spend[vendor] += amount

        # Category
        cat = ext.get("category", "General")
        category_spend[cat] += amount

        # GST
        gst = ext.get("gst")
        if gst:
            for k in ["cgst", "sgst", "igst"]:
                if gst.get(k):
                    try:
                        gst_total[k] += float(gst[k].replace(',', '').strip())
                    except ValueError:
                        pass

        # Validation stats
        vs = val.get("status", "")
        if vs == "valid":
            valid_count += 1
        elif vs == "warning":
            warning_count += 1
        else:
            invalid_count += 1

    return {
        "total_spend": round(total_spend, 2),
        "invoice_count": len(documents),
        "avg_invoice": round(total_spend / len(documents), 2) if documents else 0,
        "monthly_spend": dict(sorted(monthly_spend.items())),
        "top_vendors": sorted(
            [{"name": k, "spend": round(v, 2)} for k, v in vendor_spend.items()],
            key=lambda x: x["spend"], reverse=True
        )[:10],
        "categories": dict(sorted(category_spend.items(), key=lambda x: x[1], reverse=True)),
        "gst_summary": {k: round(v, 2) for k, v in gst_total.items()},
        "gst_total": round(sum(gst_total.values()), 2),
        "validation_stats": {"valid": valid_count, "warning": warning_count, "invalid": invalid_count},
        "vendors_count": len(vendor_spend),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  AI ASSISTANT — NATURAL LANGUAGE QUERY ENGINE
# ═══════════════════════════════════════════════════════════════════════════

STOP_WORDS = {"show", "me", "invoices", "from", "the", "all", "invoice", "list",
              "find", "get", "what", "which", "how", "much", "many", "is", "are",
              "my", "our", "any", "give", "tell", "display", "of", "a", "an", "to"}

MONTHS = {"january": "01", "february": "02", "march": "03", "april": "04",
          "may": "05", "june": "06", "july": "07", "august": "08",
          "september": "09", "october": "10", "november": "11", "december": "12",
          "jan": "01", "feb": "02", "mar": "03", "apr": "04",
          "jun": "06", "jul": "07", "aug": "08", "sep": "09",
          "oct": "10", "nov": "11", "dec": "12"}


def _parse_amt(s: str) -> float:
    try:
        return float(s.replace(',', '').replace('₹', '').strip())
    except (ValueError, AttributeError):
        return 0.0


def ai_query(query: str, documents: list[dict]) -> dict:
    """AI assistant: understands questions, filters, and generates answers."""
    if not query:
        return {"answer": "Ask me anything about your invoices!", "results": documents, "query": query}

    q = query.lower().strip()
    all_ext = [(d, d.get("extracted_data", {})) for d in documents if d.get("extracted_data")]

    # ── Analytical questions (return answer, no doc list) ────────────────

    # "Which vendor had highest spend?"
    if any(k in q for k in ["highest spend", "top vendor", "biggest vendor", "most spent"]):
        vendors = get_all_vendors()
        if vendors:
            top = vendors[0]
            answer = f"**{top['name']}** has the highest spend at **₹{top['total_spend']:,.2f}** across {top['invoice_count']} invoice(s)."
            if len(vendors) > 1:
                answer += f"\n\nTop 3 vendors:\n"
                for i, v in enumerate(vendors[:3]):
                    answer += f"{i+1}. **{v['name']}** — ₹{v['total_spend']:,.2f} ({v['invoice_count']} invoices)\n"
            return {"answer": answer, "results": [], "query": query}
        return {"answer": "No vendor data yet. Process some invoices first.", "results": [], "query": query}

    # "How much GST?"
    if "gst" in q and any(k in q for k in ["how much", "total", "summary"]):
        analytics = compute_analytics(documents)
        gs = analytics.get("gst_summary", {})
        answer = f"**GST Summary:**\n- CGST: ₹{gs.get('cgst', 0):,.2f}\n- SGST: ₹{gs.get('sgst', 0):,.2f}\n- IGST: ₹{gs.get('igst', 0):,.2f}\n- **Total GST: ₹{analytics.get('gst_total', 0):,.2f}**"
        return {"answer": answer, "results": [], "query": query}

    # "Total spend" / "how much spent"
    if any(k in q for k in ["total spend", "total amount", "how much spent", "overall spend"]):
        analytics = compute_analytics(documents)
        answer = f"**Total spend: ₹{analytics['total_spend']:,.2f}** across {analytics['invoice_count']} invoices (avg ₹{analytics['avg_invoice']:,.2f})."
        return {"answer": answer, "results": [], "query": query}

    # "Show duplicates" / "duplicate invoices"
    if "duplicate" in q:
        inv_nums = defaultdict(list)
        for d, ext in all_ext:
            inv = ext.get("invoice_number")
            if inv:
                inv_nums[inv].append(d)
        dupes = {k: v for k, v in inv_nums.items() if len(v) > 1}
        if dupes:
            answer = f"**{len(dupes)} duplicate invoice number(s) found:**\n"
            for inv, docs in dupes.items():
                answer += f"- **#{inv}** appears {len(docs)} times\n"
            flat = [d for docs in dupes.values() for d in docs]
            return {"answer": answer, "results": flat, "query": query}
        return {"answer": "✅ No duplicate invoices found.", "results": [], "query": query}

    # "Suspicious" / "risky" / "anomalies"
    if any(k in q for k in ["suspicious", "risky", "risk", "anomal", "fraud", "flag"]):
        flagged = [(d, ext) for d, ext in all_ext if ext.get("risk_score", 0) > 0]
        flagged.sort(key=lambda x: x[1].get("risk_score", 0), reverse=True)
        if flagged:
            answer = f"**{len(flagged)} invoice(s) with risk flags:**\n"
            for d, ext in flagged[:5]:
                answer += f"- **{ext.get('invoice_number', 'N/A')}** from {ext.get('vendor_name', 'Unknown')} — {ext.get('risk_level', '')} (score: {ext.get('risk_score', 0)})\n"
            return {"answer": answer, "results": [d for d, _ in flagged], "query": query}
        return {"answer": "✅ No suspicious invoices detected.", "results": [], "query": query}

    # "Summarize [category] expenses"
    if "summarize" in q or "summary" in q:
        for cat in CATEGORY_KEYWORDS:
            if cat.lower().split()[0] in q:
                matching = [d for d, ext in all_ext if ext.get("category", "").lower().startswith(cat.lower().split()[0])]
                total = sum(_parse_amt(d.get("extracted_data", {}).get("total_amount", "0")) for d in matching)
                answer = f"**{cat} Summary:**\n- {len(matching)} invoice(s)\n- Total: ₹{total:,.2f}"
                return {"answer": answer, "results": matching, "query": query}
        # General summary
        analytics = compute_analytics(documents)
        cats = analytics.get("categories", {})
        answer = f"**Expense Summary ({analytics['invoice_count']} invoices, ₹{analytics['total_spend']:,.2f} total):**\n"
        for cat, amt in cats.items():
            answer += f"- {cat}: ₹{amt:,.2f}\n"
        return {"answer": answer, "results": [], "query": query}

    # "Recurring vendors"
    if "recurring" in q:
        vendors = [v for v in get_all_vendors() if v["is_recurring"]]
        if vendors:
            answer = f"**{len(vendors)} recurring vendor(s):**\n"
            for v in vendors:
                answer += f"- **{v['name']}**: {v['invoice_count']} invoices, ₹{v['total_spend']:,.2f}\n"
            return {"answer": answer, "results": [], "query": query}
        return {"answer": "No recurring vendors yet.", "results": [], "query": query}

    # ── Filter-based queries ────────────────────────────────────────────

    results = []
    amount_gt, amount_lt, month_filter, gst_only, category_filter = None, None, None, False, None

    # Amount filters
    m = re.search(r'(?:above|over|greater|more than|>)\s*(?:₹|rs\.?|inr)?\s*([\d,]+)', q)
    if m:
        amount_gt = float(m.group(1).replace(',', ''))
    m = re.search(r'(?:below|under|less than|<)\s*(?:₹|rs\.?|inr)?\s*([\d,]+)', q)
    if m:
        amount_lt = float(m.group(1).replace(',', ''))

    # Month
    for mname, mnum in MONTHS.items():
        if mname in q:
            month_filter = mnum
            break

    # GST
    if "gst" in q and not any(k in q for k in ["how much", "total", "summary"]):
        gst_only = True

    # Category
    for cat in CATEGORY_KEYWORDS:
        if cat.lower().split()[0] in q:
            category_filter = cat
            break

    # Text search terms
    search_terms = [w for w in q.split() if w not in STOP_WORDS and len(w) > 2
                    and w not in MONTHS and w not in {"above", "below", "over", "under", "gst"}]

    for doc, ext in all_ext:
        amt = _parse_amt(ext.get("total_amount", "0"))
        if amount_gt and amt <= amount_gt:
            continue
        if amount_lt and amt >= amount_lt:
            continue

        if month_filter:
            date_str = ext.get("date", "")
            match = False
            if f"/{month_filter}/" in date_str or f"-{month_filter}-" in date_str:
                match = True
            for mname, mnum in MONTHS.items():
                if mnum == month_filter and mname in date_str.lower():
                    match = True
            if not match:
                continue

        if gst_only and not ext.get("gst"):
            continue

        if category_filter and ext.get("category", "") != category_filter:
            continue

        if search_terms:
            searchable = f"{ext.get('vendor_name', '')} {ext.get('invoice_number', '')} {doc.get('filename', '')} {ext.get('category', '')}".lower()
            if not any(t in searchable for t in search_terms):
                if not (amount_gt or amount_lt or month_filter or gst_only or category_filter):
                    continue

        results.append(doc)

    # Generate contextual answer
    if results:
        total = sum(_parse_amt(d.get("extracted_data", {}).get("total_amount", "0")) for d in results)
        answer = f"Found **{len(results)} invoice(s)** matching your query (total: ₹{total:,.2f})."
    elif amount_gt or amount_lt or month_filter or gst_only or category_filter or search_terms:
        answer = "No invoices match your query. Try different criteria."
    else:
        answer = "I can help you find invoices, analyze spending, detect duplicates, and more. Try asking:\n- *\"invoices above 50000\"*\n- *\"which vendor had highest spend?\"*\n- *\"show duplicates\"*\n- *\"summarize expenses\"*"
        results = documents[:5]

    return {"answer": answer, "results": results, "query": query}


# Keep backward compatibility
def nl_search(query: str, documents: list[dict]) -> list[dict]:
    """Legacy wrapper."""
    result = ai_query(query, documents)
    return result.get("results", [])

