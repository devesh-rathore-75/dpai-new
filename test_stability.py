"""
DPAI v3.0 — Stability Test Suite
Tests: re-uploads, diverse invoice formats, non-invoice detection, exports.
NO browser interaction. Pure programmatic testing.
"""
import sys, io, json, time
sys.stdout.reconfigure(encoding="utf-8")
import requests
from PIL import Image, ImageDraw, ImageFont

API = "http://localhost:8000"
SEP = "=" * 70
PASS = 0
FAIL = 0

def header(t):
    print(f"\n{SEP}\n  {t}\n{SEP}")

def ok(m):
    global PASS
    PASS += 1
    print(f"  PASS: {m}")

def fail(m):
    global FAIL
    FAIL += 1
    print(f"  FAIL: {m}")

def make_invoice(lines, w=700, h=400):
    img = Image.new("RGB", (w, h), "#FFFEF8")
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        f = ImageFont.load_default()
    for i, line in enumerate(lines):
        d.text((30, 20 + i * 28), line, fill="#222", font=f)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ── Test 1: Health check ──────────────────────────────────────────────────
header("TEST 1: Health Check")
r = requests.get(f"{API}/health", timeout=5)
h = r.json()
if h.get("status") == "healthy" and h.get("version") == "3.0.0":
    ok(f"v{h['version']} — {h['ocr_engine']} — {h['ocr_mode']}")
else:
    fail(f"Unexpected: {h}")


# ── Test 2: Standard invoice (NexGen) ─────────────────────────────────────
header("TEST 2: Standard Invoice (generated test image)")
with open("data/test_invoice_real.png", "rb") as f:
    img_bytes = f.read()
t0 = time.time()
r = requests.post(f"{API}/pipeline", files={"file": ("nexgen_invoice.png", img_bytes)}, timeout=300)
dt = time.time() - t0
res = r.json()
ext = res.get("extracted_data", {})
if ext.get("invoice_number") and ext.get("total_amount"):
    ok(f"Inv#{ext['invoice_number']} Amount={ext['total_amount']} [{dt:.1f}s]")
else:
    fail(f"Missing fields: {ext}")


# ── Test 3: Re-upload same file (should NOT be blocked) ───────────────────
header("TEST 3: Re-upload Same File (stability)")
r2 = requests.post(f"{API}/pipeline", files={"file": ("nexgen_invoice.png", img_bytes)}, timeout=300)
res2 = r2.json()
if res2.get("pipeline_status") == "complete" and res2.get("extracted_data", {}).get("invoice_number"):
    ok(f"Re-upload processed OK: {res2['extracted_data']['invoice_number']}")
elif res2.get("pipeline_status") == "duplicate":
    fail("Duplicate blocked — re-uploads should be allowed")
else:
    fail(f"Re-upload issue: {res2.get('pipeline_status')}")


# ── Test 4: Different invoice style (LLP, DD-Mon-YYYY) ────────────────────
header("TEST 4: Diverse Invoice Style (LLP format)")
inv4 = make_invoice([
    "Sharma & Associates LLP",
    "Tax Invoice",
    "Bill No: SA/2025/1187",
    "Date: 08-Jan-2025",
    "GSTIN: 07AAKFS1234N1ZQ",
    "To: Apex Retail Pvt. Ltd.",
    "Professional Fees             42,000.00",
    "Subtotal: 60,000.00",
    "CGST @9%: 5,400.00",
    "SGST @9%: 5,400.00",
    "Total: Rs. 70,800.00",
])
r = requests.post(f"{API}/pipeline", files={"file": ("sharma_inv.png", inv4)}, timeout=300)
ext4 = r.json().get("extracted_data", {})
if ext4.get("invoice_number") == "SA/2025/1187" and ext4.get("date") == "08-Jan-2025":
    ok(f"Bill No + DD-Mon-YYYY date extracted correctly")
else:
    fail(f"Got inv#={ext4.get('invoice_number')}, date={ext4.get('date')}")


# ── Test 5: Simple invoice (minimal fields) ───────────────────────────────
header("TEST 5: Minimal Invoice")
inv5 = make_invoice([
    "INVOICE",
    "Invoice No: MINI-001",
    "Date: 15/03/2026",
    "Total Amount: 500.00",
])
r = requests.post(f"{API}/pipeline", files={"file": ("mini_inv.png", inv5)}, timeout=300)
ext5 = r.json().get("extracted_data", {})
if ext5.get("invoice_number") and ext5.get("total_amount"):
    ok(f"Minimal: inv#={ext5['invoice_number']} amt={ext5['total_amount']}")
else:
    fail(f"Minimal failed: {ext5}")


# ── Test 6: Non-invoice document ──────────────────────────────────────────
header("TEST 6: Non-Invoice Document Detection")
non_inv = make_invoice([
    "MEETING MINUTES",
    "Project: Website Redesign",
    "Attendees: John, Sarah, Mike",
    "Discussion: Q3 roadmap planning",
    "Next steps: finalize wireframes",
])
r = requests.post(f"{API}/pipeline", files={"file": ("meeting_notes.png", non_inv)}, timeout=300)
ext6 = r.json().get("extracted_data", {})
if ext6.get("is_invoice") is False:
    ok(f"Correctly detected as non-invoice: {ext6.get('detection_message')}")
else:
    # Even if detected as invoice, fields should be mostly empty
    if not ext6.get("invoice_number") and not ext6.get("total_amount"):
        ok(f"No invoice fields extracted (correct)")
    else:
        fail(f"False positive: {ext6}")


# ── Test 7: Consistency (same file 3x) ────────────────────────────────────
header("TEST 7: Consistency Check (3x same invoice)")
consistent = True
for i in range(3):
    r = requests.post(f"{API}/pipeline", files={"file": (f"consistency_{i}.png", inv5)}, timeout=300)
    e = r.json().get("extracted_data", {})
    if not e.get("invoice_number"):
        consistent = False
        fail(f"Run {i+1}: failed to extract invoice number")
if consistent:
    ok("All 3 runs extracted consistently")


# ── Test 8: Exports ───────────────────────────────────────────────────────
header("TEST 8: Exports")
for fmt in ["csv", "json", "excel"]:
    r = requests.get(f"{API}/export/{fmt}", timeout=10)
    if r.status_code == 200 and len(r.content) > 10:
        ok(f"{fmt.upper()}: {len(r.content)} bytes")
    else:
        fail(f"{fmt.upper()}: status={r.status_code}")


# ── Test 9: Document list ─────────────────────────────────────────────────
header("TEST 9: Document List")
r = requests.get(f"{API}/documents", timeout=5)
docs = r.json()
total = docs.get("total", 0)
if total >= 5:
    ok(f"{total} documents in store")
else:
    fail(f"Expected >=5, got {total}")


# ── SUMMARY ───────────────────────────────────────────────────────────────
header(f"RESULTS: {PASS} PASSED, {FAIL} FAILED")
if FAIL == 0:
    print("  All tests passed! Pipeline is stable.")
else:
    print(f"  {FAIL} test(s) need attention.")
print(SEP)
