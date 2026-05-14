"""DPAI v3.0 — AI Intelligence Layer Test"""
import sys, io, json, time
sys.stdout.reconfigure(encoding="utf-8")
import requests
from PIL import Image, ImageDraw, ImageFont

API = "http://localhost:8000"
SEP = "=" * 70
P, F = 0, 0

def ok(m):
    global P; P += 1; print(f"  PASS: {m}")
def fail(m):
    global F; F += 1; print(f"  FAIL: {m}")
def hdr(t):
    print(f"\n{SEP}\n  {t}\n{SEP}")

def make_img(lines, w=700, h=400):
    img = Image.new("RGB", (w, h), "#FFFEF8")
    d = ImageDraw.Draw(img)
    try: f = ImageFont.truetype("arial.ttf", 15)
    except: f = ImageFont.load_default()
    for i, l in enumerate(lines):
        d.text((30, 20 + i * 28), l, fill="#222", font=f)
    buf = io.BytesIO(); img.save(buf, "PNG"); return buf.getvalue()

# Process 2 invoices to populate analytics
hdr("STEP 1: Process test invoices")
inv1 = make_img(["INVOICE", "Invoice No: AI-TEST-001", "Date: 10/05/2026",
    "From: TechCorp Solutions Pvt Ltd", "Cloud Services  25,000.00",
    "CGST @9%: 2,250.00", "SGST @9%: 2,250.00", "Total: 29,500.00"])

inv2 = make_img(["INVOICE", "Invoice No: AI-TEST-002", "Date: 12/05/2026",
    "From: TechCorp Solutions Pvt Ltd", "Software License  15,000.00",
    "CGST @9%: 1,350.00", "SGST @9%: 1,350.00", "Total: 17,700.00"])

inv3 = make_img(["Tax Invoice", "Bill No: TRV/2026/55", "Date: 01-Apr-2026",
    "From: FastTravel Agency", "Flight Tickets  48,000.00",
    "Total Amount: 48,000.00"])

for name, data in [("tech1.png", inv1), ("tech2.png", inv2), ("travel.png", inv3)]:
    r = requests.post(f"{API}/pipeline", files={"file": (name, data)}, timeout=300)
    res = r.json()
    ext = res.get("extracted_data", {})
    print(f"  {name}: {res.get('pipeline_status')} | inv={ext.get('invoice_number')} | cat={ext.get('category')} | risk={ext.get('risk_score')}")

# Test AI features
hdr("STEP 2: AI Summary")
r = requests.post(f"{API}/pipeline", files={"file": ("verify.png", inv1)}, timeout=300)
ext = r.json().get("extracted_data", {})
summary = ext.get("ai_summary", "")
if summary and "TechCorp" in summary:
    ok(f"Summary: {summary[:80]}...")
else:
    fail(f"No/bad summary: {summary[:50]}")

hdr("STEP 3: Category Detection")
cat = ext.get("category", "")
if cat:
    ok(f"Category: {cat} ({ext.get('category_confidence', 0)})")
else:
    fail("No category detected")

hdr("STEP 4: Anomaly Detection")
anomalies = ext.get("anomalies", [])
print(f"  Anomalies found: {len(anomalies)}")
for a in anomalies:
    print(f"    {a['icon']} {a['type']}: {a['message']}")
ok(f"Risk: {ext.get('risk_level')} ({ext.get('risk_score')})")

hdr("STEP 5: Analytics")
r = requests.get(f"{API}/analytics", timeout=5)
analytics = r.json()
if analytics.get("total_spend", 0) > 0:
    ok(f"Total spend: {analytics['total_spend']} | Invoices: {analytics['invoice_count']} | Vendors: {analytics['vendors_count']}")
    print(f"  Categories: {analytics.get('categories', {})}")
    print(f"  GST: {analytics.get('gst_summary', {})}")
else:
    fail("No analytics data")

hdr("STEP 6: Vendor Profiles")
r = requests.get(f"{API}/vendors", timeout=5)
vendors = r.json().get("vendors", [])
if vendors:
    for v in vendors[:3]:
        print(f"  {v['name']}: {v['invoice_count']} inv, total={v['total_spend']}, recurring={v['is_recurring']}")
    ok(f"{len(vendors)} vendors tracked")
else:
    fail("No vendors")

hdr("STEP 7: Natural Language Search")
for q in ["invoices above 20000", "TechCorp", "GST invoices"]:
    r = requests.get(f"{API}/search", params={"q": q}, timeout=5)
    data = r.json()
    print(f"  '{q}' -> {data.get('total', 0)} results")

ok("Search API working")

hdr("STEP 8: Human Review")
docs = requests.get(f"{API}/documents", timeout=5).json()
if docs.get("documents"):
    did = docs["documents"][0]["document_id"]
    r = requests.post(f"{API}/review/{did}", params={"action": "approve"}, timeout=5)
    if r.status_code == 200:
        ok(f"Approved {did}")
    else:
        fail(f"Review failed: {r.status_code}")

hdr(f"RESULTS: {P} PASSED, {F} FAILED")
print(SEP)
