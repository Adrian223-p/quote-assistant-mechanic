"""
Bay 1 — Owner Quote Console
Flask backend that reads the AI-optimized Google Sheet, builds a system
prompt from live data, and calls the Anthropic API.

Memory-optimized for Render free tier (512 MB):
- Sheet data is cached in memory for SHEET_CACHE_TTL seconds (60s default)
- Request size limits enforced
- Conversation history bounded
- Garbage collection forced after each chat request
- Single gunicorn worker recommended (see render.yaml)
"""

import os
import json
import gc
import time
import threading
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from anthropic import Anthropic
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ----- Configuration -----
SHEET_CACHE_TTL         = int(os.environ.get("SHEET_CACHE_TTL", "60"))        # seconds
MAX_REQUEST_BYTES       = int(os.environ.get("MAX_REQUEST_BYTES", "200000"))  # 200 KB (matches client max payload)
MAX_MESSAGES_IN_CONTEXT = int(os.environ.get("MAX_MESSAGES_IN_CONTEXT", "20"))
MAX_MESSAGE_CHARS       = int(os.environ.get("MAX_MESSAGE_CHARS", "8000"))

app = Flask(__name__, static_folder="public", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES

anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SHEET_ID = os.environ["GOOGLE_SHEET_ID"]
_creds_info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
_creds = service_account.Credentials.from_service_account_info(
    _creds_info,
    scopes=["https://www.googleapis.com/auth/spreadsheets"],
)
sheets = build("sheets", "v4", credentials=_creds, cache_discovery=False)

# Free up the credential dict — we don't need to keep it in memory after build()
_creds_info = None

RANGES = [
    "shop_config!A1:D100",
    "labour_rates!A1:G200",
    "parts_catalog!A1:I200",
    "vehicle_multipliers!A1:F50",
    "environmental_fees!A1:F50",
    "job_bundles!A1:D50",
]

# ----- Sheet data cache -----
# CHANGED: lock is now held around the actual fetch to prevent cache stampede.
# Multiple concurrent requests at TTL expiry no longer all hit Google in parallel.
_sheet_cache = {"data": None, "fetched_at": 0}
_cache_lock = threading.Lock()


def _do_fetch():
    """Pulls fresh data from Google Sheets. Caller must hold _cache_lock."""
    resp = (
        sheets.spreadsheets()
        .values()
        .batchGet(
            spreadsheetId=SHEET_ID,
            ranges=RANGES,
            valueRenderOption="UNFORMATTED_VALUE",
        )
        .execute()
    )
    out = {}
    for vr in resp.get("valueRanges", []):
        tab = vr["range"].split("!")[0].strip("'")
        values = vr.get("values", [])
        if not values:
            out[tab] = []
            continue
        headers = [str(h).strip() for h in values[0]]
        records = []
        for row in values[1:]:
            if not row or all(c in ("", None) for c in row):
                continue
            rec = {h: (row[i] if i < len(row) else "") for i, h in enumerate(headers)}
            records.append(rec)
        out[tab] = records
    return out


def fetch_sheet_tables():
    """Cache-stampede-safe sheet fetch. Returns dict of {tab_name: [record_dict, ...]}."""
    with _cache_lock:
        now = time.time()
        if _sheet_cache["data"] is not None and (now - _sheet_cache["fetched_at"]) < SHEET_CACHE_TTL:
            return _sheet_cache["data"]
        # Lock held during fetch — concurrent callers wait, no parallel API calls.
        out = _do_fetch()
        _sheet_cache["data"] = out
        _sheet_cache["fetched_at"] = time.time()
        return out


def invalidate_sheet_cache():
    """Force the next fetch_sheet_tables() call to refresh from Google."""
    with _cache_lock:
        _sheet_cache["data"] = None
        _sheet_cache["fetched_at"] = 0


def _f(v, fb):
    """Coerce a sheet value to float, falling back if it's missing or non-numeric."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return fb


def get_shop_config(records):
    return {r.get("key"): r.get("value") for r in records if r.get("key")}


def fmt_labour_line(r):
    notes = f" — {r['notes']}" if r.get("notes") else ""
    kw = f" [{r['keywords']}]" if r.get("keywords") else ""
    return f"- {r.get('job_id','')} | {r.get('category','')} | {r.get('job_name','')} | {r.get('flat_hours','')}h | {r.get('rate_type','')}{kw}{notes}"


def fmt_parts_line(r):
    prices = []
    if r.get("oem_cost") not in ("", None):         prices.append(f"OEM ${r['oem_cost']}")
    if r.get("aftermarket_cost") not in ("", None): prices.append(f"AFM ${r['aftermarket_cost']}")
    if r.get("reman_cost") not in ("", None):       prices.append(f"Reman ${r['reman_cost']}")
    core = ""
    if r.get("core_charge") not in ("", None, 0, "0"):
        core = f" (core ${r['core_charge']})"
    unit = f" /{r['unit']}" if r.get("unit") else ""
    return f"- {r.get('part_id','')} | {r.get('part_name','')} | {' · '.join(prices)}{core}{unit}"


def fmt_vehicle_line(r):
    kw = f" [{r['keywords']}]" if r.get("keywords") else ""
    return f"- {r.get('category_id','')} | {r.get('category_name','')} | labour ×{r.get('labour_multiplier','')} | parts ×{r.get('parts_multiplier','')}{kw}"


def fmt_fee_line(r):
    applies = f" (applies: {r['applies_when']})" if r.get("applies_when") else ""
    return f"- {r.get('fee_id','')} | {r.get('fee_name','')}: ${r.get('amount','')} /{r.get('unit','')}{applies}"


def fmt_bundle_line(r):
    return f"- {r.get('primary_job','')} → {r.get('related_items','')} ({r.get('rationale','')})"


def build_system_prompt():
    data = fetch_sheet_tables()
    cfg = get_shop_config(data.get("shop_config", []))

    def num(key, default):
        v = cfg.get(key, default)
        try: return float(v)
        except (TypeError, ValueError): return default

    std_rate     = num("standard_labour_rate", 135)
    spec_rate    = num("specialty_labour_rate", 165)
    diag_fee     = num("diagnostic_fee", 135)
    markup       = num("parts_markup", 0.35)
    supplies_pct = num("shop_supplies_pct_of_labour", 0.05)
    supplies_cap = num("shop_supplies_cap", 45)
    rush         = num("rush_premium", 0.25)
    hst          = num("hst_rate", 0.13)

    labour_lines  = "\n".join(fmt_labour_line(r)  for r in data.get("labour_rates", []))
    parts_lines   = "\n".join(fmt_parts_line(r)   for r in data.get("parts_catalog", []))
    vehicle_lines = "\n".join(fmt_vehicle_line(r) for r in data.get("vehicle_multipliers", []))
    fee_lines     = "\n".join(fmt_fee_line(r)     for r in data.get("environmental_fees", []))
    bundle_lines  = "\n".join(fmt_bundle_line(r)  for r in data.get("job_bundles", []))

    # CHANGED: stamp is now actually used (was dead code before)
    stamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    return f"""Current time: {stamp}

You are "Bay 1", the internal pricing assistant for the OWNER of an independent auto repair shop in Ontario, Canada. Your job is to produce ACCURATE, itemized, CAD-priced repair quotes — but parts prices change daily, so you NEVER guess or assume parts prices for the final quote. The owner looks up real-time prices on retailer websites and tells you what they are. You are the calculator and quote formatter.

# SHOP CONFIGURATION
- Standard labour rate: ${std_rate}/hr
- Specialty labour rate: ${spec_rate}/hr  (EV HV work, module programming, deep diagnosis, head gasket, exotic vehicles)
- Diagnostic fee: ${diag_fee} (flat 1 hr — waive if customer proceeds)
- Parts markup: {markup*100:.0f}%
- Shop supplies: {supplies_pct*100:.1f}% of labour, capped at ${supplies_cap}
- Rush premium (if rush = Yes): +{rush*100:.0f}% on labour
- Ontario HST: {hst*100:.1f}% — applies to labour AND parts

# VEHICLE CATEGORY MULTIPLIERS (applied to BOTH labour AND parts)
Format: id | category | labour_multiplier | parts_multiplier | keywords
{vehicle_lines}

# LABOUR RATES (flat-rate hours)
Format: job_id | category | job_name | flat_hours | rate_type | keywords | notes
{labour_lines}

# PARTS CATALOG (REFERENCE ONLY — fallback estimates if owner can't find a real price)
Format: part_id | part_name | OEM/AFM/Reman prices (core if applicable) | unit
{parts_lines}

# ENVIRONMENTAL AND DISPOSAL FEES
Format: fee_id | fee_name | amount | unit | applies_when
{fee_lines}

# JOB BUNDLES (recommend proactively when primary job matches)
{bundle_lines}

# THE WORKFLOW (CRITICAL — TWO STAGES)

## STAGE 1: Preliminary breakdown (FIRST response to a job request)

When the owner first describes a job, you DO NOT give a final quote. Instead you:
1. Identify the vehicle category and multipliers
2. List the labour line items with hours and labour costs (calculated, since labour is fixed)
3. List the PARTS NEEDED but with prices marked "TBC" (To Be Confirmed) — do NOT calculate parts subtotal or grand total
4. List the applicable environmental fees
5. End with a clear ASK telling the owner to click the "Find Parts" button below your message, look up the real prices on the retailer website, then come back and tell you the prices.

Use this EXACT output format for Stage 1:

## QUOTE: <Year Make Model — Job> (PRELIMINARY)

**Vehicle:** <year make model engine> · **Category:** <category_name> (labour ×<LM>, parts ×<PM>) · **Rush:** <Yes/No>

### 🧮 HOW THE MATH WORKS
Show this block BEFORE the labour table so the owner understands exactly how every number is calculated.

> **Step 1 — Effective labour rate**
> Base rate is $<standard_labour_rate>/hr. This vehicle's category multiplier is ×<LM> because <plain reason — e.g. "it's a German luxury vehicle that needs proprietary tools and takes more time per job">.<br>
> So the effective rate is: $<standard_labour_rate> × <LM> = **$<effective_rate>/hr**<br>
> *(If rush is Yes, also multiply by 1 + rush_premium and explain.)*<br>
> *(If any specialty jobs, also show: $<specialty_labour_rate> × <LM> = $<spec_effective>/hr for specialty work.)*
>
> **Step 2 — Each labour line**
> Hours × effective rate = line total. Example for this quote: <hours>h × $<effective_rate> = $<line_total>.
>
> **Step 3 — Parts (Stage 2 — once prices are confirmed)**
> Each part: confirmed_price × <markup+1> (your <markup_pct>% markup) × <PM> (vehicle parts multiplier) × qty.
>
> **Step 4 — Fees**
> Shop supplies = <supplies_pct>% of labour subtotal, capped at $<supplies_cap>. Plus any environmental/disposal fees that apply to this specific job.
>
> **Step 5 — Tax & total**
> All of the above add up to the pre-tax subtotal. Then add HST at <hst_pct>% on top of EVERYTHING (labour, parts, and fees) to get the grand total.

### LABOUR
The "Rate" column shows the EFFECTIVE rate (base rate × vehicle multiplier × rush adjustment), not the base rate.
| Task | Hrs | Rate | Total |
|---|---|---|---|
| ... | ... | $XXX.XX | $XXX.XX |

**Labour subtotal: $X.XX**

### PARTS NEEDED (prices TBC)
| Part | Tier | Qty | Unit Price | Total |
|---|---|---|---|---|
| Front brake rotors | OEM | 2 | TBC | TBC |
| ... | ... | ... | TBC | TBC |

### FEES
| Item | Qty | Rate | Total |
|---|---|---|---|
| Shop supplies | — | — | $X.XX |
| ... (only fees that apply) | ... | ... | ... |

**Fees subtotal: $X.XX**

### NEXT STEP
👉 Click **🔧 Find Parts** below to look up real prices for each part. Then come back and tell me the price you found for each one (e.g., "front rotors $180 each, brake pads $95"), and I'll generate the final quote with the exact total.

**Reference estimate (from internal catalog):** Based on stale catalog prices, this job would estimate around **$X.XX** total — but DO NOT use this number with the customer. Look up current prices first.

**End of preliminary breakdown.**

## STAGE 2: Final quote (after the owner confirms part prices)

When the owner replies with actual part prices (e.g., "front rotors are $180 each, brake pads are $95"), you produce the FINAL quote using those exact prices. Use this format:

## QUOTE: <Year Make Model — Job> (FINAL)

**Vehicle:** <year make model engine> · **Category:** <category_name> (labour ×<LM>, parts ×<PM>) · **Rush:** <Yes/No>

### 🧮 HOW THE MATH WORKS
Show this block BEFORE all the tables, with the ACTUAL numbers worked out for THIS quote (not generic placeholders). Use plain language.

> **Step 1 — Effective labour rate**
> Base rate $<standard_labour_rate>/hr × <LM> (vehicle multiplier — <plain reason>) = **$<effective_rate>/hr**.<br>
> *(If specialty work is in the quote, also show the specialty effective rate the same way.)*<br>
> *(If rush is Yes: also × <1+rush_premium> for rush. Show the math.)*
>
> **Step 2 — Labour line totals**
> Each labour line: hours × $<effective_rate>. For example: <hours>h × $<effective_rate> = $<line_total>.<br>
> All labour lines added together = **labour subtotal of $<labour_subtotal>**.
>
> **Step 3 — Parts**
> For each part: confirmed price × <markup+1> (your <markup_pct>% shop markup) × <PM> (vehicle parts multiplier) × qty.<br>
> Example: $<unit_price> × <1+markup> × <PM> × <qty> = $<part_total>.<br>
> All parts added together = **parts subtotal of $<parts_subtotal>**.
>
> **Step 4 — Fees**
> Shop supplies = <supplies_pct>% × labour subtotal = $<supplies_amount>, capped at $<supplies_cap>.<br>
> Plus any environmental/disposal fees that apply to this job.<br>
> All fees added together = **fees subtotal of $<fees_subtotal>**.
>
> **Step 5 — Tax & total**
> Pre-tax subtotal = labour + parts + fees = $<pretax>.<br>
> HST <hst_pct>% × $<pretax> = $<hst_amount>.<br>
> **Grand total (with tax) = $<grand_total>.**

### LABOUR
Show the EFFECTIVE rate in the Rate column (base rate × LM × rush adjustment).
| Task | Hrs | Rate | Total |
|---|---|---|---|
| ... | ... | $XXX.XX | $XXX.XX |

### PARTS
| Part | Tier | Qty | Unit Price | Total |
|---|---|---|---|---|
| Front brake rotors | OEM | 2 | $180.00 | $360.00 |
| ... | ... | ... | ... | ... |

### FEES
| Item | Qty | Rate | Total |
|---|---|---|---|
| Shop supplies (capped) | — | — | $X.XX |
| ... (only fees that apply) | ... | ... | ... |

### SUMMARY
| | |
|---|---|
| Labour subtotal | $X.XX |
| Parts subtotal | $X.XX |
| Fees subtotal | $X.XX |
| Pre-tax subtotal | $X.XX |
| HST ({hst*100:.0f}%) | $X.XX |

<table><tr class="grand"><td><strong>GRAND TOTAL (CAD)</strong></td><td class="total"><strong>$X.XX</strong></td></tr></table>

### NOTES
- Parts prices confirmed from retailer on <date>
- Bundle recommendations (if applicable)
- Core charges flagged
- Alignment/programming flagged if mandatory but not requested

**End of quote.**

# CALCULATION ORDER (apply for both stages — show all intermediate steps in your reasoning)

## STEP 1 — Vehicle match
Match vehicle to category (use keywords; fuzzy match). Get labour_multiplier (LM) and parts_multiplier (PM).

## STEP 2 — Compute the EFFECTIVE LABOUR RATE first (do this once, then reuse)
This is the trick to getting labour math right. Compute the rate ONCE:
  effective_standard_rate = standard_labour_rate × LM × (1 + rush_premium if rush else 1)
  effective_specialty_rate = specialty_labour_rate × LM × (1 + rush_premium if rush else 1)
Round each effective rate to 2 decimals BEFORE multiplying by hours.

Example (Toyota Camry, mid-size, no rush, $135/hr standard rate, LM = 1.0):
  effective_standard_rate = 135 × 1.00 × 1 = $135.00

Example (BMW 3-series, German luxury, no rush, $135/hr standard rate, LM = 1.30):
  effective_standard_rate = 135 × 1.30 × 1 = $175.50  (NOT 175.5, not 175)

## STEP 3 — Labour line items
For each labour line: line_total = billed_hours × effective_rate (use the rate matching the rate_type)
  - Standard job → use effective_standard_rate
  - Specialty job → use effective_specialty_rate
Compute each line independently. SHOW THE MATH in the table's "Rate" column as the effective rate, not the base rate.

⚠️ ARITHMETIC SELF-CHECK (mandatory):
Before finalizing, recompute each line a second way to verify:
  line_total = billed_hours × effective_rate
  Verify: effective_rate × billed_hours = same number? If not, recalculate.
LLMs often make small arithmetic errors. ALWAYS verify multiplication.

Example: 1.8h × $175.50 = $315.90 (NOT $316.35 — verify by computing 175.5 × 1.8 again).

## STEP 4 — Labour subtotal
Labour subtotal = sum of all labour line totals. Verify the sum a second time.

## STEP 5 — Parts (STAGE 2 ONLY)
For each part: line_total = confirmed_unit_price × (1 + markup) × PM × qty.
The markup and parts multiplier are still applied to the price the owner gave you.
⚠️ EXCEPTION: If the owner says "use this price as-is" or "no markup" then skip both markup and PM.

## STEP 6 — Parts subtotal
Sum of part line totals.

## STEP 7 — Shop supplies
shop_supplies = MIN(labour_subtotal × supplies_pct, supplies_cap)

## STEP 8 — Environmental fees
Add applicable environmental fees only.

## STEP 9 — Pre-tax subtotal
pre_tax = labour_subtotal + parts_subtotal + shop_supplies + env_fees_total

## STEP 10 — HST and Grand Total
hst = pre_tax × {hst}
grand_total = pre_tax + hst

# CRITICAL ARITHMETIC RULES
- Show the EFFECTIVE rate in the labour table (e.g., "$175.50/hr" for a German luxury vehicle), not the base rate.
- Verify EVERY multiplication by computing it a second way. Common error: getting the second decimal wrong.
- Do not chain multiplications without intermediate rounding. Round the rate, THEN multiply by hours.
- If a number looks weird, recompute it. Better to be slow than wrong.
- Use standard arithmetic: 1.8 × 135 = 243. 243 × 1.3 = 315.9. Final: $315.90.

# RULES
- Ontario rust-belt jobs (rust/salt/stuck fasteners mentioned): add 15-30% to labour hours.
- For control arm / ball joint / tie rod / strut work: ALWAYS add a 4-wheel alignment line.
- For timing belt jobs: ALWAYS add water pump, tensioner, idler, coolant flush unless owner explicitly says belt only.
- Default parts tier when not specified: Aftermarket.
- If owner gives prices for SOME parts but not others, ask about the missing ones explicitly. Do not guess.
- If owner skips Stage 1 and says "just give me the full quote with these prices: X, Y, Z" then go directly to Stage 2.
- Never invent or guess part prices. Always wait for owner-confirmed prices before producing a Stage 2 final quote.
- Round every dollar value to 2 decimals.
"""


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        body = request.get_json(force=True) or {}
        messages = body.get("messages", [])
        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        if len(messages) > MAX_MESSAGES_IN_CONTEXT:
            messages = messages[-MAX_MESSAGES_IN_CONTEXT:]

        bounded_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > MAX_MESSAGE_CHARS:
                content = content[:MAX_MESSAGE_CHARS] + "\n\n[...truncated for memory bounds...]"
            bounded_messages.append({"role": msg.get("role", "user"), "content": content})

        system_prompt = build_system_prompt()

        # CHANGED: prompt caching enabled. The system prompt (sheet catalog data,
        # ~20-30K tokens) is identical across requests within the SHEET_CACHE_TTL
        # window, so Anthropic charges 10% of normal input cost on cache hits.
        # Real-world: ~10× input cost reduction at typical shop volume.
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2500,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=bounded_messages,
        )

        result = {
            "content": [
                {"type": b.type, "text": getattr(b, "text", "")}
                for b in response.content
                if getattr(b, "type", None) == "text"
            ]
        }
        return jsonify(result)
    except Exception:
        # CHANGED: don't leak Python exception detail to the client.
        # Full stack trace still goes to Render logs via app.logger.
        app.logger.exception("chat error")
        return jsonify({"error": "Server error. Check logs."}), 500
    finally:
        # CHANGED: removed dead `response = None` (was a no-op — local var about
        # to go out of scope anyway). gc.collect() does the actual work.
        gc.collect()


@app.route("/api/log-quote", methods=["POST"])
def log_quote():
    """Append a quote summary row + full markdown text to the quotes_log tab."""
    try:
        body = request.get_json(force=True) or {}
        # CHANGED: ignore client-supplied timestamp. Server clock is the audit truth;
        # client could otherwise backdate or futuredate quotes.
        row = [
            datetime.utcnow().isoformat(),
            body.get("vehicle", ""),
            body.get("job", ""),
            body.get("labour", ""),
            body.get("parts", ""),
            body.get("fees", ""),
            body.get("hst", ""),
            body.get("total", ""),
            body.get("full_quote", ""),
        ]
        sheets.spreadsheets().values().append(
            spreadsheetId=SHEET_ID,
            range="quotes_log!A:I",
            valueInputOption="RAW",
            body={"values": [row]},
        ).execute()
        return jsonify({"ok": True})
    except Exception:
        app.logger.exception("log-quote error")
        return jsonify({"error": "Server error. Check logs."}), 500


@app.route("/api/past-quotes", methods=["GET"])
def past_quotes():
    """Return all past quotes from the quotes_log tab, newest first."""
    try:
        resp = (
            sheets.spreadsheets()
            .values()
            .get(
                spreadsheetId=SHEET_ID,
                range="quotes_log!A2:I1000",
                valueRenderOption="UNFORMATTED_VALUE",
            )
            .execute()
        )
        rows = resp.get("values", [])
        quotes = []
        for row in rows:
            while len(row) < 9:
                row.append("")
            timestamp, vehicle, job, labour, parts, fees, hst, total, full_quote = row[:9]
            if not timestamp:
                continue
            quotes.append({
                "timestamp": timestamp,
                "vehicle": vehicle,
                "job": job,
                "labour": labour,
                "parts": parts,
                "fees": fees,
                "hst": hst,
                "total": total,
                "full_quote": full_quote,
            })
        quotes.reverse()
        return jsonify({"quotes": quotes, "count": len(quotes)})
    except Exception:
        app.logger.exception("past-quotes error")
        return jsonify({"error": "Server error. Check logs."}), 500


@app.route("/api/config")
def get_config():
    try:
        data = fetch_sheet_tables()
        cfg = get_shop_config(data.get("shop_config", []))
        # CHANGED: coerce to float server-side so the client never receives a
        # string that turns into $NaN. Falls back to documented defaults if missing.
        return jsonify({
            "std_rate":     _f(cfg.get("standard_labour_rate"),       135),
            "spec_rate":    _f(cfg.get("specialty_labour_rate"),      165),
            "diag_fee":     _f(cfg.get("diagnostic_fee"),             135),
            "markup":       _f(cfg.get("parts_markup"),               0.35),
            "supplies_pct": _f(cfg.get("shop_supplies_pct_of_labour"), 0.05),
            "supplies_cap": _f(cfg.get("shop_supplies_cap"),          45),
            "hst":          _f(cfg.get("hst_rate"),                   0.13),
        })
    except Exception:
        app.logger.exception("config error")
        return jsonify({"error": "Server error. Check logs."}), 500


@app.route("/health")
def health():
    """Lightweight health check (used by Render to verify the service is alive)."""
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})


@app.route("/health/memory")
def health_memory():
    """Detailed memory diagnostics. Useful when investigating OOM events."""
    try:
        import resource
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss_kb / 1024  # Linux: KB → MB

        cache_age = None
        with _cache_lock:
            if _sheet_cache.get("fetched_at"):
                cache_age = round(time.time() - _sheet_cache["fetched_at"], 1)
            cache_has_data = _sheet_cache.get("data") is not None

        return jsonify({
            "status": "ok",
            "rss_mb": round(rss_mb, 1),
            "rss_pct_of_512mb": round((rss_mb / 512) * 100, 1),
            "sheet_cache_age_seconds": cache_age,
            "sheet_cache_populated": cache_has_data,
            "sheet_cache_ttl_seconds": SHEET_CACHE_TTL,
            "max_request_bytes": MAX_REQUEST_BYTES,
            "max_messages_in_context": MAX_MESSAGES_IN_CONTEXT,
        })
    except Exception:
        app.logger.exception("health/memory error")
        return jsonify({"error": "Server error. Check logs."}), 500


@app.errorhandler(413)
def request_too_large(e):
    """Friendly error when the user sends a payload larger than MAX_CONTENT_LENGTH."""
    return jsonify({
        "error": f"Request too large. Max payload is {MAX_REQUEST_BYTES} bytes ({MAX_REQUEST_BYTES // 1024} KB). Try clearing the conversation and starting fresh."
    }), 413


@app.errorhandler(500)
def internal_error(e):
    """Force gc on any uncaught 500 error to release memory before returning."""
    gc.collect()
    return jsonify({"error": "Internal server error. Please try again."}), 500


@app.route("/")
def root():
    return send_from_directory("public", "index.html")


@app.after_request
def cleanup_after_request(response):
    # gc.collect() is global and pauses all worker threads, so only run it on
    # the heaviest endpoint (chat). past-quotes and the others release small
    # objects that the next allocation cycle will reclaim anyway.
    if request.path == "/api/chat":
        gc.collect()
    return response


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)

