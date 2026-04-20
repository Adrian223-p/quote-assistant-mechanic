"""
Bay 1 — Owner Quote Console
Flask backend that reads the AI-optimized Google Sheet on every chat request,
builds a system prompt from live data, and calls the Anthropic API.

Expects a Google Sheet with these tabs (all flat tables, row 1 = headers):
  shop_config          (key, value, unit, notes)
  labour_rates         (job_id, category, job_name, flat_hours, rate_type, keywords, notes)
  parts_catalog        (part_id, category, part_name, oem_cost, aftermarket_cost, reman_cost, core_charge, unit, notes)
  vehicle_multipliers  (category_id, category_name, labour_multiplier, parts_multiplier, keywords, notes)
  environmental_fees   (fee_id, fee_name, amount, unit, applies_when, notes)
  job_bundles          (bundle_id, primary_job, related_items, rationale)
  quotes_log           (timestamp, vehicle, job_description, labour, parts, fees, hst, total)

Env vars required:
  ANTHROPIC_API_KEY
  GOOGLE_SHEET_ID
  GOOGLE_SERVICE_ACCOUNT_JSON   (full JSON contents of service account key)
"""

import os
import json
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from anthropic import Anthropic
from google.oauth2 import service_account
from googleapiclient.discovery import build

app = Flask(__name__, static_folder="public", static_url_path="")

anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SHEET_ID = os.environ["GOOGLE_SHEET_ID"]
_creds_info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
_creds = service_account.Credentials.from_service_account_info(
    _creds_info,
    scopes=["https://www.googleapis.com/auth/spreadsheets"],
)
sheets = build("sheets", "v4", credentials=_creds, cache_discovery=False)

RANGES = [
    "shop_config!A1:D100",
    "labour_rates!A1:G200",
    "parts_catalog!A1:I200",
    "vehicle_multipliers!A1:F50",
    "environmental_fees!A1:F50",
    "job_bundles!A1:D50",
]


def fetch_sheet_tables():
    """Pull fresh data on every call. Returns dict of {tab_name: [record_dict, ...]}."""
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

    labour_lines   = "\n".join(fmt_labour_line(r)  for r in data.get("labour_rates", []))
    parts_lines    = "\n".join(fmt_parts_line(r)   for r in data.get("parts_catalog", []))
    vehicle_lines  = "\n".join(fmt_vehicle_line(r) for r in data.get("vehicle_multipliers", []))
    fee_lines      = "\n".join(fmt_fee_line(r)     for r in data.get("environmental_fees", []))
    bundle_lines   = "\n".join(fmt_bundle_line(r)  for r in data.get("job_bundles", []))

    stamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    return f"""You are "Bay 1", the internal pricing assistant for the OWNER of an independent auto repair shop in Ontario, Canada. Your ONLY job is to produce accurate, itemized, CAD-priced repair quotes. All data below was read live from the owner's Google Sheet at {stamp}.

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

# PARTS CATALOG (pre-markup CAD, per unit listed)
Format: part_id | part_name | OEM/AFM/Reman prices (core if applicable) | unit
{parts_lines}

# ENVIRONMENTAL AND DISPOSAL FEES
Format: fee_id | fee_name | amount | unit | applies_when
{fee_lines}

# JOB BUNDLES (recommend proactively when primary job matches)
{bundle_lines}

# CALCULATION ORDER (every quote, in this order)
1. Match vehicle to a category (use the keywords column; fuzzy match). Get labour_multiplier (LM) and parts_multiplier (PM).
2. For each labour line: billed_hours × base_rate × LM × (1 + rush_premium if rush else 1).
   - Use standard_labour_rate for rate_type = standard.
   - Use specialty_labour_rate for rate_type = specialty.
3. Labour subtotal = sum of labour lines.
4. For each part: base_cost × (1 + markup) × PM × qty. Core charges tracked separately (not added to subtotal).
5. Parts subtotal = sum of parts lines.
6. Shop supplies = MIN(labour_subtotal × supplies_pct, supplies_cap).
7. Environmental fees: add any from environmental_fees whose applies_when matches the job.
8. Pre-tax subtotal = labour + parts + supplies + env fees.
9. HST = pre-tax × {hst}.
10. Grand total = pre-tax + HST.

# ADDITIONAL RULES
- Ontario rust-belt jobs (user mentions rust, salt, or stuck fasteners): add 15-30% to labour.
- If a part isn't in parts_catalog, estimate it and prefix the part_name with "EST " so the owner double-checks.
- Default parts tier when user doesn't specify: Aftermarket.
- For control arm / ball joint / tie rod / strut work: ALWAYS add a 4-wheel alignment line (not optional).
- For timing belt jobs: ALWAYS add water pump, tensioner, idler, coolant flush unless owner explicitly says belt only.

# OUTPUT FORMAT (strict — use this exact structure every quote)

## QUOTE: <Year Make Model — Job>

**Vehicle:** <year make model engine> · **Category:** <category_name> (labour ×<LM>, parts ×<PM>) · **Rush:** <Yes/No>

### LABOUR
| Task | Hrs | Rate | Total |
|---|---|---|---|
| ... | ... | ... | ... |

### PARTS
| Part | Tier | Qty | Unit | Total |
|---|---|---|---|---|
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
- Bundle recommendations (if applicable)
- Parts tier assumption if user didn't specify
- Core charges flagged
- Alignment/programming flagged if mandatory but not requested

End every quote with this on its own line: **End of quote.**
"""


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        body = request.get_json(force=True) or {}
        messages = body.get("messages", [])
        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        system_prompt = build_system_prompt()

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2500,
            system=system_prompt,
            messages=messages,
        )
        return jsonify({
            "content": [
                {"type": b.type, "text": getattr(b, "text", "")}
                for b in response.content
                if getattr(b, "type", None) == "text"
            ]
        })
    except Exception as e:
        app.logger.exception("chat error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/log-quote", methods=["POST"])
def log_quote():
    try:
        body = request.get_json(force=True) or {}
        row = [
            body.get("timestamp", datetime.utcnow().isoformat()),
            body.get("vehicle", ""),
            body.get("job", ""),
            body.get("labour", ""),
            body.get("parts", ""),
            body.get("fees", ""),
            body.get("hst", ""),
            body.get("total", ""),
        ]
        sheets.spreadsheets().values().append(
            spreadsheetId=SHEET_ID,
            range="quotes_log!A:H",
            valueInputOption="USER_ENTERED",
            body={"values": [row]},
        ).execute()
        return jsonify({"ok": True})
    except Exception as e:
        app.logger.exception("log-quote error")
        return jsonify({"error": str(e)}), 500


@app.route("/api/config")
def get_config():
    try:
        data = fetch_sheet_tables()
        cfg = get_shop_config(data.get("shop_config", []))
        return jsonify({
            "std_rate":     cfg.get("standard_labour_rate", 135),
            "spec_rate":    cfg.get("specialty_labour_rate", 165),
            "diag_fee":     cfg.get("diagnostic_fee", 135),
            "markup":       cfg.get("parts_markup", 0.35),
            "supplies_pct": cfg.get("shop_supplies_pct_of_labour", 0.05),
            "supplies_cap": cfg.get("shop_supplies_cap", 45),
            "hst":          cfg.get("hst_rate", 0.13),
        })
    except Exception as e:
        app.logger.exception("config error")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})


@app.route("/")
def root():
    return send_from_directory("public", "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
