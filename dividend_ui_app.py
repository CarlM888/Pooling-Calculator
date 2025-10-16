#!/usr/bin/env python3
# Streamlit UI for Dividend Calculation (National Tote-style) with Dead-heat Matrix builder
# Run with: streamlit run dividend_ui_app.py

import math
from decimal import Decimal, ROUND_FLOOR, getcontext
getcontext().prec = 28


# --- Guard to ensure FFR helpers exist before any use ---
try:
    ffr_declared_dividend
    ffr_approximate_dividend
except NameError:
    def ffr_declared_dividend(net: float, per1_units: float, rules):
        from decimal import Decimal, ROUND_FLOOR
        try:
            if per1_units is None or float(per1_units) <= 0 or float(net) <= 0:
                return 0.0
        except Exception:
            return 0.0
        dp = int(getattr(rules, "display_dp", 2))
        step_c = int(round(float(getattr(rules, "break_step", 0.10)) * 100))
        if step_c <= 0:
            step_c = 10
        net_c = (Decimal(str(net)) * 100).to_integral_value(rounding=ROUND_FLOOR)
        per_unit_c = (net_c / Decimal(str(per1_units))).to_integral_value(rounding=ROUND_FLOOR)
        mode = getattr(rules, "break_mode", "down")
        if mode == "down":
            per_unit_c -= per_unit_c % step_c
        elif mode == "up":
            rem = per_unit_c % step_c
            if rem:
                per_unit_c += (step_c - rem)
        else:
            rem = per_unit_c % step_c
            if rem >= (step_c // 2):
                per_unit_c += (step_c - rem)
            else:
                per_unit_c -= rem
        return float((per_unit_c / Decimal(100)).quantize(Decimal(10) ** -dp))

    def ffr_approximate_dividend(net: float, per1_units: float, rules):
        return ffr_declared_dividend(net, per1_units, rules)
# --- End guard ---




# --- Safety guard: ensure tri_approximate_dividend exists before any UI uses it ---
try:
    tri_approximate_dividend
except NameError:  # define if missing for any reason
    def tri_approximate_dividend(net: float, triple_units: float, rules) -> float:
        """Approx/WillPay per $1 for a triple (no min-div floor), cents-first."""
        if not triple_units or triple_units <= 0 or net <= 0:
            return 0.0
        dp = int(getattr(rules, "display_dp", 2))
        step_c = int(round(float(getattr(rules, "break_step", 0.10)) * 100))
        if step_c <= 0:
            step_c = 10
        net_c = (Decimal(str(net)) * 100).to_integral_value(rounding=ROUND_FLOOR)
        per_unit_c = (net_c / Decimal(str(triple_units))).to_integral_value(rounding=ROUND_FLOOR)
        mode = getattr(rules, "break_mode", "down")
        if mode == "down":
            per_unit_c = per_unit_c - (per_unit_c % step_c)
        elif mode == "up":
            rem = per_unit_c % step_c
            if rem:
                per_unit_c = per_unit_c + (step_c - rem)
        else:  # nearest
            rem = per_unit_c % step_c
            if rem >= (step_c // 2):
                per_unit_c = per_unit_c + (step_c - rem)
            else:
                per_unit_c = per_unit_c - rem
        return float((per_unit_c / Decimal(100)).quantize(Decimal(10) ** -dp))
# --- End guard ---
from dataclasses import dataclass, asdict
from typing import Dict, List
import json
import itertools

import pandas as pd
import streamlit as st

# === Collation parser (supports horizontal 1–4, 5–8, 9–11 and vertical) ===
import re

# Map border glyphs to spaces to preserve token boundaries
_BOX_CHARS = "│┼┤├┬┴┌┐└┘─━—│┃╭╮╯╰▕▏|"
_TRANS_TABLE = str.maketrans({c: " " for c in _BOX_CHARS})

def parse_collation_text(text: str, field_index: int = 1, scale: float = 1.0, force_horizontal: bool=False):
    """
    Parses a WIN collation dump.
      - If force_horizontal=True, treat each row as: start_runner followed by up to 4 consecutive runner values.
      - Otherwise, auto-detect: horizontal if rows have >1 numeric values; else vertical per-runner.
      - field_index is only used in vertical mode (1..4).
    Returns: (df, meta) where df has columns Runner, Units; meta has total, num_runners, layout.
    """
    # Clean: remove borders -> spaces; collapse extra spaces within lines
    t = "\n".join(" ".join(line.translate(_TRANS_TABLE).split()) for line in text.splitlines())

    # Meta
    total = None
    m_total = re.search(r"^\s*TOTAL\s*:?\s*\$?\s*([-\d.,]+)", t, re.M)
    if m_total:
        try: total = float(m_total.group(1).replace(",", ""))
        except Exception: total = None

    num_runners = None
    for m in re.finditer(r"NUM RUNNERS\s+(\d+)", t):
        try: num_runners = int(m.group(1))
        except Exception: pass

    # Row capture: start id + any number of numeric columns (horizontal)
    # Example row: "1 100.000000 100.000000 100.000000 100.000000 100.000000 100.000000"
    # Row capture: support multiple start-id groups per line (e.g., "1 ... 5 ... 9 ...")
    _line_tokens = [" ".join(line.translate(_TRANS_TABLE).split()) for line in text.splitlines()]
    HGROUP = re.compile(r"(?<!\S)(\d{1,2})\s+((?:[\d,]*\.\d+|\d+)(?:\s+(?:[\d,]*\.\d+|\d+))*)")
    rows = []
    for _ln in _line_tokens:
        for m in HGROUP.finditer(_ln):
            start_id = int(m.group(1))
            vals = re.findall(r"[\d,]*\.\d+|\d+", m.group(2))
            rows.append((start_id, vals))

    # Decide layout
    horizontal = force_horizontal or any(len(v) > 1 for _, v in rows)

    last = {}
    if horizontal:
        for start, vals in rows:
            for i, s in enumerate(vals):
                rid = start + i
                try:
                    v = float(s.replace(",", "")) * scale
                except Exception:
                    continue
                if num_runners and rid > num_runners:
                    break
                last[rid] = v
    else:
        for start, vals in rows:
            idx = max(0, min(field_index - 1, len(vals) - 1))
            try:
                v = float(vals[idx].replace(",", "")) * scale
            except Exception:
                continue
            last[start] = v

    # Build DataFrame
    import pandas as pd
    if num_runners:
        data = [(n, float(last.get(n, 0.0))) for n in range(1, num_runners + 1)]
    else:
        data = sorted((k, float(v)) for k, v in last.items())

    df = pd.DataFrame(data, columns=["RunnerNo", "Units"])
    df["Runner"] = df["RunnerNo"].apply(lambda n: f"Runner {n}")
    return df[["Runner", "Units"]], {"total": total, "num_runners": num_runners, "layout": "horizontal" if horizontal else "vertical"}
# === End parser ===


def parse_meta_from_collation(text: str) -> dict:
    """Extract entries at deadline (NUM RUNNERS first int) and SCRATCHINGS value (for refunds prefill)."""
    t_lines = [" ".join(line.translate(_TRANS_TABLE).split()) for line in text.splitlines()]
    t = "\n".join(t_lines)
    m = re.search(r"NUM RUNNERS\s+(\d+)", t)
    entries_at_deadline = int(m.group(1)) if m else None
    m2 = re.search(r"SCRATCHINGS\s+([\d,]+(?:\.\d+)?)", t)
    scratchings_value = float(m2.group(1).replace(",", "")) if m2 else 0.0
    return {"entries_at_deadline": entries_at_deadline, "scratchings_value": scratchings_value}
# === PLA (Place) Parser & Helpers ===
def parse_pla_collation_text(text: str, scale: float = 1.0):
    """Parse a PLA collation dump into a DataFrame with columns: Runner, Units.
    Also returns meta dict with total and num_runners (if present).
    Supports multiple numeric columns per row (horizontal layout) and multiple start-id groups per line.
    """
    # Clean lines (same normalisation used elsewhere)
    t_lines = [" ".join(line.translate(_TRANS_TABLE).split()) for line in text.splitlines()]
    t = "\n".join(t_lines)

    # TOTAL and NUM RUNNERS
    total = None
    total_patterns = [
        r"^\s*TOTAL\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*$",
        r"^\s*TOTAL\s+(?:SALES|POOL|TURNOVER)\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*$",
        r"^\s*GROSS\s+SALES\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*$",
        r"^\s*POOL\s+TOTAL\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*$",
    ]
    cands = []
    for pat in total_patterns:
        cands.extend(re.findall(pat, t, re.M))
    if cands:
        try:
            total = float(cands[-1].replace(',', ''))
        except Exception:
            total = None

    num_runners = None
    m = re.search(r"NUM RUNNERS\s+(\d+)", t)
    if m:
        try:
            num_runners = int(m.group(1))
        except Exception:
            pass

    # Per-line group detection for runner rows: e.g. "1 8000.000000 0 0 ..." and also avoid matching the NUM RUNNERS line.
    HGROUP = re.compile(r"(?<!\S)(\d{1,2})\s+((?:[\d,]*\.\d+|\d+)(?:\s+(?:[\d,]*\.\d+|\d+))*)")
    rows = []
    for ln in t_lines:
        if "NUM RUNNERS" in ln:
            continue
        for m in HGROUP.finditer(ln):
            start_id = int(m.group(1))
            vals = re.findall(r"[\d,]*\.\d+|\d+", m.group(2))
            rows.append((start_id, vals))

    # Assemble last-seen units by runner, mapping horizontally across columns
    last = {}
    for start_id, vals in rows:
        for i, s in enumerate(vals):
            rid = start_id + i
            try:
                v = float(s.replace(",", "")) * scale
            except Exception:
                continue
            if num_runners and rid > num_runners:
                break
            last[rid] = v

    # To DataFrame (ensure 1..num_runners)
    import pandas as pd
    if num_runners:
        data = [(n, float(last.get(n, 0.0))) for n in range(1, num_runners + 1)]
    else:
        data = sorted((k, float(v)) for k, v in last.items())
    df = pd.DataFrame(data, columns=["RunnerNo", "Units"])
    df["Runner"] = df["RunnerNo"].apply(lambda n: f"Runner {int(n)}")
    df = df[["Runner", "Units"]]
    return df, {"total": total, "num_runners": num_runners}

def pla_declared_dividend(net: float, units: float, place_winners: int, rules: "PoolRules") -> float:
    """Declared PLA dividend per $1 for a single selection.
    Assumes the net pool is split equally by the number of place winners (2 or 3),
    then divided by the winning runner's units.
    """
    if units <= 0 or place_winners <= 0:
        return 0.0
    raw = (net / place_winners) / units
    div = _apply_breakage(raw, rules.break_step, rules.break_mode)
    div = max(div, rules.min_div)
    return _format_display(div, rules.display_dp)

def pla_declared_dividends_with_deficiency(
    net: float,
    winners_units: dict[int, float],
    place_winners: int,
    rules: "PoolRules",
) -> dict[int, float]:
    """Compute declared PLA dividends for multiple winners with deficiency handling.
    If any raw part is < $1 per $1, top it up to $1 by proportionally deducting dollars
    from parts that are >= $1 before applying breakage and min-div.
    """
    if place_winners <= 0 or not winners_units:
        return {}
    # Raw dividends per $1 stake ignoring deficiency
    part_pool = net / place_winners
    raw: dict[int, float] = {r: (part_pool / u) if u > 0 else 0.0 for r, u in winners_units.items()}

    # Dollars needed to raise deficient parts to $1
    need_pool = {r: (1.0 - d) * winners_units[r] for r, d in raw.items() if 0.0 < d < 1.0}
    total_need = sum(need_pool.values()) if need_pool else 0.0

    # Dollars above $1 available from other parts
    surplus_pool = {r: (d - 1.0) * winners_units[r] for r, d in raw.items() if d >= 1.0}
    total_surplus = sum(v for v in surplus_pool.values() if v > 0.0)

    if total_need > 0.0 and total_surplus > 0.0:
        # Proportionally deduct from surplus parts
        for r, surplus in list(surplus_pool.items()):
            if surplus <= 0.0:
                continue
            take = total_need * (surplus / total_surplus)
            part_dollars = raw[r] * winners_units[r]
            new_part_dollars = max(part_dollars - take, winners_units[r] * 1.0)
            raw[r] = new_part_dollars / winners_units[r]
        # Set all deficient parts to $1
        for r in need_pool:
            raw[r] = 1.0

    final: dict[int, float] = {}
    for r, d in raw.items():
        div = _apply_breakage(d, rules.break_step, rules.break_mode)
        div = max(div, rules.min_div)
        final[r] = _format_display(div, rules.display_dp)
    return final

# === End PLA Parser & Helpers ===



# === QIN (Quinella) Parser & Helpers ===
def _norm_pair(a: int, b: int):
    return tuple(sorted((int(a), int(b))))

def parse_qin_collation_text(text: str, scale: float = 1.0):
    """Parse a QIN collation dump into a DataFrame with columns: Runner A, Runner B, Units.
    Also returns meta dict with total and num_runners (if present).
    We use the FIRST numeric token after a pair as its Units (repeated columns are ignored).
    """
    # Clean lines similar to WIN parser
    t_lines = [" ".join(line.translate(_TRANS_TABLE).split()) for line in text.splitlines()]
    t = "\n".join(t_lines)

    # TOTAL and NUM RUNNERS
    total = None
    total_patterns = [
        r"^\s*TOTAL\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*$",
        r"^\s*TOTAL\s+(?:SALES|POOL|TURNOVER)\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*$",
        r"^\s*GROSS\s+SALES\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*$",
        r"^\s*POOL\s+TOTAL\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*$",
    ]
    cands = []
    for pat in total_patterns:
        cands.extend(re.findall(pat, t, re.M))
    if cands:
        try: total = float(cands[-1].replace(',', ''))
        except Exception: total = None

    num_runners = None
    m = re.search(r"NUM RUNNERS\s+(\d+)", t)
    if m:
        try: num_runners = int(m.group(1))
        except Exception: pass

    # Per-line group detection: e.g. '1- 2  4026.9  26.9  26.9', possibly multiple groups per line
    HGROUP = re.compile(r"(?<!\S)(\d{1,2})\s*-\s*(\d{1,2})\s+((?:[\d,]*\.\d+|\d+)(?:\s+(?:[\d,]*\.\d+|\d+))*)")
    pairs = {}  # (a,b) -> units
    for ln in t_lines:
        for m in HGROUP.finditer(ln):
            a, b = int(m.group(1)), int(m.group(2))
            nums = re.findall(r"[\d,]*\.\d+|\d+", m.group(3))
            if not nums:
                continue
            try:
                units = float(nums[0].replace(',', '')) * scale
            except Exception:
                continue
            pairs[_norm_pair(a, b)] = units

    # To DataFrame
    import pandas as pd
    if pairs:
        data = [(f"Runner {a}", f"Runner {b}", float(u)) for (a,b), u in sorted(pairs.items())]
    else:
        data = []
    df = pd.DataFrame(data, columns=["Runner A", "Runner B", "Units"])
    return df, {"total": total, "num_runners": num_runners}

def qin_declared_dividend(net: float, pair_units: float, rules: "PoolRules") -> float:
    """Declared QIN dividend per $1 for a single winning pair."""
    if pair_units <= 0:
        return 0.0
    raw = net / pair_units
    div = _apply_breakage(raw, rules.break_step, rules.break_mode)
    div = max(div, rules.min_div)
    return _format_display(div, rules.display_dp)
# === End QIN Parser & Helpers ===


# === TRI (Trifecta) Parser & Helpers ===
def parse_tri_collation_text(txt: str, scale: float = 1.0):
    """
    Parse TRI collation to triples (First, Second, Third, Units).
    Robust to NBSPs and mixed dash characters; takes the first number after the triple as Units.
    Returns (DataFrame, meta) with optional TOTAL/NUM RUNNERS.
    """
    import re, pandas as pd
    t = (txt or "").replace("\r","").replace("\u00a0"," ").replace("–","-").replace("—","-").replace("−","-")
    lines = [ln for ln in t.split("\n") if ln.strip()]
    out = []
    rx = re.compile(r"(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\D+(-?\d+(?:,\d{3})*(?:\.\d+)?)")
    for ln in lines:
        m = rx.search(ln)
        if m:
            a,b,c = int(m.group(1)), int(m.group(2)), int(m.group(3))
            units = float(m.group(4).replace(",","")) * float(scale or 1.0)
            out.append((a,b,c,units))
        else:
            m2 = re.search(r"(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", ln)
            if m2:
                rest = ln[m2.end():]
                mnum = re.search(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)", rest)
                if mnum:
                    a,b,c = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
                    units = float(mnum.group(1).replace(",","")) * float(scale or 1.0)
                    out.append((a,b,c,units))
    df = (pd.DataFrame(out, columns=["First","Second","Third","Units"])
          if out else pd.DataFrame([], columns=["First","Second","Third","Units"]))
    meta = {}
    try:
        mt = re.search(r"TOTAL\s+([0-9,]+(?:\.\d+)?)", t)
        if mt: meta["total"] = float(mt.group(1).replace(",",""))
        mr = re.search(r"NUM\s+RUNNERS\s+(\d+)", t)
        if mr: meta["num_runners"] = int(mr.group(1))
    except Exception:
        pass
    return df, meta

def tri_declared_dividend(net: float, triple_units: float, rules) -> float:
    import math
    if triple_units is None or triple_units <= 0 or net <= 0:
        return 0.0
    raw = net / float(triple_units)
    step = getattr(rules,"break_step",0.10)
    mode = getattr(rules,"break_mode","down")
    scale = int(round(1.0/step)) if step>0 else 1
    if mode=="down":
        val = math.floor(raw*scale)/scale
    elif mode=="nearest":
        val = round(raw*scale)/scale
    elif mode=="up":
        val = math.ceil(raw*scale)/scale
    else:
        val = raw
    min_div = getattr(rules,"min_div",1.00)
    if val < min_div: val = min_div
    dp = int(getattr(rules,"display_dp",2))
    return round(val, dp)
# === End TRI Parser & Helpers ===





# -------------------------------
# Core calculation logic (embedded)
# -------------------------------

@dataclass
class PoolRules:
    commission: float = 0.14  # e.g., 0.14 = 14%
    min_div: float = 1.00
    break_step: float = 0.10  # e.g., 0.10 = 10c
    break_mode: str = "down"  # "down" | "nearest" | "up"
    max_dividends: int = 8
    display_dp: int = 2

def _apply_breakage(value: float, step: float, mode: str) -> float:
    # Guard non-finite inputs
    try:
        import math as _m
        if value is None or not _m.isfinite(float(value)):
            return 0.0
    except Exception:
        return 0.0
    if step <= 0:
        return float(value)
    scale = 1.0 / step
    if mode == "down":
        snapped = math.floor(float(value) * scale) / scale
    elif mode == "up":
        snapped = math.ceil(float(value) * scale) / scale
    else:
        snapped = round(float(value) * scale) / scale
    return snapped

def _format_display(value: float, dp: int) -> float:
    return float(f"{value:.{dp}f}")

def net_pool(gross_sales: float, refunds: float, jackpot_in: float, commission: float) -> float:
    base = max(gross_sales + jackpot_in - st.session_state.get('refunds', 0.0), 0.0)
    return base * (1.0 - max(commission, 0.0))

def approximates_from_spread(net: float, spread_units: Dict[str, float], rules: PoolRules, enforce_min_div: bool = True) -> Dict[str, float]:
    out = {}
    for runner, units in spread_units.items():
        if units <= 0:
            out[runner] = 0.0
            continue
        div = net / units
        div = _apply_breakage(div, rules.break_step, rules.break_mode)
        if enforce_min_div:
            div = max(div, rules.min_div)
        out[runner] = _format_display(div, rules.display_dp)
    return out

def approximates_from_spread_by_pool(net: float, spread_units: Dict[str, float], rules: PoolRules, pool: str, place_winners: int = 3, enforce_min_div: bool = True) -> Dict[str, float]:
    """Pool-aware approximates:
       - WIN (default): net / units
       - PLA: (net / place_winners) / units
       Other pools fall back to WIN-style.
    """
    eff_net = float(net)
    if str(pool).upper() == "PLA":
        pw = max(1, int(place_winners or 1))
        eff_net = net / pw if pw > 0 else 0.0
    out = {}
    for runner, units in spread_units.items():
        try:
            u = float(units)
        except Exception:
            u = 0.0
        if u <= 0 or not math.isfinite(u) or eff_net <= 0:
            out[runner] = 0.0
            continue
        div = eff_net / u
        div = _apply_breakage(div, rules.break_step, rules.break_mode)
        if enforce_min_div:
            div = max(div, rules.min_div)
        out[runner] = _format_display(div, rules.display_dp)
    return out


def dividends_from_spread(net: float, winners: List[str], spread_units: Dict[str, float], rules: PoolRules, declare_per_winner: bool = True) -> Dict[str, float]:
    # Cap number of dividends
    if len(winners) > rules.max_dividends:
        winners = winners[: rules.max_dividends]

    winning_units = 0.0
    for w in winners:
        winning_units += max(spread_units.get(w, 0.0), 0.0)

    if winning_units <= 0:
        return {w: 0.0 for w in winners}

    per_unit = net / winning_units
    per_unit = _apply_breakage(per_unit, rules.break_step, rules.break_mode)
    per_unit = max(per_unit, rules.min_div)

    if declare_per_winner:
        return {w: _format_display(per_unit, rules.display_dp) for w in winners}
    else:
        return {"ALL_WINNERS": _format_display(per_unit, rules.display_dp)}

def single_pool_dividend(net: float, winning_units: float, rules: PoolRules) -> float:
    if winning_units <= 0:
        return 0.0
    div = net / winning_units
    div = _apply_breakage(div, rules.break_step, rules.break_mode)
    div = max(div, rules.min_div)
    return _format_display(div, rules.display_dp)

# -------------------------------
# Preset rules (sourced from your spec; adjust if jurisdiction differs)
# -------------------------------

POOL_PRESETS = {
    # Single-leg
    "WIN":  {"commission": 0.1500, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 1,  "display_dp": 2},
    "PLA":  {"commission": 0.1475, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 1,  "display_dp": 2},
    "QIN":  {"commission": 0.1750, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 1,  "display_dp": 2},
    "EXA":  {"commission": 0.2000, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 1,  "display_dp": 2},
    "TRI":  {"commission": 0.2150, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 1,  "display_dp": 2},
    "FFR":  {"commission": 0.2300, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 1,  "display_dp": 2},
    "DUE":  {"commission": 0.1450, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 1,  "display_dp": 2},

    # Short multi-leg
    "RD":   {"commission": 0.2000, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 2,  "display_dp": 2},
    "DD":   {"commission": 0.2000, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 2,  "display_dp": 2},
    "TBL":  {"commission": 0.2500, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 3,  "display_dp": 2},

    # Quaddies
    "EQD":  {"commission": 0.2050, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 8,  "display_dp": 2},
    "QAD":  {"commission": 0.2050, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 8,  "display_dp": 2},

    # BIG6
    "BIG6": {"commission": 0.2500, "min_div": 1.00, "break_step": 0.10, "break_mode": "down", "max_dividends": 12, "display_dp": 2},
}


# Hard-coded commissions by jurisdiction (pool -> rate)
COMMISSION_BY_JURISDICTION = {
    "VIC & NSW": {
        "WIN": 0.1500, "PLA": 0.1475, "QIN": 0.1750, "EXA": 0.2000, "TRI": 0.2150,
        "DUE": 0.1450, "FFR": 0.2300, "RD": 0.2000, "DD": 0.2000, "TBL": 0.2500,
        "EQD": 0.2050, "QAD": 0.2050, "BIG6": 0.2500,
    },
    "QLD": {
        "WIN": 0.1500, "PLA": 0.1475, "QIN": 0.1750, "EXA": 0.2000, "TRI": 0.2150,
        "DUE": 0.1450, "FFR": 0.2300, "RD": 0.2000, "DD": 0.2000, "TBL": 0.2500,
        "EQD": 0.2050, "QAD": 0.2050, "BIG6": 0.2500,
    },
}



# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title='National Pooling Calculator', layout="wide")

st.title('National Pooling Calculator')
st.caption("Compute net pools, approximates (WillPays), and final dividends (dead-heats supported).")

# Sidebar: Pool + Preset rules
with st.sidebar:
    st.header("Pool & Rules")
    pool = st.selectbox("Pool", list(POOL_PRESETS.keys()), index=list(POOL_PRESETS.keys()).index("QAD"))
    preset = POOL_PRESETS[pool].copy()
    jurisdiction = st.selectbox("Jurisdiction", ["VIC & NSW", "QLD"], index=0, key="jurisdiction")
    try:
        preset["commission"] = COMMISSION_BY_JURISDICTION[jurisdiction][pool]
    except Exception:
        pass

    st.markdown("**Rule Overrides** (optional)")
    commission = st.number_input("Commission (0–1)", value=float(preset["commission"]), min_value=0.0, max_value=0.99, step=0.01, format="%.2f")
    min_div = st.number_input("Minimum Dividend ($)", value=float(preset["min_div"]), min_value=0.0, step=0.01, format="%.2f")
    break_step = st.number_input("Breakage Step ($)", value=float(preset["break_step"]), min_value=0.0, step=0.01, format="%.2f")
    break_mode = st.selectbox("Breakage Mode", ["down", "nearest", "up"], index=["down", "nearest", "up"].index(preset["break_mode"]))
    max_dividends = st.number_input("Max Dividends", value=int(preset["max_dividends"]), min_value=1, max_value=64, step=1)
    display_dp = st.number_input("Display Decimals", value=int(preset["display_dp"]), min_value=0, max_value=4, step=1)

    rules = PoolRules(
        commission=commission,
        min_div=min_div,
        break_step=break_step,
        break_mode=break_mode,
        max_dividends=max_dividends,
        display_dp=display_dp
    )

st.subheader("Last-leg Spread (multi-leg pools)")
st.caption("Enter runner → units (flexi-cent) across all valid combinations. Use the data editor below, upload CSV/JSON, or build via the Dead-heat Matrix.")

# Data editor initial state
default_df = pd.DataFrame({
    "Runner": ["Runner A", "Runner B", "Runner C", "Runner D"],
    "Units":  [1200.0,      950.0,      600.0,      400.0]
})

if "spread_df" not in st.session_state:
    st.session_state["spread_df"] = default_df.copy()

# Uploaders (optional)
up_col1, up_col2 = st.columns([1,1])
with up_col1:
    csv_file = st.file_uploader("Upload Spread (CSV with columns Runner,Units)", type=["csv"], accept_multiple_files=False)
    if csv_file is not None:
        df_csv = pd.read_csv(csv_file)
        if set(df_csv.columns) >= {"Runner", "Units"}:
            st.session_state["spread_df"] = df_csv[["Runner", "Units"]]
        else:
            st.error("CSV must include columns: Runner, Units")

with up_col2:
    json_file = st.file_uploader("Load Scenario (JSON)", type=["json"], accept_multiple_files=False)
    if json_file is not None:
        try:
            cfg = json.load(json_file)
            # rules
            r = cfg.get("rules", {})
            if r:
                st.info("Loaded rules from JSON. Apply via sidebar manually if needed.")
            # inputs
            ins = cfg.get("inputs", {})
            if "spread_units" in ins:
                rows = [{"Runner": k, "Units": float(v)} for k, v in ins["spread_units"].items()]
                st.session_state["spread_df"] = pd.DataFrame(rows)
            st.success("Scenario loaded (spread table updated).")
        except Exception as e:
            st.error(f"Failed to parse JSON: {e}")

# Editable spread table

# Show last parse result message after rerun
if st.session_state.get("just_parsed"):
    meta = st.session_state.get("last_parse_meta", {})
    st.success(f"Loaded {len(st.session_state.get('spread_df', []))} runners. "
               f"(NUM RUNNERS={meta.get('num_runners')}, TOTAL={meta.get('total')}, layout={meta.get('layout')})")
    st.session_state["just_parsed"] = False




# Make sure a spread table exists and render it
if "spread_df" not in st.session_state or not isinstance(st.session_state["spread_df"], pd.DataFrame):
    st.session_state["spread_df"] = pd.DataFrame(columns=["Runner", "Units"])

_spread_df = st.session_state["spread_df"]

st.markdown("#### Runner / Units (Editable)")
edited_spread = st.data_editor(
    _spread_df,
    width='stretch',
    hide_index=True,
    num_rows="dynamic",
    key="spread_editor_top",
    column_config={
        "Runner": st.column_config.TextColumn("Runner"),
        "Units": st.column_config.NumberColumn("Units", min_value=0.0, step=1.0, format="%.0f"),
    },
)
st.session_state["spread_df"] = edited_spread

# === Importer UI (with "Force horizontal" toggle) ===
if pool == 'WIN':
    with st.expander("📥 Import WIN Collation", expanded=False):
        up = st.file_uploader("Upload .txt/.log", type=["txt","log"],key="collation_upload_v6_1")
        default_text = ""
        if up is not None:
            default_text = up.read().decode("utf-8", errors="ignore")
    
        txt = st.text_area("…or paste a raw collation dump here", value=default_text, height=220,key="collation_area_v6_1")
    
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        with c1:
            field_choice = st.radio("Use this field as Units", options=[1,2,3,4], index=0, help="Used only in vertical mode.",key="field_choice_v8_1")
        with c2:
            scale = st.number_input("Scale factor", value=1.0, step=0.1,key="scale_input_v8_1")
        with c3:
            force_horizontal = st.checkbox("Force horizontal rows (1–4, 5–8, 9–11)", value=True, key="force_horizontal_v8_1")
        with c4:
            st.caption("If values import as 0, toggle force horizontal or try a different field.")
    
        if st.button("Parse & load to Runner/Units", key="btn_parse_collation_v6_1"):
            df_imp, meta = parse_collation_text(txt, field_index=field_choice, scale=scale, force_horizontal=force_horizontal)
            if df_imp.empty:
                st.error("No runner lines found. Check the paste/field choice.")
            else:
                # store parsed data + meta in session
                st.session_state["spread_df"] = df_imp.copy()
                if isinstance(meta, dict) and meta.get("total") is not None:
                    try:
                        st.session_state["gross_sales"] = float(meta.get("total"))
                    except Exception:
                        pass
                st.session_state["last_parse_meta"] = {
                    "num_runners": meta.get("num_runners") if isinstance(meta, dict) else None,
                    "total": meta.get("total") if isinstance(meta, dict) else None,
                    "layout": meta.get("layout") if isinstance(meta, dict) else None,
                }
                st.session_state["just_parsed"] = True
                import streamlit as _st
                _st.rerun()
                st.success(f"Loaded {len(df_imp)} runners. (NUM RUNNERS={meta.get('num_runners')}, TOTAL={meta.get('total')}, layout={meta.get('layout')})")
    # === End importer UI ===
    
    st.markdown('---')

if pool == 'QIN':
    st.subheader('Quinella (QIN)')
    st.caption('Import a QIN collation dump → pairs & units. TOTAL will prefill Gross Sales.')
    import pandas as pd

    with st.expander('📥 Import QIN Collation', expanded=False):
        up_qin = st.file_uploader('Upload .txt/.log', type=['txt','log'], key='qin_collation_upload_v1')
        default_qin_text = ''
        if up_qin is not None:
            default_qin_text = up_qin.read().decode('utf-8', errors='ignore')

        txt_qin = st.text_area('…or paste a QIN collation dump here', value=default_qin_text, height=220, key='qin_collation_area_v1')

        cqa1, cqa2 = st.columns([1,1])
        with cqa1:
            qin_scale = st.number_input('Scale factor', value=1.0, step=0.1, key='qin_scale_input_v1')
        with cqa2:
            st.caption('Parser uses first numeric field on each line as units.')

        if st.button('Parse & load QIN pairs', key='btn_parse_qin_v1'):
            try:
                df_qin, meta_qin = parse_qin_collation_text(txt_qin, scale=qin_scale)
            except Exception as e:
                st.exception(e)
                df_qin, meta_qin = pd.DataFrame([], columns=['Runner A','Runner B','Units']), {}
            if df_qin is None or df_qin.empty:
                st.error('No pairs found.')
            else:
                st.session_state['qin_df'] = df_qin.copy()
                if isinstance(meta_qin, dict) and meta_qin.get('total') is not None:
                    try:
                        st.session_state['gross_sales'] = float(meta_qin.get('total'))
                    except Exception:
                        pass
                st.success(f"Loaded {len(df_qin)} pairs. (TOTAL={meta_qin.get('total') if isinstance(meta_qin, dict) else None})")

    # Editable QIN pairs table
    qin_df = st.session_state.get('qin_df')
    if qin_df is None:
        qin_df = pd.DataFrame([], columns=['Runner A','Runner B','Units'])
    qin_df = st.data_editor(qin_df, num_rows='dynamic', width='stretch', key='qin_spread_editor',
                            column_config={
                                'Runner A': st.column_config.TextColumn('Runner A'),
                                'Runner B': st.column_config.TextColumn('Runner B'),
                                'Units': st.column_config.NumberColumn('Units', min_value=0.0, step=0.1, format='%.2f'),
                            })
    st.session_state['qin_df'] = qin_df

    # Winner selection + declared dividend
    all_nums = set()
    try:
        for col in ['Runner A','Runner B']:
            nums = qin_df[col].astype(str).str.extract(r'(\d+)')[0].dropna().astype(int).tolist()
            all_nums.update(nums)
    except Exception:
        pass

    if all_nums:
        ra, rb = st.columns([1,1])
        with ra:
            qin_a = st.number_input('First (winner)', value=min(all_nums), step=1, format='%d', key='qin_sel_first')
        with rb:
            default_second = sorted(all_nums)[1] if len(all_nums) > 1 else min(all_nums)
            qin_b = st.number_input('Second (runner-up)', value=default_second, step=1, format='%d', key='qin_sel_second')

        
        if st.button('Calculate QIN Declared Dividend', key='btn_qin_declared_div'):
                    # build pair dict (robustly extract digits from 'Runner 1' etc.)
                    import re as _re
                    pairs = {}
                    for _, row in qin_df.iterrows():
                        try:
                            ma = _re.search(r'(\d+)', str(row['Runner A']))
                            mb = _re.search(r'(\d+)', str(row['Runner B']))
                            if not ma or not mb:
                                continue
                            a = int(ma.group(1))
                            b = int(mb.group(1))
                            u = float(row['Units'])
                        except Exception:
                            continue
                        if a <= 0 or b <= 0 or a == b:
                            continue
                        pairs[_norm_pair(a, b)] = u

                    gross = float(st.session_state.get('gross_sales', 0.0))
                    refunds_val = float(st.session_state.get('refunds', 0.0))
                    jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                    net = net_pool(gross, refunds_val, jackpot_val, rules.commission)

                    pair_units = float(pairs.get(_norm_pair(qin_a, qin_b), 0.0))
                    qin_div = qin_declared_dividend(net, pair_units, rules)
                    st.success(f"Declared QIN dividend for Runner {min(qin_a, qin_b)} - Runner {max(qin_a, qin_b)}: ${qin_div}")
    else:
        st.info('Add some QIN pairs above to enable winner selection.')
    
    st.markdown('---')
if pool == 'PLA':
    st.markdown('---')
    st.subheader('Place (PLA)')
    st.caption('Import a PLA collation dump → runner units. TOTAL will prefill Gross Sales.')
    with st.expander('📥 Import PLA Collation', expanded=False):
        up_pla = st.file_uploader('Upload .txt/.log', type=['txt','log'], key='pla_collation_upload_v1')
        default_pla_text = ''
        if up_pla is not None:
            default_pla_text = up_pla.read().decode('utf-8', errors='ignore')
    
        txt_pla = st.text_area('…or paste a PLA collation dump here', value=default_pla_text, height=220, key='pla_collation_area_v1')
    
        cpa1, cpa2 = st.columns([1,1])
        with cpa1:
            pla_scale = st.number_input('Scale factor', value=1.0, step=0.1, key='pla_scale_input_v1')
        with cpa2:
            st.caption('PLA importer maps across columns horizontally from each starting runner (same style as WIN).')
    
        if st.button('Parse & load PLA runners', key='btn_parse_pla_v1'):
            df_pla, meta_pla = parse_pla_collation_text(txt_pla, scale=pla_scale)
            if df_pla.empty:
                st.error('No runner rows found. Check the paste.')
            else:
                st.session_state['pla_df'] = df_pla.copy()
                # Sync Approximates spread table BEFORE any rerun so UI refresh picks it up
                try:
                    _sdf = df_pla[['Runner','Units']].copy()
                    _sdf['Runner'] = _sdf['Runner'].astype(str)
                    _sdf = _sdf[_sdf['Runner'].str.strip() != '']
                    _sdf = _sdf[_sdf['Units'].astype(float).notna()]
                    st.session_state['spread_df'] = _sdf.reset_index(drop=True)
                except Exception:
                    pass
    
                try:
                    _tmp = st.session_state['pla_df']
                    _tmp['Runner'] = _tmp['Runner'].astype(str)
                    _tmp = _tmp[_tmp['Runner'].str.strip() != '']
                    _tmp = _tmp[_tmp['Units'].astype(float).notna()]
                    st.session_state['pla_df'] = _tmp
                except Exception:
                    pass
                try:
                    _pla_meta2 = parse_meta_from_collation(txt_pla)
                    st.session_state['pla_entries_at_deadline'] = _pla_meta2.get('entries_at_deadline')
                    if 'refunds' not in st.session_state or float(st.session_state.get('refunds', 0.0)) == 0.0:
                        st.session_state['refunds'] = float(_pla_meta2.get('scratchings_value', 0.0))
                except Exception:
                    pass
                # Prefill Gross Sales from PLA TOTAL
                try:
                    _pla_meta2 = parse_meta_from_collation(txt_pla)
                    st.session_state['pla_entries_at_deadline'] = _pla_meta2.get('entries_at_deadline')
                    if 'refunds' not in st.session_state or float(st.session_state.get('refunds', 0.0)) == 0.0:
                        st.session_state['refunds'] = float(_pla_meta2.get('scratchings_value', 0.0))
                except Exception:
                    pass
                
                try:
                    if isinstance(meta_pla, dict) and meta_pla.get('total') is not None:
                        st.session_state['gross_sales'] = float(meta_pla.get('total'))
                        import streamlit as _st
                        _st.rerun()
                except Exception:
                    pass
                            # (sync moved earlier)
                try:
                    _sdf = df_pla[['Runner','Units']].copy()
                    _sdf['Runner'] = _sdf['Runner'].astype(str)
                    _sdf = _sdf[_sdf['Runner'].str.strip() != '']
                    _sdf = _sdf[_sdf['Units'].astype(float).notna()]
                    st.session_state['spread_df'] = _sdf.reset_index(drop=True)
                except Exception:
                    pass
                st.success(f"Loaded {len(df_pla)} runners. (TOTAL={meta_pla.get('total')}, runners={meta_pla.get('num_runners')})")
    
    # Editable PLA table
    pla_df = st.session_state.get('pla_df')
    if pla_df is None:
        import pandas as pd
        pla_df = pd.DataFrame([], columns=['Runner','Units'])
    else:
        try:
            pla_df['Runner'] = pla_df['Runner'].astype(str)
            pla_df = pla_df[pla_df['Runner'].str.strip() != '']
            pla_df = pla_df[pla_df['Units'].astype(float).notna()]
        except Exception:
            pass
    pla_df = st.data_editor(pla_df, num_rows='fixed', width='stretch', key='pla_spread_editor')
    st.session_state['pla_df'] = pla_df
    
    # PLA winners selection and declared dividend calculation
    try:
        pla_runners = sorted({int(str(r).replace('Runner','').strip()) for r in pla_df['Runner'].dropna().str.replace('Runner','',regex=False).str.strip() if str(r).strip() != ''})
    except Exception:
        pla_runners = []
    
    c_pla1, c_pla2, c_pla3 = st.columns([1,1,1])
    with c_pla1:
        # Default place winners: 3 if field >= 8, else 2. Allow override.
        entries_at_deadline = st.session_state.get('pla_entries_at_deadline')
        default_places = 3 if (entries_at_deadline is not None and entries_at_deadline >= 8) else (3 if (len(pla_runners) >= 8) else 2)
        pla_winners_n = st.number_input('Place winners (2 or 3)', min_value=1, max_value=3, value=default_places, step=1, key='pla_winners_n')
    with c_pla2:
        # Winner A select
        if pla_runners:
            pla_sel = st.multiselect('Winning runner(s)', pla_runners, default=pla_runners[:min(pla_winners_n, len(pla_runners))], key='pla_winners_sel')
        else:
            pla_sel = []
    
    if st.button('Calculate Place Declared Dividend(s)', key='btn_pla_div_v1') and pla_sel:
        # Build dict from editor
        units_by_runner = {}
        for _, row in pla_df.iterrows():
            try:
                rn = int(str(row['Runner']).replace('Runner','').strip())
                u = float(row['Units'])
            except Exception:
                continue
            if rn > 0:
                units_by_runner[rn] = u
    
        gross_sales = float(st.session_state.get('gross_sales', 0.0))
        refunds = float(st.session_state.get('refunds', 0.0)) if 'refunds' in st.session_state else 0.0
        jackpot_in = float(st.session_state.get('jackpot_in', 0.0)) if 'jackpot_in' in st.session_state else 0.0
        gross = float(st.session_state.get('gross_sales', 0.0))
        refunds_val = float(st.session_state.get('refunds', 0.0))
        jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
        net = net_pool(gross, refunds_val, jackpot_val, rules.commission)
    
        pw = max(1, int(pla_winners_n))
        winners_units = {int(r): float(units_by_runner.get(int(r), 0.0)) for r in pla_sel}
        declared_map = pla_declared_dividends_with_deficiency(net, winners_units, pw, rules)
        for rn in pla_sel:
            st.success(f"Declared PLA dividend for Runner {rn}: ${declared_map.get(int(rn), 0.0)}")
    elif st.button('Calculate Place Declared Dividend(s)', key='btn_pla_div_v1_dummy', help='Disabled until you add winning runner(s).'):
        st.info('Add winning runner(s) above to calculate.')
    
    
    
    # Inputs
    colA, colB, colC, colD = st.columns([1,1,1,1])
    with colA:
        gross_sales = st.number_input("Gross Sales ($)", value=float(st.session_state.get("gross_sales", 100000.00)), min_value=0.0, step=100.0, format="%.2f", key="gross_sales")
    with colB:
        refunds = st.number_input("Refunds ($)", value=2000.00, min_value=0.0, step=10.0, format="%.2f", key="refunds")
    with colC:
        jackpot_in = st.number_input("Jackpot In ($)", value=5000.00, min_value=0.0, step=10.0, format="%.2f", key="jackpot_in")
    with colD:
        single_leg_units = st.number_input("Single-leg Winning Units (optional)", value=0.0, min_value=0.0, step=1.0, format="%.2f")
    
    st.markdown("---")
    
    st.session_state["spread_df"] = st.data_editor(
        st.session_state.get("spread_df"),
        num_rows="dynamic",
        width='stretch', key="spread_editor_main")

# --- Winner(s) picker for single-leg pools (e.g., WIN) ---
try:
    _opts = [str(x).strip() for x in st.session_state.get("spread_df", pd.DataFrame()).get("Runner", pd.Series([], dtype="object")).dropna().astype(str).unique().tolist()]
except Exception:
    _opts = []
if _opts:
    if pool == 'WIN':
            st.markdown("##### Winner(s)")
            _sel = st.multiselect("Select the winner(s)", _opts, default=st.session_state.get("winners", []), key="winners_picker")
            st.session_state["winners"] = _sel
    
            # session already updated
        
            # Winner selection
            runner_list = list(st.session_state.get("spread_df", pd.DataFrame())["Runner"].dropna().astype(str).unique())
            winners = st.multiselect("Winning runners (for declared dividends)", runner_list, default=runner_list[:3])
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button("Calculate Approximates / WillPays"):
                    gross = float(st.session_state.get('gross_sales', 0.0))
                    refunds_val = float(st.session_state.get('refunds', 0.0))
                    jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                    net = net_pool(gross, refunds_val, jackpot_val, rules.commission)
                    spread_units = {row["Runner"]: float(row["Units"]) for _, row in st.session_state["spread_df"].iterrows() if str(row["Runner"]).strip() != ""}
                    pw = int(st.session_state.get('pla_winners_n', 3))
                    approxs = approximates_from_spread_by_pool(net, spread_units, rules, pool=pool, place_winners=pw, enforce_min_div=True)
                    st.session_state["net"] = net
                    st.session_state["approxs"] = approxs
            with c2:
                if st.button("Calculate Declared Dividends"):
                    # Build net from session
                    gross = float(st.session_state.get('gross_sales', 0.0))
                    refunds_val = float(st.session_state.get('refunds', 0.0))
                    jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                    net = net_pool(gross, refunds_val, jackpot_val, rules.commission)
    
                    # Build spread from the editable table
                    _df = st.session_state.get("spread_df")
                    spread_units = {}
                    if _df is not None:
                        try:
                            for _, r in _df.iterrows():
                                k = str(r.get("Runner", "")).strip()
                                if k == "" or k.lower() == "runner":
                                    continue
                                v = float(r.get("Units", 0.0))
                                spread_units[k] = v
                        except Exception:
                            pass
    
                    # Winners may be defined only in some pool UIs; fall back to session or empty list
                    try:
                        _winners_tmp = winners
                    except NameError:
                        _winners_tmp = st.session_state.get("winners", []) or []
    
                    divs = dividends_from_spread(net, _winners_tmp, spread_units, rules, declare_per_winner=True)
                    if not _winners_tmp:
                        st.warning("Please select at least one winner above before calculating declared dividends.")
                    st.session_state["net"] = net
                    st.session_state["divs"] = divs
            with c3:
                if st.button("Calculate Single-leg Dividend"):
                    gross = float(st.session_state.get('gross_sales', 0.0))
                    refunds_val = float(st.session_state.get('refunds', 0.0))
                    jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                    net = net_pool(gross, refunds_val, jackpot_val, rules.commission)
                    st.session_state["net"] = net
                    st.session_state["single"] = single_pool_dividend(net, single_leg_units, rules)
    
            # Results
    
    if 'exa_approxs' in st.session_state:
        st.subheader('EXA Approximates / WillPays ($1)')
        df_exa_app = pd.DataFrame({'Pair': list(st.session_state['exa_approxs'].keys()),
                                   'Approx ($1)': list(st.session_state['exa_approxs'].values())})
        st.dataframe(df_exa_app, width='stretch', hide_index=True)
        st.download_button('Download EXA Approximates (CSV)',
                           df_exa_app.to_csv(index=False).encode('utf-8'),
                           'exa_approximates.csv','text/csv')

    st.markdown("### Results")
    net_disp = st.session_state.get("net", None)
    if net_disp is not None:
        st.info(f"Net Pool: ${net_disp:,.2f}  (Commission {rules.commission:.2%})")

    if "approxs" in st.session_state:
        st.subheader("Approximates / WillPays ($1)")
        df_a = pd.DataFrame({"Runner": list(st.session_state["approxs"].keys()), "Approx ($1)": list(st.session_state["approxs"].values())})
        st.dataframe(df_a, width='stretch', hide_index=True)
        st.download_button("Download Approximates (CSV)", df_a.to_csv(index=False).encode("utf-8"), "approximates.csv", "text/csv")

    if "divs" in st.session_state:
        st.subheader("Declared Dividends ($1)")
        df_d = pd.DataFrame({"Declared Dividend": list(st.session_state["divs"].keys()), "Amount ($1)": list(st.session_state["divs"].values())})
        st.dataframe(df_d, width='stretch', hide_index=True)
        st.download_button("Download Declared Dividends (CSV)", df_d.to_csv(index=False).encode("utf-8"), "declared_dividends.csv", "text/csv")

    if "single" in st.session_state:
        st.subheader("Single-leg Dividend ($1)")
        st.write(float(st.session_state["single"]))


# ===================== FFR — Transaction-based (scan sells) ONLY =====================
if pool == "FFR":
    with st.expander("🧾 Transaction-based FFR (scan sells)", expanded=False):
        import re as _re
        from dataclasses import dataclass
        from typing import Optional, List
        from decimal import Decimal, ROUND_FLOOR

        @dataclass
        class _FFRTicket:
            kind: str                  # STRAIGHT | BOX | ROVER | STANDOUT
            stake: float               # total $ on this ticket
            lines: int                 # number of combinations covered
            legs: dict                 # {'F':[...], 'S':[...], 'T':[...], 'Q':[...]
            rover: Optional[int] = None

        def _ffr_to_ints(s: str) -> list[int]:
            return [int(x) for x in _re.findall(r"\d+", s or "")]
        def _ffr_parse_transactions(text: str) -> dict:
            """Parse FFR transaction dump.
        
            - Reads POOL TOTALS / SUB TOTALS (SELLS, optional PAID SELL) for gross and payout percent
            - Parses STRAIGHT bets (A/B/C/D) and FIELD bets F(x-y)/F(x-y)/F(x-y)/F(x-y), even when wrapped
            - Amounts may include commas and need not be EOL
            - If totals exist, do **not** add ticket stakes to gross (avoid double count)
            - If no totals, set gross = sum(stakes)
            """
            import re as _re
            from dataclasses import dataclass
            from typing import List, Optional
        
            tickets: List[_FFRTicket] = []
            gross = 0.0
            refunds = 0.0
            jackpot = 0.0
            percent: Optional[float] = None
            got_total = False
        
            if not text:
                return dict(tickets=tickets, gross=gross, refunds=refunds, jackpot=jackpot, percent=percent)
        
            # ---- Totals ----
            m_pool = _re.search(r"(?is)POOL\s+TOTALS:\s.*?FFR\s*\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2}))", text)
            if m_pool:
                try:
                    gross = float(m_pool.group(1).replace(',', ''))
                    got_total = True
                except Exception:
                    pass
        
            m_sub = _re.search(r"(?is)SUB\s+TOTALS:\s.*?SELLS\s*\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2}))(?:.*?PAID\s+SELL\s*\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})))?", text)
            if m_sub:
                try:
                    sells = float(m_sub.group(1).replace(',', ''))
                    if gross <= 0:
                        gross = sells
                    if m_sub.group(2):
                        paid = float(m_sub.group(2).replace(',', ''))
                        if sells > 0:
                            percent = (paid / sells) * 100.0
                    got_total = True
                except Exception:
                    pass
        
            # ---- Ticket lines ----
            def _rng(lo: int, hi: int) -> list[int]:
                lo, hi = int(lo), int(hi)
                return list(range(min(lo, hi), max(lo, hi)+1))
        
            lines = (text or "").splitlines()
            i = 0
            while i < len(lines):
                raw = lines[i]
                up  = " ".join(raw.upper().split())
                if "FFR" not in up:
                    i += 1; continue
        
                # For wrapped field specs, join two lookahead lines
                joined = " ".join([raw, (lines[i+1] if i+1 < len(lines) else ""), (lines[i+2] if i+2 < len(lines) else "")])
        
                # pick the last amount on the first line
                m_amt = _re.findall(r"([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2}))", raw)
                amt = float(m_amt[-1].replace(',', '')) if m_amt else None
        
                # STRAIGHT
                m_st = _re.search(r"\b(\d+)[-/](\d+)[-/](\d+)[-/](\d+)\b", joined)
                if m_st and amt is not None:
                    a,b,c,d = [int(x) for x in m_st.groups()]
                    legs = dict(F=[a], S=[b], T=[c], Q=[d])
                    tickets.append(_FFRTicket("STRAIGHT", amt, 1, legs))
                    if not got_total: gross += amt
                    i += 1; continue
        
                # FIELD (allow whitespace after '(' )
                m_field = _re.search(r"F\(\s*(\d+)\s*-\s*(\d+)\)\s*/\s*F\(\s*(\d+)\s*-\s*(\d+)\)\s*/\s*F\(\s*(\d+)\s*-\s*(\d+)\)\s*/\s*F\(\s*(\d+)\s*-\s*(\d+)\)", joined)
                if m_field and amt is not None:
                    a1,b1,a2,b2,a3,b3,a4,b4 = [int(x) for x in m_field.groups()]
                    F = _rng(a1,b1); S = _rng(a2,b2); T = _rng(a3,b3); Q = _rng(a4,b4)
                    ahead = "\n".join(lines[i+1:i+6])
                    m_combs = _re.search(r"No\.\s*of\s*combs\s*=\s*(\d+)", ahead, flags=_re.I)
                    lines_count = int(m_combs.group(1)) if m_combs else 0
                    if lines_count <= 0:
                        n = len(F) if (len(F)==len(S)==len(T)==len(Q) and F==S==T==Q) else 0
                        lines_count = n*(n-1)*(n-2)*(n-3) if n >= 4 else 0
                    tickets.append(_FFRTicket("STANDOUT", amt, lines_count, dict(F=F,S=S,T=T,Q=Q)))
                    if not got_total: gross += amt
                    i += 1; continue
        
                i += 1
        
            # Percent override anywhere
            m_pct = _re.search(r"(?i)\bPERCENT\s+([0-9]+(?:\.[0-9]+)?)\b", text)
            if m_pct:
                try:
                    percent = float(m_pct.group(1))
                except Exception:
                    pass
        
            # If no totals were present, fall back to sum of stakes
            if (not got_total) and (not gross) and tickets:
                try:
                    gross = float(sum(getattr(t, 'stake', 0.0) for t in tickets))
                except Exception:
                    pass
        
            return dict(tickets=tickets, gross=gross, refunds=refunds, jackpot=jackpot, percent=percent)





        def _ffr_ticket_covers(ticket: _FFRTicket, order: tuple[int,int,int,int]) -> bool:
            a, b, c, d = order
            F, S, T, Q = ticket.legs["F"], ticket.legs["S"], ticket.legs["T"], ticket.legs["Q"]
            if ticket.kind == "STRAIGHT":
                return F == [a] and S == [b] and T == [c] and Q == [d]
            if a not in F or b not in S or c not in T or d not in Q:
                return False
            if len({a, b, c, d}) < 4:
                return False
            if ticket.kind == "ROVER" and ticket.rover is not None:
                return ticket.rover in (a, b, c, d)
            return True

        def _ffr_units_per1(order: tuple[int,int,int,int], tickets: List[_FFRTicket]) -> float:
            u = 0.0
            for t in tickets:
                if t.lines <= 0 or t.stake <= 0:
                    continue
                if _ffr_ticket_covers(t, order):
                    u += (t.stake / t.lines)
            return u

        def _ffr_declared_from_tx(order: tuple[int,int,int,int],
                                  tickets: List[_FFRTicket],
                                  commission: float,
                                  gross: float,
                                  refunds: float,
                                  jackpot: float,
                                  percent: Optional[float],
                                  break_step: float,
                                  display_dp: int,
                                  break_mode: str = "down") -> float:
            """
            Declared dividend per $1 stake using transaction-derived units.
            If `percent` is provided (e.g., 77.12), it overrides commission: net = base * (percent/100).
            """
            base = max((gross or 0.0) - (refunds or 0.0), 0.0) + max(jackpot or 0.0, 0.0)
            pct = (percent / 100.0) if percent is not None else (1.0 - float(commission))
            net = base * pct

            u = _ffr_units_per1(order, tickets)
            if u <= 0 or net <= 0:
                return 0.0

            step_c = int(round(break_step * 100)) or 10
            net_c = (Decimal(str(net)) * 100).to_integral_value(rounding=ROUND_FLOOR)
            per_unit_c = (net_c / Decimal(str(u))).to_integral_value(rounding=ROUND_FLOOR)

            if break_mode == "down":
                per_unit_c -= per_unit_c % step_c
            elif break_mode == "up":
                rem = per_unit_c % step_c
                if rem:
                    per_unit_c += (step_c - rem)
            else:  # nearest
                rem = per_unit_c % step_c
                if rem >= (step_c // 2):
                    per_unit_c += (step_c - rem)
                else:
                    per_unit_c -= rem

            return float((per_unit_c / Decimal(100)).quantize(Decimal(10) ** -int(display_dp)))

        # --- Inputs/controls ---
        up_tx = st.file_uploader("Upload transaction dump (.txt/.log)", type=["txt", "log"], key="ffr_tx_upload_v6")
        default_tx = ""
        if up_tx is not None:
            default_tx = up_tx.read().decode("utf-8", errors="ignore")
        txt_tx = st.text_area("…or paste transaction text here", value=default_tx, height=220, key="ffr_tx_area_v6")

        ca, cb, cc, cd = st.columns(4)
        with ca:
            pick_a = st.number_input("First", min_value=1, value=1, step=1, key="ffr_tx_a_v6")
        with cb:
            pick_b = st.number_input("Second", min_value=1, value=2, step=1, key="ffr_tx_b_v6")
        with cc:
            pick_c = st.number_input("Third", min_value=1, value=3, step=1, key="ffr_tx_c_v6")
        with cd:
            pick_d = st.number_input("Fourth", min_value=1, value=4, step=1, key="ffr_tx_d_v6")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            use_percent = st.checkbox("Use 'Percent' from file (if present)", value=True, key="ffr_tx_usepct_v6")
        with c2:
            comm = st.number_input("Commission (%)", min_value=0.0, max_value=100.0, value=23.00, step=0.05, format="%.2f", key="ffr_tx_comm_v6")
        with c3:
            break_step = st.selectbox("Breakage step", [0.10, 0.05, 0.01], index=0, key="ffr_tx_break_step_v6")
        with c4:
            dp = st.selectbox("Display DP", [2, 3], index=0, key="ffr_tx_dp_v6")

        if st.button("Compute FFR from transactions", key="btn_ffr_txn_compute_v6"):
            try:
                parsed = _ffr_parse_transactions(txt_tx or "")
                tickets = parsed["tickets"]
                gross = parsed["gross"]
                refunds = parsed["refunds"]
                jackpot = parsed["jackpot"]
                percent = parsed["percent"] if use_percent else None

                units = _ffr_units_per1((int(pick_a), int(pick_b), int(pick_c), int(pick_d)), tickets)
                payout_pool = ((percent/100.0)*gross + jackpot - refunds) if (percent is not None) else ((1.0-comm/100.0)*gross + jackpot - refunds)
                if payout_pool < 0: payout_pool = 0.0
                div = _ffr_declared_from_tx(
                    (int(pick_a), int(pick_b), int(pick_c), int(pick_d)),
                    tickets,
                    commission=comm/100.0,
                    gross=gross,
                    refunds=refunds,
                    jackpot=jackpot,
                    percent=percent,
                    break_step=float(break_step),
                    display_dp=int(dp),
                    break_mode="down",
                )

                pool_label = (f"Paid-sell {percent:.2f}%" if percent is not None else f"Commission {comm:.2f}%")
                msg = (
                    f"**${div:,.2f}** per $1  •  "
                    f"Units {units:,.5f}  •  "
                    f"Payout pool ${payout_pool:,.2f} ({pool_label})  •  "
                    f"Gross ${gross:,.2f}  •  Refunds ${refunds:,.2f}  •  Jackpot ${jackpot:,.2f}"
                )
                st.success(msg)

                # Optional: populate session for downstream displays
                st.session_state['ffr_model_price_pending'] = float(div)
                st.session_state['single'] = float(div)
            except Exception as e:
                st.exception(e)
    # =================== End transaction-based FFR ONLY =====================


    # Export scenario JSONst.markdown("---")# ===================== FFR — Transaction-based (scan sells) ONLY =====================
# =================== End transaction-based FFR ONLY ===================
st.subheader("Export / Save Scenario")
if st.button("Generate Scenario JSON"):
    try:
        # Build spread units from the visible spread table, if present
        spread_df = st.session_state.get("spread_df")
        if spread_df is not None:
            spread_units = {str(row["Runner"]).strip(): float(row["Units"]) for _, row in spread_df.iterrows() if str(row["Runner"]).strip() != ""}
        else:
            spread_units = {}
        # Winners may not exist in all pool views
        try:
            _winners_tmp = winners
        except NameError:
            _winners_tmp = []
        # Single leg optional
        single_leg_units_val = st.session_state.get("single_leg_units", 0.0)
        scenario = {
            "rules": asdict(rules),
            "inputs": {
                "gross_sales": float(st.session_state.get("gross_sales", 0.0)),
                "refunds": float(st.session_state.get("refunds", 0.0)),
                "jackpot_in": float(st.session_state.get("jackpot_in", 0.0)),
                "spread_units": spread_units,
                "winners": _winners_tmp,
                "single_leg_winning_units": single_leg_units_val if single_leg_units_val else None
            }
        }
        st.session_state['scenario_json'] = json.dumps(scenario, indent=2)
        st.markdown('---')
        st.success("Scenario JSON generated below.")
    except Exception as e:
        st.exception(e)
# -------------------------------
# Dead-heat Matrix Builder (optional)
# -------------------------------
st.caption("Define winners for each earlier leg and last-leg runners, then input units per combination. The app will aggregate to a last-leg spread for you.")

with st.expander("Open Dead-heat Matrix Builder"):
    col_n, col_warn = st.columns([1,3])
    with col_n:
        num_legs = st.number_input("Number of legs", value=4, min_value=2, max_value=6, step=1)
    with col_warn:
        st.write("For QAD/EQD use 4 legs, TBL 3, DD/RD 2, BIG6 6. "
                 "Large tie sets create many combinations; we cap at 5000 rows for performance.")

    # Text inputs for leg winners and last leg runners
    leg_inputs = []
    for i in range(1, int(num_legs)):
        leg_inputs.append(st.text_input(f"Leg {i} winners (comma-separated)", value="A,B" if i == 1 else ""))
    last_leg_runners = st.text_input(f"Leg {int(num_legs)} (last) runners (comma-separated)", value="A,B,C")

    # Build grid
    if st.button("Generate Combination Grid"):
        legs = []
        for txt in leg_inputs:
            vals = [x.strip() for x in txt.split(",") if x.strip()]
            legs.append(vals if vals else ["*"])
        last_vals = [x.strip() for x in last_leg_runners.split(",") if x.strip()]
        if not last_vals:
            st.error("Please enter at least one last-leg runner.")
        else:
            combos = list(itertools.product(*legs, last_vals))
            if len(combos) > 5000:
                st.error(f"Too many combinations ({len(combos)}). Reduce winners per leg.")
            else:
                columns = [f"L{i}" for i in range(1, int(num_legs))] + [f"L{int(num_legs)}"]
                df = pd.DataFrame(combos, columns=columns)
                df["Units"] = 0.0
                st.session_state["combo_df"] = df
                st.success(f"Generated {len(df)} combinations. Enter Units per combination below.")

    if "combo_df" in st.session_state:
        st.markdown("#### Combination Units Editor")
        combo_df = st.data_editor(st.session_state["combo_df"], num_rows="dynamic", width='stretch',key="combo_editor")
        st.session_state["combo_df"] = combo_df

        # Aggregate to spread
        if st.button("Compute Spread From Grid"):
            last_col = f"L{int(num_legs)}"
            g = combo_df.groupby(last_col)["Units"].sum().reset_index()
            g.columns = ["Runner", "Units"]
            st.session_state["spread_df"] = g
            st.success("Spread table updated from combination grid. Scroll up to see the updated spread.")

        # Downloads
        cdl, cdr = st.columns([1,1])
        with cdl:
            if st.session_state.get("combo_df") is not None:
                csv_data = st.session_state["combo_df"].to_csv(index=False).encode("utf-8")
                st.download_button("Download Combination Grid (CSV)", csv_data, "combination_grid.csv", "text/csv")
        with cdr:
            if st.session_state.get("spread_df") is not None:
                csv_data2 = st.session_state["spread_df"].to_csv(index=False).encode("utf-8")
                st.download_button("Download Spread (CSV)", csv_data2, "derived_spread.csv", "text/csv")

if "scenario_json" in st.session_state:
    st.code(st.session_state["scenario_json"], language="json")
    st.download_button("Download Scenario JSON", st.session_state["scenario_json"].encode("utf-8"), "scenario.json", "application/json")

st.markdown("---")
st.caption("Dead-heat Matrix lets you enter ties in earlier legs and assign units per combination; we then aggregate to a last-leg spread. "
           "Presets: commissions (aligned to QLD), MIN_DIV_BREAK=10c, min-div 1.04 (most pools) / 1.00 (BIG6), MAXDIV caps (EQD/QAD 8; BIG6 12).")








# === End improved parser ===




# === End clean parser ===




# === End parser ===



# =====================
# TRIFECTA (TRI) Section
# =====================
if pool == 'TRI':
    st.subheader('Trifecta (TRI)')
    st.caption('Import a TRI collation dump → triples & units. TOTAL will prefill Gross Sales.')
    import pandas as pd, re as _re

    with st.expander('📥 Import TRI Collation', expanded=False):
        up_tri = st.file_uploader('Upload .txt/.log', type=['txt','log'], key='tri_collation_upload_v1')
        default_tri_text = ''
        if up_tri is not None:
            default_tri_text = up_tri.read().decode('utf-8', errors='ignore')
        txt_tri = st.text_area('…or paste a TRI collation dump here', value=default_tri_text, height=220, key='tri_collation_area_v1')

        cta1, cta2 = st.columns([1,1])
        with cta1:
            tri_scale = st.number_input('Scale factor', value=1.0, step=0.1, key='tri_scale_input_v1')
        with cta2:
            st.caption('Parser uses the first numeric field after each triple as Units.')

        if st.button('Parse & load TRI triples', key='btn_parse_tri_v1'):
            try:
                df_tri, meta_tri = parse_tri_collation_text(txt_tri, scale=tri_scale)
            except Exception as e:
                st.exception(e)
                df_tri, meta_tri = pd.DataFrame([], columns=['First','Second','Third','Units']), {}
            if df_tri is None or df_tri.empty:
                st.error('No triples found.')
            else:
                st.session_state['tri_df'] = df_tri.copy()
                if isinstance(meta_tri, dict) and meta_tri.get('total') is not None:
                    try:
                        st.session_state['gross_sales'] = float(meta_tri.get('total'))
                    except Exception:
                        pass
                st.success(f"Loaded {len(df_tri)} triples. (TOTAL={meta_tri.get('total') if isinstance(meta_tri, dict) else None})")

    tri_df = st.session_state.get('tri_df')
    if tri_df is None:
        tri_df = pd.DataFrame([], columns=['First','Second','Third','Units'])
    tri_df = st.data_editor(
        tri_df,
        num_rows='dynamic',
        width='stretch',
        key='tri_spread_editor',
        column_config={
            'First': st.column_config.NumberColumn('First', min_value=0, step=1, format='%d'),
            'Second': st.column_config.NumberColumn('Second', min_value=0, step=1, format='%d'),
            'Third': st.column_config.NumberColumn('Third', min_value=0, step=1, format='%d'),
            'Units': st.column_config.NumberColumn('Units', min_value=0.0, step=0.1, format='%.4f'),
        }
    )
    st.session_state['tri_df'] = tri_df

    # Winners selection
    nums = set()
    try:
        for col in ['First','Second','Third']:
            lst = tri_df[col].astype(str).str.extract(r'(\d+)')[0].dropna().astype(int).tolist()
            nums.update(lst)
    except Exception:
        pass

    if nums:
        cfa, cfb, cfc = st.columns([1,1,1])
        with cfa:
            tri_first = st.number_input('First (winner)', value=min(nums), step=1, format='%d', key='tri_sel_first')
        with cfb:
            tri_second = st.number_input('Second', value=min(nums), step=1, format='%d', key='tri_sel_second')
        with cfc:
            tri_third = st.number_input('Third', value=min(nums), step=1, format='%d', key='tri_sel_third')


        
# Approximates / WillPays for TRI
        if st.button('Calculate TRI Approximates / WillPays', key='btn_tri_approxs'):
            try:
                import pandas as pd
                from collections import defaultdict
                gross = float(st.session_state.get('gross_sales', 0.0))
                refunds_val = float(st.session_state.get('refunds', 0.0))
                jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                net = net_pool(gross, refunds_val, jackpot_val, rules.commission)
                st.session_state['net'] = net

                units_map = defaultdict(float)
                for _, r in tri_df.iterrows():
                    try:
                        a = int(_re.search(r'(\d+)', str(r['First'])).group(1))
                        b = int(_re.search(r'(\d+)', str(r['Second'])).group(1))
                        c = int(_re.search(r'(\d+)', str(r['Third'])).group(1))
                        u = float(r['Units'])
                    except Exception:
                        continue
                    if a <= 0 or b <= 0 or c <= 0 or len({a, b, c}) < 3 or u <= 0:
                        continue
                    units_map[(a, b, c)] += u

                approxs_tri = {f"{a}-{b}-{c}": tri_approximate_dividend(net, u, rules)
                               for (a, b, c), u in units_map.items()}

                st.session_state['tri_approxs'] = approxs_tri

                # Display results
                if approxs_tri:
                    df_tri_app = pd.DataFrame({'Triple': list(approxs_tri.keys()),
                                               'Approx ($1)': [round(v, rules.display_dp) for v in approxs_tri.values()]})
                    st.subheader('TRI Approximates / WillPays ($1)')
                    st.dataframe(df_tri_app, width='stretch', hide_index=True)
                    st.download_button('Download TRI Approximates (CSV)',
                                       df_tri_app.to_csv(index=False).encode('utf-8'),
                                       'tri_approximates.csv','text/csv', key='dl_tri_approxs')
                    st.success(f"Calculated TRI approximates for {len(approxs_tri)} triples.")
                else:
                    st.info('No valid TRI triples found to calculate approximates.')
            except Exception as e:
                st.exception(e)


        # Declared dividend calc
        if st.button('Calculate TRI Declared Dividend', key='btn_tri_declared_div'):
            triples = {}
            for _, r in tri_df.iterrows():
                try:
                    a = int(_re.search(r'(\d+)', str(r['First'])).group(1))
                    b = int(_re.search(r'(\d+)', str(r['Second'])).group(1))
                    c = int(_re.search(r'(\d+)', str(r['Third'])).group(1))
                    u = float(r['Units'])
                except Exception:
                    continue
                if a<=0 or b<=0 or c<=0 or len({a,b,c})<3:
                    continue
                triples[(a,b,c)] = u
            gross = float(st.session_state.get('gross_sales', 0.0))
            refunds_val = float(st.session_state.get('refunds', 0.0))
            jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
            net = net_pool(gross, refunds_val, jackpot_val, rules.commission)
            combo = (int(tri_first), int(tri_second), int(tri_third))
            triple_units = float(triples.get(combo, 0.0))
            tri_div = tri_declared_dividend(net, triple_units, rules)
            st.success(f"Declared TRI dividend for {combo[0]}-{combo[1]}-{combo[2]}: ${tri_div}")
    else:
        st.info('Add some TRI triples above to enable winner selection.')
    st.markdown('---')

# ======================= EXA Helpers & UI (appended) =======================

def parse_exa_collation_text(txt: str, scale: float = 1.0):
    """
    Parse an EXA collation dump into pairs (First, Second, Units).

    Robust to:
    - NBSPs and mixed whitespace
    - en/em/minus dashes (– — −) vs normal hyphen (-)
    - extra columns; we grab the FIRST numeric after the pair as Units
    """
    import re, pandas as pd

    t = (txt or "")
    # Normalise weird whitespace/dashes
    t = (t.replace("\r", "")
           .replace("\u00a0", " ")
           .replace("–", "-")
           .replace("—", "-")
           .replace("−", "-"))

    lines = [ln for ln in t.split("\n") if ln.strip()]
    out = []

    # Primary: pair anywhere in the line, then the first numeric after it
    rx = re.compile(
        r"(\d+)\s*-\s*(\d+)"                # 1- 2 (with flexible spaces)
        r"\D+?"                                # non-digits between pair and first number
        r"(-?\d+(?:,\d{3})*(?:\.\d+)?)"     # first numeric => units (e.g. 22.777777 or 1,234.5)
    )

    for ln in lines:
        m = rx.search(ln)
        if not m:
            # Fallback: very loose—find pair, then scan rest for first number
            m2 = re.search(r"(\d+)\s*-\s*(\d+)", ln)
            if m2:
                rest = ln[m2.end():]
                mnum = re.search(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)", rest)
                if mnum:
                    a = int(m2.group(1)); b = int(m2.group(2))
                    units = float(mnum.group(1).replace(",", "")) * float(scale or 1.0)
                    out.append((a, b, units))
            continue

    # Extract for primary match
        a = int(m.group(1)); b = int(m.group(2))
        units = float(m.group(3).replace(",", "")) * float(scale or 1.0)
        out.append((a, b, units))

    df = pd.DataFrame(out, columns=["First", "Second", "Units"]) if out else pd.DataFrame([], columns=["First", "Second", "Units"])

    # Meta (optional): allow TOTAL prefill
    meta = {}
    try:
        mt = re.search(r"TOTAL\s+([0-9,]+(?:\.\d+)?)", t)
        if mt:
            meta["total"] = float(mt.group(1).replace(",", ""))
    except Exception:
        pass

    return df, meta

def due_declared_dividend(net: float, pair_units: float, rules: 'PoolRules') -> float:
    if pair_units <= 0:
        return 0.0
    raw = float(net) / float(pair_units)
    div = _apply_breakage(raw, rules.break_step, rules.break_mode)
    div = max(div, rules.min_div)
    return _format_display(div, rules.display_dp)

# UI
try:
    if pool == 'EXA':
        st.subheader('Exacta (EXA)')
        with st.expander('📥 Import EXA Collation', expanded=False):
            up_exa_file = st.file_uploader('Upload .txt/.log', type=['txt','log'], key='exa_file_uploader_v1')
            _exa_text_from_file = ''
            if up_exa_file is not None:
                try:
                    _exa_text_from_file = (up_exa_file.read() or b'').decode('utf-8', errors='ignore')
                except Exception:
                    pass
            txt_exa = st.text_area('Paste EXA collation dump', value=_exa_text_from_file, height=220, key='txt_exa')
            colx1, colx2 = st.columns([1,3])
            with colx1:
                exa_scale = st.number_input('Units scale', value=1.0, min_value=0.0, step=0.1, key='exa_scale')
            with colx2:
                if st.button('Parse & load EXA pairs', key='btn_parse_exa'):
                    df_exa, meta_exa = parse_exa_collation_text(txt_exa, scale=exa_scale)
                    if df_exa is not None and len(df_exa) > 0:
                        st.session_state['exa_df'] = df_exa.copy()
                        try:
                            total = float(meta_exa.get('total') or 0.0)
                            if total > 0:
                                st.session_state['gross_sales'] = total
                        except Exception:
                            pass
                        st.success(f"Loaded {len(df_exa)} pairs. (TOTAL={meta_exa.get('total')}, runners={meta_exa.get('num_runners')})")
                    else:
                        st.warning('No pairs parsed from the text.')

        exa_df = st.session_state.get('exa_df')
        if exa_df is None:
            import pandas as pd
            exa_df = pd.DataFrame([], columns=['First','Second','Units'])

        try:
            exa_df = exa_df.dropna(subset=['First','Second'])
            exa_df['Units'] = exa_df['Units'].astype(float)
        except Exception:
            pass

        exa_df = st.data_editor(exa_df, num_rows='dynamic', key='exa_df_editor', width='stretch')

        cols = st.columns(2)
        with cols[0]:
            exa_first = st.number_input('First (winner)', min_value=1, step=1, value=1, key='exa_first')
        with cols[1]:
            exa_second = st.number_input('Second (runner-up)', min_value=1, step=1, value=2, key='exa_second')

        # Approximates / WillPays for EXA
        if st.button('Calculate EXA Approximates / WillPays', key='btn_exa_approxs'):
            try:
                import pandas as pd
                gross = float(st.session_state.get('gross_sales', 0.0))
                refunds_val = float(st.session_state.get('refunds', 0.0))
                jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                net = net_pool(gross, refunds_val, jackpot_val, rules.commission)
                st.session_state['net'] = net
                approxs_exa = {}
                for _, r in exa_df.iterrows():
                    pair = f"{int(r['First'])}-{int(r['Second'])}"
                    units = float(r['Units'])
                    div = due_declared_dividend(net, units, rules)
                    approxs_exa[pair] = div
                st.session_state['exa_approxs'] = approxs_exa
                # Show results immediately below the button
                st.subheader('EXA Approximates / WillPays ($1)')
                df_exa_app = pd.DataFrame({'Pair': list(approxs_exa.keys()),
                                           'Approx ($1)': [round(v, rules.display_dp) for v in approxs_exa.values()]})
                st.dataframe(df_exa_app, width='stretch', hide_index=True)
                st.download_button('Download EXA Approximates (CSV)',
                                   df_exa_app.to_csv(index=False).encode('utf-8'),
                                   'exa_approximates.csv','text/csv')
                st.success(f"Calculated EXA approximates for {len(approxs_exa)} pairs.")
            except Exception as e:
                st.exception(e)


        if st.button('Calculate EXA Declared Dividend', key='btn_exa_calc'):
            try:
                gross = float(st.session_state.get('gross_sales', 0.0))
                refunds_val = float(st.session_state.get('refunds', 0.0))
                jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                net = net_pool(gross, refunds_val, jackpot_val, rules.commission)
                sel = exa_df[(exa_df['First'].astype(int) == int(exa_first)) & (exa_df['Second'].astype(int) == int(exa_second))]
                pair_units = float(sel['Units'].iloc[0]) if len(sel) else 0.0
                div = due_declared_dividend(net, pair_units, rules)
                st.info(f'EXA {int(exa_first)}-{int(exa_second)} units={pair_units:.4f}  →  Declared: ${div}')
            except Exception as e:
                st.exception(e)
except Exception as _exa_err:
    st.error(f'EXA UI error: {_exa_err}')

# =================== End EXA Helpers & UI ===================


# =================== FFR Helpers (top-loaded) ===================
import re as _re_ffr
from decimal import Decimal as _D, ROUND_FLOOR as _RF


def parse_ffr_collation(text: str):
    """
    Robust FFR parser (patched):
    - Normalizes box-drawing glyphs (│ ─ ├ ┤ ┌ ┐ └ ┘ etc.) to spaces
    - Accepts '...' / '…' fillers and numbers with commas
    - Uses the FIRST four numeric tokens after the runner as First/Second/Third/Fourth units
      (ignores trailing tokens like duplicated end-of-line figures)
    - Ignores the later 'Percent' table entirely
    Returns: (pool_total, first, second, third, fourth) with Decimal values.
    """
    if not text:
        return (_D(0), {}, {}, {}, {})
    # Normalize glyphs & whitespace
    glyphs = "│┼┤├┬┴┌┐└┘─━—│┃╭╮╯╰▕▏|"
    trans = str.maketrans({c: " " for c in glyphs})
    t = "\n".join(" ".join(line.translate(trans).split()) for line in text.replace('…','...').splitlines())

    # Pool Total
    m_total = _re_ffr.search(r"Pool\s*Total\s*([0-9][0-9,]*(?:\.\d+)?)", t, flags=_re_ffr.I)
    pool_total = _D(m_total.group(1).replace(",","")) if m_total else _D(0)

    # Only consider the section before any "Percent" table
    head = t.split("Percent")[0]

    first, second, third, fourth = {}, {}, {}, {}

    for line in head.splitlines():
        # runner number at start
        m = _re_ffr.match(r'^\s*(\d{1,3})\b', line)
        if not m:
            continue
        runner = int(m.group(1))
        rest = line[m.end():]
        # keep only the segment before any '...' filler
        rest = _re_ffr.split(r'\.\.\.|…', rest)[0]
        # split by multiple spaces and/or pipes, then keep the first 4 numeric tokens
        tokens = [tok for tok in _re_ffr.split(r'[|]+|\s{2,}', rest) if tok and tok.strip()]
        nums = []
        for tok in tokens:
            mnum = _re_ffr.search(r'([0-9][0-9,]*(?:\.\d+)?)', tok)
            if mnum:
                nums.append(mnum.group(1).replace(',', ''))
            if len(nums) >= 4:
                break
        if len(nums) >= 4:
            f1, s2, t3, q4 = map(_D, nums[:4])
            first[runner]  = f1
            second[runner] = s2
            third[runner]  = t3
            fourth[runner] = q4

    if pool_total == 0 and first:
        pool_total = sum(first.values())

    return (pool_total, first, second, third, fourth)

def ffr_units_from_spread(first, second, third, fourth, total: _D, a:int, b:int, c:int, d:int) -> _D:
    """
    Units ≈ Total * (F[a]/sumF) * (S[b]/sumS) * (T[c]/sumT) * (Q[d]/sumQ)
    Robust even if each column's sum ≠ total. Requires all four runners distinct.
    """
    if any(x is None for x in (a,b,c,d)) or len({a,b,c,d}) < 4:
        return _D(0)
    F = _D(first.get(a, 0)); S = _D(second.get(b, 0))
    T = _D(third.get(c, 0)); Q = _D(fourth.get(d, 0))
    sumF = _D(sum(first.values())) if first else _D(0)
    sumS = _D(sum(second.values())) if second else _D(0)
    sumT = _D(sum(third.values())) if third else _D(0)
    sumQ = _D(sum(fourth.values())) if fourth else _D(0)

    if any(v <= 0 for v in (F,S,T,Q,total,sumF,sumS,sumT,sumQ)):
        return _D(0)

    # Independence-style expected units, scaled to pool total
    u = _D(total) * (F/sumF) * (S/sumS) * (T/sumT) * (Q/sumQ)
    return u

def ffr_approximate_dividend(net: float, quad_units: float, rules) -> float:
    # Use existing Decimal-based approx path (no min-div floor)
    return tri_approximate_dividend(net, quad_units * 10.0, rules)

def ffr_declared_dividend(net: float, quad_units: float, rules) -> float:
    # Use existing cents-first declared path (breakage down + min_div)
    return tri_declared_dividend(net, quad_units * 10.0, rules)

def _fallback_spreads_from_session(first_sp, second_sp, third_sp, fourth_sp, pool_total):
    """If parsed spreads are empty, pull from FFR data-only parser maps in session_state."""
    try:
        has = all(isinstance(x, dict) and sum(map(float, x.values())) > 0 for x in (first_sp, second_sp, third_sp, fourth_sp))
        if has:
            return first_sp, second_sp, third_sp, fourth_sp, pool_total
    except Exception:
        pass
    try:
        ss = st.session_state
        F = dict({int(k): float(v) for k, v in (ss.get("ffr_Fmap") or {}).items() if float(v) > 0})
        S = dict({int(k): float(v) for k, v in (ss.get("ffr_Smap") or {}).items() if float(v) > 0})
        T = dict({int(k): float(v) for k, v in (ss.get("ffr_Tmap") or {}).items() if float(v) > 0})
        Q = dict({int(k): float(v) for k, v in (ss.get("ffr_Qmap") or {}).items() if float(v) > 0})
        total = float(ss.get("ffr_pool_total") or ss.get("gross_sales") or pool_total or 0.0)
        if sum(F.values())>0 and sum(S.values())>0 and sum(T.values())>0 and sum(Q.values())>0:
            return F, S, T, Q, total
    except Exception:
        pass
    return first_sp, second_sp, third_sp, fourth_sp, pool_total
# ================= End FFR Helpers (top-loaded) =================



def parse_ffr_collation(text: str):
    """
    Robust FFR parser:
    - Accepts artefacts like '│' and '...' column fillers
    - Accepts numbers with thousands separators
    - Reads the first table (units by position) and 'Pool Total'
    """
    if not text:
        return (_D(0), {}, {}, {}, {})
    # Normalise weird characters
    t = text.replace('│',' ').replace('…','...')
    # Pool Total
    m_total = _re_ffr.search(r"Pool\s*Total\s*([0-9][0-9,]*(?:\.\d+)?)", t, flags=_re_ffr.I)
    pool_total = _D(m_total.group(1).replace(',','')) if m_total else _D(0)
    # Cut before "Percent" table if present
    head = t.split("Percent")[0]

    first, second, third, fourth = {}, {}, {}, {}
    # Row pattern: line that starts with (runner) then contains >=4 numeric fields anywhere
    for line in head.splitlines():
        m = _re_ffr.match(r'^\s*(\d{1,3})\b', line)
        if not m:
            continue
        runner = int(m.group(1))
        # capture ALL numeric tokens after the runner
        rest = line[m.end():]
        nums = [_D(x.replace(',','')) for x in _re_ffr.findall(r'(?<!\d)(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+)', rest)]
        if len(nums) >= 4:
            # use the LAST four tokens; the line often has '... 101.99' at the end
            f1, s2, t3, q4 = nums[-4:]
            first[runner]  = f1
            second[runner] = s2
            third[runner]  = t3
            fourth[runner] = q4

    # If Pool Total missing, try sum of first column
    if pool_total == 0 and first:
        pool_total = sum(first.values())

    return (pool_total, first, second, third, fourth)
# ================ End FFR Helpers ================


# ========================= DUE (Duet) — using EXA collation format =========================
def due_declared_dividend(net: float, pair_units: float, rules: 'PoolRules') -> float:
    # Duet: split net across 3 winning pairs, then apply breakage/min like EXA
    if pair_units <= 0:
        return 0.0
    per_pair_net = float(net) / 3.0
    raw = per_pair_net / float(pair_units)
    div = _apply_breakage(raw, rules.break_step, rules.break_mode)
    div = max(div, rules.min_div)
    return _format_display(div, rules.display_dp)

# ---------- DUE parser (moved above UI) ----------

# ---------- DUE parser (matrix-style: "A- B0  v1 v2 v3 ...") ----------
def due_parse_exa_collation_text(text: str, scale: float = 1.0):
    import re as _re
    import pandas as pd
    txt = text or ""

    def _f(m):
        try: return float(m.group(1).replace(",", ""))
        except: return None

    m_total = _re.search(r"(?im)^\s*TOTAL\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)", txt)
    m_jp    = _re.search(r"(?im)^\s*JACKPOT\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)", txt)
    m_nr    = _re.search(r"(?im)^\s*(?:NUM\s+RUNNERS|RUNNERS)\s+(\d+)", txt)
    total   = _f(m_total) if m_total else None
    jackpot = _f(m_jp) if m_jp else None
    num_runners = int(m_nr.group(1)) if m_nr else None

    rows = []
    line_re = _re.compile(r"(?m)^\s*(\d+)\s*-\s*(\d+)\s+(.+)$")
    float_re = _re.compile(r"[-+]?\d+(?:\.\d+)?")

    for a, b0, tail in line_re.findall(txt):
        a = int(a); b0 = int(b0)
        vals = [float(v) for v in float_re.findall(tail)]
        if not vals:
            continue
        for k, v in enumerate(vals):
            b = b0 + k
            if num_runners is not None and b > num_runners:
                break
            rows.append({"First": a, "Second": b, "Units": float(v) * float(scale or 1.0)})

    df = pd.DataFrame(rows, columns=["First", "Second", "Units"])
    meta = {"total": total, "jackpot": jackpot, "num_runners": num_runners, "layout": "pairs-horizontal"}
    return df, meta
# ---------- end DUE parser ----------

if pool == 'DUE':
    st.subheader('Duet (DUE)')
    with st.expander('📥 Import DUE Collation (EXA format)', expanded=False):
        up_due_file = st.file_uploader('Upload .txt/.log', type=['txt','log'], key='due_file_uploader_v1')
        _due_text_from_file = ''
        if up_due_file is not None:
            try:
                _due_text_from_file = (up_due_file.read() or b'').decode('utf-8', errors='ignore')
            except Exception:
                pass
        txt_due = st.text_area('Paste DUE collation dump (EXA-style pairs)', value=_due_text_from_file, height=220, key='txt_due')
        c1, c2 = st.columns([1,3])
        with c1:
            due_scale = st.number_input('Units scale', value=1.0, min_value=0.0, step=0.1, key='due_scale')
        with c2:
            if st.button('Parse & load DUE pairs', key='btn_parse_due'):
                df_due, meta_due = due_parse_exa_collation_text(txt_due or _due_text_from_file, scale=due_scale)
                if df_due is not None and len(df_due) > 0:
                    st.session_state['due_df'] = df_due.copy()
                    try:
                        total = float(meta_due.get('total') or 0.0)
                        if total > 0:
                            st.session_state['gross_sales'] = total
                    except Exception:
                        pass
                    st.success(f"Loaded {len(df_due)} pairs. (TOTAL={meta_due.get('total')}, runners={meta_due.get('num_runners')})")
                else:
                    st.warning('No pairs parsed from the text.')
    # Editor
    due_df = st.session_state.get('due_df')
    if due_df is None:
        import pandas as pd
        due_df = pd.DataFrame([], columns=['First','Second','Units'])
    try:
        due_df = due_df.dropna(subset=['First','Second'])
        due_df['Units'] = due_df['Units'].astype(float)
    except Exception:
        pass
    due_df = st.data_editor(due_df, num_rows='dynamic', key='due_df_editor', width='stretch')

    # Pick pair (order agnostic)
    cA, cB = st.columns(2)
    with cA:
        due_a = st.number_input('Runner A', min_value=1, step=1, value=1, key='due_a')
    with cB:
        due_b = st.number_input('Runner B', min_value=1, step=1, value=2, key='due_b')

    # Approximates
    if st.button('Calculate DUE Approximates', key='btn_due_approx'):
        try:
            import pandas as pd
            gross = float(st.session_state.get('gross_sales', 0.0))
            refunds_val = float(st.session_state.get('refunds', 0.0))
            jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
            net = net_pool(gross, refunds_val, jackpot_val, rules.commission)
            st.session_state['net'] = net
            approxs_due = {}
            for _, r in due_df.iterrows():
                pair = f"{int(r['First'])}-{int(r['Second'])}"
                units = float(r['Units'])
                div = due_declared_dividend(net, units, rules)
                approxs_due[pair] = float(div)
            # Present as table
            df_due_app = pd.DataFrame([{'Pair': k, 'Approx per $1': v} for k,v in approxs_due.items()])
            st.dataframe(df_due_app, width='stretch', hide_index=True)
            st.download_button('Download DUE Approximates (CSV)',
                               df_due_app.to_csv(index=False).encode('utf-8'),
                               'due_approximates.csv','text/csv')
            st.success(f"Calculated DUE approximates for {len(approxs_due)} pairs.")
        except Exception as e:
            st.exception(e)

    if st.button('Calculate DUE Declared Dividend', key='btn_due_calc'):
        try:
            gross = float(st.session_state.get('gross_sales', 0.0))
            refunds_val = float(st.session_state.get('refunds', 0.0))
            jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
            net = net_pool(gross, refunds_val, jackpot_val, rules.commission)

            a = int(due_a); b = int(due_b)
            # search for both orientations and sum if both present
            sel_ab = due_df[(due_df['First'].astype(int)==a) & (due_df['Second'].astype(int)==b)]
            sel_ba = due_df[(due_df['First'].astype(int)==b) & (due_df['Second'].astype(int)==a)]
            units = 0.0
            if len(sel_ab): units += float(sel_ab['Units'].iloc[0])
            if len(sel_ba): units += float(sel_ba['Units'].iloc[0])

            if units <= 0:
                st.warning(f'No units found for pair {a}-{b}.')
            else:
                div = due_declared_dividend(net, units, rules)
                pool_label = f'Commission {rules.commission*100:.2f}%'
                st.success(f"**${{div:,.2f}}** per $1  •  Pair {{a}}-{{b}} units {{units:,.5f}}  •  "
                           f"Net ${{net:,.2f}} ({{pool_label}})  •  Gross ${{gross:,.2f}}  •  Refunds ${{refunds_val:,.2f}}  •  Jackpot ${{jackpot_val:,.2f}}")
        except Exception as e:
            st.exception(e)
# ======================= End DUE (Duet) — EXA collation format =======================


# === DUE: Declared for Result (auto-calc the 3 winning duets) ===
if pool == 'DUE':
    import re
    import itertools
    import pandas as pd

    with st.expander("🏁 Declared from Result (Duet)", expanded=False):
        st.caption("Enter the finishing order (e.g., 1-2-3-4). We'll calculate the 3 winning Duets: 1-2, 1-3, 2-3.")
        res_str = st.text_input("Result", value="1-2-3-4", key="due_result_str_v1")

        if st.button("Calculate DUE Declared for Result", key="btn_due_declared_for_result_v1"):
            nums = [int(x) for x in re.split(r"[^0-9]+", res_str) if x.strip().isdigit()]
            if len(nums) < 3:
                st.warning("Please enter at least the first three placings, e.g., 1-2-3-4")
            else:
                a, b, c = nums[0], nums[1], nums[2]
                pairs = [(a,b), (a,c), (b,c)]

                due_df = st.session_state.get('due_df')
                if due_df is None or due_df.empty:
                    st.warning("No DUE pairs loaded. Import the DUE collation first.")
                else:
                    gross = float(st.session_state.get('gross_sales', 0.0))
                    refunds_val = float(st.session_state.get('refunds', 0.0))
                    jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                    net = net_pool(gross, refunds_val, jackpot_val, rules.commission)

                    rows = []
                    for x, y in pairs:
                        sel_ab = due_df[(due_df['First'].astype(int)==x) & (due_df['Second'].astype(int)==y)]
                        sel_ba = due_df[(due_df['First'].astype(int)==y) & (due_df['Second'].astype(int)==x)]
                        units = 0.0
                        if len(sel_ab): units += float(sel_ab['Units'].iloc[0])
                        if len(sel_ba): units += float(sel_ba['Units'].iloc[0])
                        div = None
                        if units > 0:
                            div = float(due_declared_dividend(net, units, rules))
                        rows.append({'Pair': f'{x}-{y}', 'Units': units, 'Declared per $1': div})

                    out_df = pd.DataFrame(rows)
                    st.dataframe(out_df, hide_index=True, use_container_width=True)
                    st.download_button("Download DUE declared (CSV)", out_df.to_csv(index=False).encode("utf-8"),
                                       "due_declared_from_result.csv", "text/csv")
