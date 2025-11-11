from __future__ import annotations

import re



# === FFR module helpers (ensured) ===
def _ffr_percent_to_units(percent: float) -> float:
    try:
        p = float(percent or 0.0)
        return p/100.0 if p > 1.5 else p
    except Exception:
        return 0.0
def _ffr_expand_set(spec: str) -> list[int]:
    out = set()
    for part in (spec or "").replace(" ", "").replace("\n", "").split(","):
        if not part:
            continue
        if "-" in part:
            a,b = part.split("-", 1)
            try:
                out.update(range(int(a), int(b)+1))
            except Exception:
                continue
        else:
            try:
                out.add(int(part))
            except Exception:
                continue
    return sorted(out)
def _ffr_tickets_to_units(tickets, percent) -> dict:
    """Expand tickets into ordered (First,Second,Third,Fourth) -> units dict."""
    u_per = _ffr_percent_to_units(percent)
    units = {}
    def add(a,b,c,d, u):
        key = (int(a), int(b), int(c), int(d))
        units[key] = units.get(key, 0.0) + float(u)
    for t in (tickets or []):
        kind = getattr(t, "kind", "STANDOUT")
        legs = getattr(t, "legs", {}) or {}
        F = legs.get("F") or legs.get("A") or []
        S = legs.get("S") or legs.get("B") or []
        T = legs.get("T") or legs.get("C") or []
        Q = legs.get("Q") or legs.get("D") or []
        if kind == "STRAIGHT":
            if F and S and T and Q:
                add(F[0], S[0], T[0], Q[0], u_per)
            continue
        # STANDOUT/PERM
        for a in F:
            for b in S:
                if b == a: 
                    continue
                for c in T:
                    if c in (a,b): 
                        continue
                    for d in Q:
                        if d in (a,b,c): 
                            continue
                        add(a,b,c,d, u_per)
    return units
def _ffr_parse_transactions_robust(text: str) -> dict:
    """Robust FFR transaction parser (module-level).
    Handles wrapped 'P F(...)/F(...)/F(...)/F(...)', 'No. of combs' + 'Percent per comb[.]',
    and POOL/SUB totals.
    Returns dict: tickets=[ticket-like with .legs], plus gross/refunds/jackpot/percent.
    """
    import re as _re
    tickets = []
    gross = 0.0
    refunds = 0.0
    jackpot = 0.0
    percent = None
    if not text:
        return dict(tickets=tickets, gross=gross, refunds=refunds, jackpot=jackpot, percent=percent)

    # Normalise per-line spacing but keep newlines for multi-line regex
    glyphs = "│┼┤├┬┴┌┐└┘─━—│┃╭╮╯╰▕▏|"
    trans = str.maketrans({c: " " for c in glyphs})
    lines = []
    for line in text.splitlines():
        line = " ".join(line.translate(trans).rstrip().split())
        lines.append(line)
    t = "\n".join(lines)

    # Totals
    m_pool = _re.search(r"(?is)POOL\s+TOTALS:\s.*?FFR\s*\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2}))", t)
    if m_pool:
        try:
            gross = float(m_pool.group(1).replace(',', ''))
        except Exception:
            pass
    m_sub = _re.search(r"(?is)SUB\s+TOTALS:\s.*?SELLS\s*\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2}))(?:\s+PAID\s+SELL\s*\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})))?", t)
    if m_sub:
        try:
            sells = float(m_sub.group(1).replace(',', ''))
            if not gross:
                gross = sells
        except Exception:
            pass

    # Percent per comb and comb count
    combs = None
    m_meta = _re.search(r"No\.\s*of\s*combs\s*=\s*(\d+)\s*,\s*Percent\s+per\s+comb\.?\s*([0-9.]+)%", t, _re.I | _re.S)
    if m_meta:
        try:
            percent = float(m_meta.group(2))
            combs = int(m_meta.group(1))
        except Exception:
            pass

    # Selection sets P F(...)/F(...)/F(...)/F(...)
    m_sets = _re.search(r"\bP\s*F\((.*?)\)\s*/\s*F\((.*?)\)\s*/\s*F\((.*?)\)\s*/\s*F\((.*?)\)", t, _re.I | _re.S)
    if m_sets:
        raw_sets = list(m_sets.groups())
        def _clean(s: str) -> str:
            s = (s or "").replace("\n", "").replace(" ", "")
            s = _re.sub(r"-\s*(\d+)", r"-\1", s)
            return s
        sets_clean = [_clean(s) for s in raw_sets]
        F = _ffr_expand_set(sets_clean[0]); S = _ffr_expand_set(sets_clean[1])
        T = _ffr_expand_set(sets_clean[2]); Q = _ffr_expand_set(sets_clean[3])
        class _MiniTicket:
            def __init__(self, legs):
                self.kind="STANDOUT"; self.stake=0.0; self.lines=0; self.legs=legs
        tk = _MiniTicket(dict(F=F,S=S,T=T,Q=Q))
        # if combs/percent known, set stake/lines
        try:
            if combs is not None and percent is not None:
                u_per = percent/100.0 if percent > 1.0 else percent
                tk.lines = int(combs)
                tk.stake = float(combs) * u_per
        except Exception:
            pass
        tickets.append(tk)

    return dict(tickets=tickets, gross=gross, refunds=refunds, jackpot=jackpot, percent=percent)
# === end helpers ===



# ---- FFR declared dividend helper (module-level) ----
def _ffr_declared_dividend(*, units: float, payout_pool: float, break_step: float, display_dp: int, break_mode: str="down") -> float:
    """
    Compute declared dividend per $1 given units and payout pool (already net of commission/refunds + jackpots).
    Applies breakage to the nearest/below/above step (e.g., 0.10).
    """
    try:
        u = float(units or 0.0)
        if u <= 0.0:
            return 0.0
        per = float(payout_pool or 0.0) / u
        from decimal import Decimal
        dper = Decimal(per) * Decimal(100)
        step = Decimal(str(break_step)) * Decimal(100)
        if break_mode == "down":
            dper -= (dper % step)
        elif break_mode == "up":
            rem = dper % step
            if rem:
                dper += (step - rem)
        else:  # nearest
            rem = dper % step
            if rem >= (step // 2):
                dper += (step - rem)
            else:
                dper -= rem
        return float((dper / Decimal(100)).quantize(Decimal(10) ** -int(display_dp)))
    except Exception:
        return 0.0
# ---- end helper ----


# ===== QAD/EQD compatibility shims (added) =====
try:
    _eqd_units_per1
except NameError:
    def _eqd_units_per1_legacy(order, tickets):
        try:
            return _qad_units_per1(order, tickets)  # delegate if available later
        except NameError:
            u = 0.0
            for t in (tickets or []):
                try:
                    lines = getattr(t, "lines", 0)
                    stake = getattr(t, "stake", 0.0)
                    if lines and stake and _eqd_ticket_covers(t, tuple(order)):
                        u += stake / lines
                except Exception:
                    pass
            return u

try:
    _eqd_declared_from_tx
except NameError:
    def _eqd_declared_from_tx(order, tickets, commission, gross, refunds, jackpot, percent, break_step, display_dp, break_mode="down"):
        if "_qad_declared_from_tx" in globals():
            return _qad_declared_from_tx(order, tickets, commission, gross, refunds, jackpot, percent, break_step, display_dp, break_mode)  # type: ignore
        base = max((gross or 0.0) - (refunds or 0.0), 0.0) + max(jackpot or 0.0, 0.0)
        pct = (percent/100.0) if (percent is not None) else (1.0 - float(commission))
        net = base * pct
        u = _eqd_units_per1(order, tickets)
        if u <= 0:
            return 0.0
        per = net / u
        from decimal import Decimal
        try:
            dper = Decimal(per) * Decimal(100)
            step = Decimal(str(break_step)) * Decimal(100)
            if break_mode == "up":
                rem = dper % step
                if rem:
                    dper += (step - rem)
            elif break_mode == "down":
                dper -= (dper % step)
            else:
                rem = dper % step
                if rem >= (step // 2):
                    dper += (step - rem)
                else:
                    dper -= rem
            return float((dper / Decimal(100)).quantize(Decimal(10) ** -int(display_dp)))
        except Exception:
            return round(per, int(display_dp))

# Ticket type used by parser
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class _EQDTicket:
    kind: str            # "FIELD" etc.
    stake: float
    lines: int
    legs: Dict[str, List[int]]  # keys: F,S,T,Q
    rover: int|None = None

def _eqd_ticket_covers(ticket: _EQDTicket, order: tuple[int,int,int,int]) -> bool:
    a,b,c,d = order
    F = ticket.legs.get("F", [])
    S = ticket.legs.get("S", [])
    T = ticket.legs.get("T", [])
    Q = ticket.legs.get("Q", [])
    return (a in F) and (b in S) and (c in T) and (d in Q)

def _eqd_parse_transactions(text: str) -> dict:
    """Parse EQD/QAD scan-sells for 4-leg flexi tickets (newline-safe) and surface:
       - tickets: list of _EQDTicket(kind="FIELD", stake, lines, legs)
       - gross:   pool sells (from SUB TOTALS/POOL TOTALS, else sum of stakes)
       - refunds / jackpot: if found (optional)
       - percent: combined units-per-$1 across all tickets (sum(stake/lines) * 100)
    """
    import re as _re

    def _rng(lo: int, hi: int):
        lo, hi = int(lo), int(hi)
        return list(range(min(lo, hi), max(lo, hi) + 1))

    def _expand(expr: str):
        s = _re.sub(r"\s+", "", expr or "")
        if not s:
            return []
        out = []
        for part in s.split(","):
            if not part:
                continue
            m = _re.match(r"^(\d+)-(\d+)$", part)
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
                if lo > hi:
                    lo, hi = hi, lo
                out += list(range(lo, hi + 1))
            elif part.isdigit():
                out.append(int(part))
        return sorted(set(out))

    tickets = []
    gross = 0.0
    refunds = 0.0
    jackpot = 0.0
    percent = None

    if not text:
        return dict(tickets=tickets, gross=gross, refunds=refunds, jackpot=jackpot, percent=percent)

    t = text.replace("\r", "")

    # Gross from SUB TOTALS or POOL TOTALS (QAD/QAD)
    m = (_re.search(r"SUB\s+TOTALS:.*?SELLS\s*\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2}))", t, _re.I | _re.S)
         or _re.search(r"POOL\s+TOTALS:\s.*?(?:EQD|QAD)\s*\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2}))", t, _re.I | _re.S))
    if m:
        try:
            gross = float(m.group(1).replace(",", ""))
        except Exception:
            pass

    # Optional: refunds / jackpot if present
    m_ref = _re.search(r"REFUNDS?\s*\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2}))", t, _re.I)
    if m_ref:
        try:
            refunds = float(m_ref.group(1).replace(",", ""))
        except Exception:
            pass
    m_jp = _re.search(r"JACKPOT\s*\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2}))", t, _re.I)
    if m_jp:
        try:
            jackpot = float(m_jp.group(1).replace(",", ""))
        except Exception:
            pass

    # Newline-safe ticket matcher: F(...)/F(...)/F(...)/F(...), allow whitespace between F and '('
    rx = _re.compile(
        r"F\s*\(\s*([^)]+?)\s*\)\s*/\s*F\s*\(\s*([^)]+?)\s*\)\s*/\s*F\s*\(\s*([^)]+?)\s*\)\s*/\s*F\s*\(\s*([^)]+?)\s*\)",
        _re.I | _re.S
    )

    percent_sum_units = 0.0

    for mm in rx.finditer(t):
        L1 = _expand(mm.group(1)); L2 = _expand(mm.group(2)); L3 = _expand(mm.group(3)); L4 = _expand(mm.group(4))

        # Local window for stake and "No. of combs"
        w0, w1 = max(0, mm.start() - 300), min(len(t), mm.end() + 300)
        window = t[w0:w1]

        # No. of combs
        lines = None
        cc = _re.search(r"No\.\s*of\s*combs\s*=\s*(\d+)", window, _re.I)
        if cc:
            try:
                lines = int(cc.group(1))
            except Exception:
                lines = None

        # Stake: last monetary value before "No. of combs"
        stake = None
        before = window[:cc.start()] if cc else window
        monies = list(_re.finditer(r"([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2}))", before))
        if monies:
            try:
                stake = float(monies[-1].group(1).replace(",", ""))
            except Exception:
                stake = None

        # Fallbacks
        if lines is None and L1 and L2 and L3 and L4:
            lines = len(L1) * len(L2) * len(L3) * len(L4)

        if stake is None:
            # try "Percent per comb." per ticket
            pp = _re.search(r"Percent\s+per\s+comb\.\s*([0-9.]+)\s*%", window, _re.I)
            if pp and lines:
                try:
                    units_per = float(pp.group(1)) / 100.0
                    stake = units_per * float(lines)
                except Exception:
                    pass

        if not (L1 and L2 and L3 and L4 and lines and stake is not None):
            continue

        # Build ticket
        legs = dict(F=L1, S=L2, T=L3, Q=L4)
        tickets.append(_EQDTicket("FIELD", float(stake), int(lines), legs))

        # accumulate units per $1
        try:
            percent_sum_units += (float(stake) / float(lines))
        except Exception:
            pass

    # Fill gross if not present
    if gross <= 0 and tickets:
        try:
            gross = float(sum(getattr(tt, "stake", 0.0) for tt in tickets))
        except Exception:
            pass

    # Final percent from tickets (units per $1 → percent)
    if percent is None:
        try:
            percent = float(percent_sum_units * 100.0) if percent_sum_units > 0 else None
        except Exception:
            percent = None

    return dict(tickets=tickets, gross=gross, refunds=refunds, jackpot=jackpot, percent=percent)
# ===== End shims/parser =====


# ===== Compatibility helpers: make _eqd_units_per1 available to QAD code =====
def _eqd_units_per1(order, tickets):
    """Compatibility function.
    - If QAD helpers are present, delegate to _qad_units_per1.
    - Otherwise, if EQD helpers are present, compute using _eqd_ticket_covers.
    This avoids NameError when QAD code calls _eqd_units_per1.
    """
    try:
        # Use QAD implementation when available
        return _qad_units_per1(order, tickets)  # type: ignore[name-defined]
    except NameError:
        pass
    # Fallback: generic calculation (requires _eqd_ticket_covers to be defined)
    u = 0.0
    for t in (tickets or []):
        try:
            lines = getattr(t, "lines", 0)
            stake = getattr(t, "stake", 0.0)
            if lines <= 0 or stake <= 0:
                continue
            if '_eqd_ticket_covers' in globals() and _eqd_ticket_covers(t, tuple(order)):
                u += (stake / lines)
        except Exception:
            continue
    return u
# ===== End compatibility helpers =====

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

# === SINGLE_RACE_DEADHEAT_BEGIN ===
def _dh__parse_csv_names(s: str):
    parts = [p.strip() for p in (s or "").replace("/", ",").split(",")]
    return [p for p in parts if p]

def _dh__build_ordered_rows(tie_sets):
    import itertools
    for combo in itertools.product(*tie_sets):
        if len(set(combo)) != len(combo):
            continue
        yield combo

def _dh__weight_from_ties(tie_sets):
    w = 1.0
    for ts in tie_sets:
        w /= max(1, len(ts))
    return w

def render_single_race_deadheat_expander():
    import streamlit as st
    try:
        import pandas as pd
    except Exception:
        pd = None

    allowed = ["WIN", "QIN", "EXA", "TRI", "FFR"]
    pos_map = {"WIN":1, "QIN":2, "EXA":2, "TRI":3, "FFR":4}

    current_pool = str(st.session_state.get("pool") or st.session_state.get("pool_sel") or "").strip().upper()
    default_idx = allowed.index(current_pool) if current_pool in allowed else 0

    with st.expander("Single-race Dead-heat Expander (WIN/QIN/EXA/TRI/FFR)", expanded=False):
        pool_sel = st.selectbox("Pool", allowed, index=default_idx, key="ui_dh_single_pool")
        npos = pos_map[pool_sel]

        st.caption("Enter tied runners per finishing position. Comma-separated (e.g., A,B). Duplicate names across positions are removed.")

        inputs = []
        for i in range(npos):
            key = f"ui_dh_pos_{i+1}"
            label = f"Position {i+1} ties (comma-separated)"
            val = st.text_input(label, key=key)
            inputs.append(val)

        max_rows = 5000
        if st.button("Generate Dead-heat Outcomes", key="ui_dh_generate"):
            tie_sets = [_dh__parse_csv_names(x) for x in inputs]
            if any(len(ts) == 0 for ts in tie_sets):
                st.warning("Please enter at least one runner for each finishing position.")
                return

            weight = _dh__weight_from_ties(tie_sets)

            if pool_sel == "WIN":
                rows = [{"Outcome": r, "weight": weight} for r in tie_sets[0]]
                if pd is not None:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)
                    st.download_button("Download CSV", df.to_csv(index=False), "deadheat_win.csv", "text/csv", key="ui_dh_dl_win")
                else:
                    st.write(rows)
                return

            rows = []
            count = 0
            for combo in _dh__build_ordered_rows(tie_sets):
                count += 1
                if count > max_rows:
                    st.warning(f"Result capped at {max_rows} rows for performance.")
                    break
                rows.append(combo)

            if pool_sel == "QIN":
                from collections import defaultdict
                agg = defaultdict(float)
                for a, b in rows:
                    if a == b:
                        continue
                    key = tuple(sorted((a, b)))
                    agg[key] += weight
                out = [{"Runner A": k[0], "Runner B": k[1], "weight": v} for k, v in agg.items()]
                if pd is not None:
                    df = pd.DataFrame(out)
                    st.dataframe(df, use_container_width=True)
                    st.download_button("Download CSV", df.to_csv(index=False), "deadheat_qin.csv", "text/csv", key="ui_dh_dl_qin")
                else:
                    st.write(out)
            else:
                cols = [f"Pos {i+1}" for i in range(npos)]
                out = [{**{cols[i]: rows[r][i] for i in range(npos)}, "weight": weight} for r in range(len(rows))]
                if pd is not None:
                    df = pd.DataFrame(out)
                    st.dataframe(df, use_container_width=True)
                    st.download_button("Download CSV", df.to_csv(index=False), f"deadheat_{pool_sel.lower()}.csv", "text/csv", key=f"ui_dh_dl_{pool_sel.lower()}")
                else:
                    st.write(out)

# === SINGLE_RACE_DEADHEAT_END ===


# --- Pre-widget scheduler: apply scheduled session ops before any widgets render ---
def __apply_scheduled_widget_values():
    ss = st.session_state
    # delete first (optional, only if you ever schedule deletions)
    for _k in list(ss.keys()):
        if _k.startswith("__del__") and _k.endswith("__"):
            target = _k[len("__del__"):-len("__")]
            try:
                if target in ss:
                    del ss[target]
            except Exception:
                pass
            finally:
                try:
                    del ss[_k]
                except Exception:
                    pass
    # then set
    for _k in list(ss.keys()):
        if _k.startswith("__set__") and _k.endswith("__"):
            target = _k[len("__set__"):-len("__")]
            try:
                ss[target] = ss[_k]
            except Exception:
                pass
            finally:
                try:
                    del ss[_k]
                except Exception:
                    pass

__apply_scheduled_widget_values()
# -------------------------------------------------------------------------------



# === BG6 (Big6) helpers ===
def parse_bg6_totals(text: str):
    """Parse BG6 transaction/collation dump to extract POOL TOTALS (Gross Sales)."""
    import re as _re
    if not text:
        return None
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    m = _re.search(r"(?ims)^\s*POOL\s+TOTALS\s*:\s*[\s\S]*?^\s*(?:BG6|BG\s*6)\s*\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2}))", t)
    if m:
        try: return float(m.group(1).replace(",", ""))
        except Exception: pass
    m = _re.search(r"(?ims)^\s*SUB\s+TOTALS\s*:\s*[\s\S]*?^\s*SELLS\s*\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2}))", t)
    if m:
        try: return float(m.group(1).replace(",", ""))
        except Exception: pass
    m = _re.search(r"(?im)^\s*(?:BG6|BG\s*6).*?\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2}))\s*$", t)
    if m:
        try: return float(m.group(1).replace(",", ""))
        except Exception: pass
    return None
# === End BG6 helpers ===


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
        except Exception: pass

    # JACKPOT line (if present)
    jackpot_val = None
    m_jp = re.search(r"^\s*JACKPOT\s*:?\s*\$?\s*([-,\d.]+)", t, re.M)
    if m_jp:
        try: jackpot_val = float(m_jp.group(1).replace(",", ""))
        except Exception: pass

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
    horizontal = force_horizontal or any(len(vals) > 1 for _, vals in rows)
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
    return df[["Runner", "Units"]], {"total": total, "num_runners": num_runners, "layout": "horizontal" if horizontal else "vertical", "jackpot": jackpot_val}
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

    
    # JACKPOT line (if present)
    jackpot_val = None
    m_jp = re.search(r'^\s*JACKPOT\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*$', t, re.M | re.I)
    if m_jp:
        try:
            jackpot_val = float(m_jp.group(1).replace(',', ''))
        except Exception:
            pass
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
    return df, {"total": total, "num_runners": num_runners, "jackpot": jackpot_val}




# [dedup removed earlier FunctionDef pla_declared_dividends_with_deficiency spanning 739-794]

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

    # init jackpot meta
    jackpot_val = None

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

    # JACKPOT (prefer 'JACKPOT' / 'JACKPOT IN'; ignore 'SEEDED JACKPOT')
    try:
        m_j = re.search(r'(?mi)^\s*JACKPOT(?:\s+IN)?\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*$', t)
        if m_j:
            jackpot_val = float(m_j.group(1).replace(',', ''))
        elif jackpot_val is None:
            m_s = re.search(r'(?mi)^\s*SEEDED\s+JACKPOT\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*$', t)
            if m_s:
                jackpot_val = float(m_s.group(1).replace(',', ''))
    except Exception:
        pass

    # To DataFrame
    import pandas as pd
    if pairs:
        data = [(f"Runner {a}", f"Runner {b}", float(u)) for (a,b), u in sorted(pairs.items())]
    else:
        data = []
    df = pd.DataFrame(data, columns=["Runner A", "Runner B", "Units"])
        # Fallback: if JACKPOT not present in dump, use current UI value
    if jackpot_val is None:
        try:
            import streamlit as st
            jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
        except Exception:
            jackpot_val = 0.0
    return df, {"total": total, "num_runners": num_runners, "jackpot": jackpot_val}

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
            units = max(0.0, float(m.group(4).replace(',', '')) * float(scale or 1.0))
            out.append((a,b,c,units))
        else:
            m2 = re.search(r"(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", ln)
            if m2:
                rest = ln[m2.end():]
                mnum = re.search(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)", rest)
                if mnum:
                    a,b,c = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
                    units = max(0.0, float(mnum.group(1).replace(',', '')) * float(scale or 1.0))
                    out.append((a,b,c,units))
    df = (pd.DataFrame(out, columns=["First","Second","Third","Units"])
          if out else pd.DataFrame([], columns=["First","Second","Third","Units"]))
    meta = {}
    try:
        mt = re.search(r"TOTAL\s+([0-9,]+(?:\.\d+)?)", t)
        if mt: meta["total"] = float(mt.group(1).replace(",",""))
        mr = re.search(r"NUM\s+RUNNERS\s+(\d+)", t)
        if mr: meta["num_runners"] = int(mr.group(1))
        # JACKPOT (prefer 'JACKPOT' / 'JACKPOT IN'; ignore 'SEEDED JACKPOT')
        try:
            m_j = re.search(r'(?mi)^\s*JACKPOT(?:\s+IN)?\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*$', t)
            if m_j:
                meta['jackpot'] = float(m_j.group(1).replace(',', ''))
            else:
                m_s = re.search(r'(?mi)^\s*SEEDED\s+JACKPOT\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*$', t)
                if m_s:
                    meta['jackpot'] = float(m_s.group(1).replace(',', ''))
        except Exception:
            pass
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
    try:
        return float(f"{float(value):.{dp}f}")
    except Exception:
        try:
            return float(value)
        except Exception:
            return 0.0

def _format_display_str(value: float, dp: int) -> str:
    try:
        return f"{float(value):.{dp}f}"
    except Exception:
        return "0"
def net_pool(gross_sales: float, refunds: float, jackpot_in: float, commission: float) -> float:
    base = max((gross_sales or 0.0) + (jackpot_in or 0.0) - (refunds or 0.0), 0.0)
    return base * (1.0 - max(commission or 0.0, 0.0))

def approximates_from_spread(net: float, spread_units: Dict[str, float], rules: "PoolRules", enforce_min_div: bool = True) -> Dict[str, float]:
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

def approximates_from_spread_by_pool(net: float, spread_units: Dict[str, float], rules: "PoolRules", pool: str, place_winners: int = 3, enforce_min_div: bool = True) -> Dict[str, float]:
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


def dividends_from_spread(net: float, winners: List[str], spread_units: Dict[str, float], rules: "PoolRules", declare_per_winner: bool = True) -> Dict[str, float]:
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

def single_pool_dividend(net: float, winning_units: float, rules: "PoolRules", *, pool: str | None = None, place_winners: int = 1) -> float:
    """Per-$1 declared dividend for a *single* winning selection.

    For WIN:     dividend = net / winning_units
    For PLA:     dividend = (net / place_winners) / winning_units
    Other pools: fallback to WIN logic.

    Returns a formatted string number (via _format_display).
    """
    if winning_units <= 0:
        return 0.0
    eff_net = float(net)
    try:
        if pool and str(pool).upper() == "PLA":
            pw = max(1, int(place_winners or 1))
            eff_net = net / pw if pw > 0 else 0.0
    except Exception:
        pass
    div = 0.0 if eff_net <= 0 else (eff_net / winning_units)
    div = _apply_breakage(div, rules.break_step, rules.break_mode)
    div = max(div, rules.min_div)
    return _format_display(div, rules.display_dp)

# -------------------------------
# Preset rules (sourced from your spec; adjust if jurisdiction differs)
# -------------------------------

POOL_PRESETS = {
    # Single-leg
    "WIN":  {"commission": 0.1500, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 1,  "display_dp": 2},
    "PLA": {"commission": 0.14, "min_div": 1.00, "break_step": 0.10, "break_mode": "down", "display_dp": 2, "max_dividends": 1},
    "QIN":  {"commission": 0.1750, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 1,  "display_dp": 2},
    "EXA":  {"commission": 0.2000, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 1,  "display_dp": 2},
    "TRI":  {"commission": 0.2150, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 1,  "display_dp": 2},
    "FFR":  {"commission": 0.2300, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 1,  "display_dp": 2},
    "DUE":  {"commission": 0.1450, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 1,  "display_dp": 2},

    # Short multi-leg
    "RD":   {"commission": 0.2000, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 2,  "display_dp": 2},
    "DD":   {"commission": 0.2000, "min_div": 1.04, "break_step": 0.10, "break_mode": "nearest", "max_dividends": 2,  "display_dp": 2},
    "TBL":  {"commission": 0.2000, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 3,  "display_dp": 2},

    # Quaddies
    "EQD":  {"commission": 0.2050, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 8,  "display_dp": 2},
    "QAD":  {"commission": 0.2050, "min_div": 1.04, "break_step": 0.10, "break_mode": "down", "max_dividends": 8,  "display_dp": 2},

    # BIG6
    "BIG6": {"commission": 0.2500, "min_div": 1.00, "break_step": 0.10, "break_mode": "down", "max_dividends": 12, "display_dp": 2},
}

# ---- Preset defaults shim (prevents KeyError on missing fields) ----
_SINGLE_LEG = {"WIN","PLA","QIN","EXA","TRI","FFR","DUE"}
_SHORT_MULTI = {"RD","DD","TBL"}
_QUADDIES = {"EQD","QAD"}
_BIG6 = {"BIG6"}
for _pool, _p in POOL_PRESETS.items():
    # sensible defaults by pool family
    if _pool in _SINGLE_LEG:
        _p.setdefault("max_dividends", 1)
    elif _pool in _SHORT_MULTI:
        _p.setdefault("max_dividends", 3)
    elif _pool in _QUADDIES:
        _p.setdefault("max_dividends", 8)
    elif _pool in _BIG6:
        _p.setdefault("max_dividends", 12)
    else:
        _p.setdefault("max_dividends", 8)
    # universal fallbacks
    _p.setdefault("display_dp", 2)
    _p.setdefault("break_step", 0.10)
    _p.setdefault("break_mode", "down")
    _p.setdefault("min_div", 1.00)
# ---- End shim ----


# Hard-coded commissions by jurisdiction (pool -> rate)
COMMISSION_BY_JURISDICTION = {
    "VIC & NSW": {
        "WIN": 0.1500, "PLA": 0.1475, "QIN": 0.1750, "EXA": 0.2000, "TRI": 0.2150,
        "DUE": 0.1450, "FFR": 0.2300, "RD": 0.2000, "DD": 0.2000, "TBL": 0.2000,
        "EQD": 0.2050, "QAD": 0.2050, "BIG6": 0.2500,
    },
    "QLD": {
        "WIN": 0.1500, "PLA": 0.1475, "QIN": 0.1750, "EXA": 0.2000, "TRI": 0.2150,
        "DUE": 0.1450, "FFR": 0.2300, "RD": 0.2000, "DD": 0.2000, "TBL": 0.2000,
        "EQD": 0.2050, "QAD": 0.2050, "BIG6": 0.2500,
    },
}




# =================== Treble (TBL) — EXA collation format =======================
def tbl_parse_exa_collation_text(text: str, scale: float = 1.0):
    # Similar to DD parser, but matches 'a-b-c' triple rows and sums across columns for Units.
    import re as _re
    import pandas as pd
    txt = text or ""

    def _to_float(x):
        try:
            s = x.group(1) if hasattr(x, "group") else x
            return float(str(s).replace(",", ""))
        except Exception:
            return None

    m_total = _re.search(r"(?im)^\s*TOTAL\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)", txt)
    m_jp    = _re.search(r"(?im)^\s*JACKPOT\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)", txt)
    m_nr3   = _re.search(r"(?im)^\s*NUM\s+RUNNERS\s+(\d+)\s+(\d+)\s+(\d+)", txt)
    total   = _to_float(m_total) if m_total else None
    jackpot = _to_float(m_jp) if m_jp else 0.0
    leg1    = int(m_nr3.group(1)) if m_nr3 else None
    leg2    = int(m_nr3.group(2)) if m_nr3 else None
    leg3    = int(m_nr3.group(3)) if m_nr3 else None

    # Lines like "  1- 1- 1   3.893517 3.893517 ..." -> sum across columns into Units
    rows = []
    for m in _re.finditer(r"(?m)^\s*(\d+)\-\s*(\d+)\-\s*(\d+)\s+(.+)$", txt):
        a = int(m.group(1)); b = int(m.group(2)); c = int(m.group(3)); rest = m.group(4).strip()
        vals = [_to_float(x) for x in _re.findall(r"([-,\d]+\.\d+)", rest)]
        _vals = [v for v in vals if v is not None]
        # If all numeric columns are equal, treat it as one source; else sum them
        if _vals and (max(_vals) - min(_vals) < 1e-9):
            units = _vals[0]
        else:
            units = sum(_vals)

        if units and units > 0:
            rows.append((a, b, c, units * float(scale or 1.0)))

    df = pd.DataFrame(rows, columns=['First','Second','Third','Units'])
    meta = {'total': total, 'jackpot': jackpot, 'num_runners_leg1': leg1, 'num_runners_leg2': leg2, 'num_runners_leg3': leg3}
    return df, meta

def tbl_declared_dividend(net: float, triple_units: float, rules: 'PoolRules') -> float:
    if not triple_units or triple_units <= 0:
        return 0.0
    raw = float(net) / float(triple_units)
    div = _apply_breakage(raw, rules.break_step, rules.break_mode)
    div = max(div, rules.min_div)
    return _format_display(div, rules.display_dp)

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title='National Pooling Calculator', layout="wide")

st.title('National Pooling Calculator')
st.caption("Compute net pools, approximates (WillPays), and final dividends (dead-heats supported).")

# Sidebar: Pool + Preset rules
with st.sidebar:
    st.header("Pool & Rules")
    pool = st.selectbox("Pool", list(POOL_PRESETS.keys()), index=0)
    preset = POOL_PRESETS[pool].copy()
    jurisdiction = st.selectbox("Jurisdiction", ["VIC & NSW", "QLD"], index=0, key="jurisdiction")
    try:
        preset["commission"] = COMMISSION_BY_JURISDICTION[jurisdiction][pool]
    except Exception:
        pass

    st.markdown("**Rule Overrides** (optional)")
    commission = st.number_input("Commission (0–1)", key="num_commission_0_1_1", value=float(preset["commission"]), min_value=0.0, max_value=0.99, step=0.01, format="%.2f")
    min_div = st.number_input("Minimum Dividend ($)", key="num_minimum_dividend_1", value=float(preset["min_div"]), min_value=0.0, step=0.01, format="%.2f")
    break_step = st.number_input("Breakage Step ($)", key="num_breakage_step_1", value=float(preset["break_step"]), min_value=0.0, step=0.01, format="%.2f")
    break_mode = st.selectbox("Breakage Mode", ["down", "nearest", "up"], index=["down", "nearest", "up"].index(preset["break_mode"]))
    max_dividends = st.number_input("Max Dividends", key="num_max_dividends_1", value=int(preset["max_dividends"]), min_value=1, max_value=64, step=1)
    display_dp = st.number_input("Display Decimals", key="num_display_decimals_1", value=int(preset["display_dp"]), min_value=0, max_value=4, step=1)

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
    hide_index=True,
    num_rows="dynamic",
    key="spread_editor_top",
    column_config={
        "Runner": st.column_config.TextColumn("Runner"),
        "Units": st.column_config.NumberColumn("Units", min_value=0.0, step=1.0, format="%.0f"),
    }, use_container_width=True)
st.session_state["spread_df"] = edited_spread

# === Importer UI (with "Force horizontal" toggle) ===
if pool == 'WIN':
    # ——— Single‑leg pool inputs (mirrors PLA; safe to ignore if unused) ———
    with st.expander("Single-leg pool inputs", expanded=False):
        colA, colB, colC, colD = st.columns([1,1,1,1])
        with colA:
            _win_gross_sales = st.number_input(
                "Gross Sales ($)",
                value=float(st.session_state.get("gross_sales", 0.0)) ,
                min_value=0.0, step=100.0, format="%.2f", key="win_gross_sales"
            )
        with colB:
            _win_refunds = st.number_input(
                "Refunds ($)",
                value=float(st.session_state.get("refunds", 0.0)),
                min_value=0.0, step=10.0, format="%.2f", key="win_refunds"
            )
        with colC:
            _win_jackpot_in = st.number_input(
                "Jackpot In ($)",
                value=float(st.session_state.get("jackpot_in", 0.0)),
                min_value=0.0, step=10.0, format="%.2f", key="win_jackpot_in"
            )
        with colD:
            _win_units = st.number_input(
                "Single-leg Winning Units",
                value=float(st.session_state.get("win_single_leg_units", 0.0)),
                min_value=0.0, step=1.0, format="%.2f", key="win_single_leg_units"
            )
# Single-race dead-heat expander
    try:
        render_single_race_deadheat_expander()
    except Exception:
        pass
    if st.button("Calculate Single-leg Dividend", key="btn_win_single_div"):
    
        try:
            gross = float(st.session_state.get('gross_sales', 0.0))
            refunds_val = float(st.session_state.get('refunds', 0.0))
            jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
            net = net_pool(gross, refunds_val, jackpot_val, rules.commission)
            st.session_state["net"] = net
    
            win_units = float(
                st.session_state.get("num_single_leg_winning_units_optional_1",
                                     st.session_state.get("win_single_leg_units", 0.0))
            )
            st.session_state["single"] = single_pool_dividend(net, win_units, rules, pool="WIN", place_winners=1)
        except Exception as e:
            st.exception(e)
        # Sync to generic keys so existing logic keeps working where used
        st.session_state["gross_sales"] = float(_win_gross_sales)
        st.session_state["refunds"] = float(_win_refunds)
        st.session_state["jackpot_in"] = float(_win_jackpot_in)
        st.session_state["num_single_leg_winning_units_optional_1"] = float(_win_units)
# ——— End single‑leg inputs ———

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
                # Apply meta totals to both generic and widget keys so inputs reflect parsed values
                if isinstance(meta, dict) and meta.get("total") is not None:
                    try:
                        _tot = float(meta.get("total"))
                        st.session_state["gross_sales"] = _tot
                        st.session_state["__set__win_gross_sales__"] = _tot
                    except Exception:
                        pass
                if isinstance(meta, dict) and meta.get("jackpot") is not None:
                    try:
                        _jp = float(meta.get("jackpot"))
                        st.session_state["jackpot_in"] = _jp
                        st.session_state["__set__win_jackpot_in__"] = _jp
                    except Exception:
                        pass
                st.session_state["refunds"] = 0.0
                st.session_state["__set__win_refunds__"] = 0.0
                st.session_state["last_parse_meta"] = {
                    "num_runners": meta.get("num_runners") if isinstance(meta, dict) else None,
                    "total": meta.get("total") if isinstance(meta, dict) else None,
                    "layout": meta.get("layout") if isinstance(meta, dict) else None,
                }
                st.session_state["just_parsed"] = True

                # refresh inputs to show parsed values
                st.rerun()


if pool == 'QIN':
    st.subheader('Quinella (QIN)')
    st.caption('Import a QIN collation dump → pairs & units. TOTAL will prefill Gross Sales.')
    import pandas as pd
    # ——— Single-leg pool inputs (QIN) ———
    with st.expander('Single-leg pool inputs', expanded=False):
        colA, colB, colC, colD = st.columns([1,1,1,1])
        with colA:
            if 'qin_gross_sales' in st.session_state:
                _qin_gross_sales = st.number_input('Gross Sales ($)', min_value=0.0, step=100.0, format='%.2f', key='qin_gross_sales')
            else:
                _qin_gross_sales = st.number_input('Gross Sales ($)', value=float(st.session_state.get('gross_sales', 0.0)), min_value=0.0, step=100.0, format='%.2f', key='qin_gross_sales')
        with colB:
            if 'qin_refunds' in st.session_state:
                _qin_refunds = st.number_input('Refunds ($)', min_value=0.0, step=10.0, format='%.2f', key='qin_refunds')
            else:
                _qin_refunds = st.number_input('Refunds ($)', value=float(st.session_state.get('refunds', 0.0)), min_value=0.0, step=10.0, format='%.2f', key='qin_refunds')
        with colC:
            if 'qin_jackpot_in' in st.session_state:
                _qin_jackpot_in = st.number_input('Jackpot In ($)', min_value=0.0, step=10.0, format='%.2f', key='qin_jackpot_in')
            else:
                _qin_jackpot_in = st.number_input('Jackpot In ($)', value=float(st.session_state.get('jackpot_in', 0.0)), min_value=0.0, step=10.0, format='%.2f', key='qin_jackpot_in')
        with colD:
            if 'qin_num_single_leg_winning_units_optional_1' in st.session_state:
                _qin_units = st.number_input('Single-leg Winning Units (optional)', min_value=0.0, step=1.0, format='%.2f', key='qin_num_single_leg_winning_units_optional_1')
            else:
                _qin_units = st.number_input('Single-leg Winning Units (optional)', value=float(st.session_state.get('num_single_leg_winning_units_optional_1', 0.0)), min_value=0.0, step=1.0, format='%.2f', key='qin_num_single_leg_winning_units_optional_1')
        # Sync generic keys used by net/div calculations
        try:
            st.session_state['gross_sales'] = float(_qin_gross_sales)
            st.session_state['refunds'] = float(_qin_refunds)
            st.session_state['jackpot_in'] = float(_qin_jackpot_in)
            st.session_state['num_single_leg_winning_units_optional_1'] = float(_qin_units)
        except Exception:
            pass
    # ——— End single‑leg inputs (QIN) ———

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

            # Schedule totals/jackpot into generic & QIN inputs (always)
            try:
                if isinstance(meta_qin, dict):
                    if meta_qin.get('total') is not None:
                        _tot = float(meta_qin.get('total', 0.0))
                        st.session_state['__set__gross_sales__'] = _tot
                        st.session_state['__set__qin_gross_sales__'] = _tot
                    if meta_qin.get('jackpot') is not None:
                        _jp = float(meta_qin.get('jackpot', 0.0))
                        st.session_state['__set__jackpot_in__'] = _jp
                        st.session_state['__set__qin_jackpot_in__'] = _jp
            except Exception:
                pass

            # Save pairs (if any) before forcing UI refresh
            if df_qin is not None and not df_qin.empty:
                st.session_state['qin_df'] = df_qin.copy()
            else:
                st.error('No pairs found.')

            # Refresh once so the scheduled values populate widgets
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

    # Editable QIN pairs table
    qin_df = st.session_state.get('qin_df')
    if qin_df is None:
        qin_df = pd.DataFrame([], columns=['Runner A','Runner B','Units'])
    qin_df = st.data_editor(qin_df, num_rows='dynamic', key='qin_spread_editor',
                            column_config={
                                'Runner A': st.column_config.TextColumn('Runner A'),
                                'Runner B': st.column_config.TextColumn('Runner B'),
                                'Units': st.column_config.NumberColumn('Units', min_value=0.0, step=0.1, format='%.2f'),
                            }, use_container_width=True)
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
                # Directly set PLA table so the editor populates immediately
                st.session_state['pla_df'] = df_pla.copy()
                st.session_state["__set__pla_df__"] = df_pla.copy()
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
                    st.session_state["__set__pla_df__"] = _tmp
                except Exception:
                    pass
                try:
                    _pla_meta2 = parse_meta_from_collation(txt_pla)
                    st.session_state["__set__pla_entries_at_deadline__"] = _pla_meta2.get('entries_at_deadline')
                except Exception:
                    pass
                # Prefill Gross Sales / Jackpot from PLA meta
                try:
                    if isinstance(meta_pla, dict) and meta_pla.get('total') is not None:
                        _tot = float(meta_pla.get('total'))
                        st.session_state['gross_sales'] = _tot
                        st.session_state["__set__win_gross_sales__"] = _tot
                    if isinstance(meta_pla, dict) and meta_pla.get('jackpot') is not None:
                        _jp = float(meta_pla.get('jackpot'))
                        st.session_state['jackpot_in'] = _jp
                        st.session_state["__set__win_jackpot_in__"] = _jp
                    st.session_state['refunds'] = 0.0
                    st.session_state["__set__win_refunds__"] = 0.0
                    import streamlit as _st
                    _st.rerun()
                except Exception:
                    pass
                try:
                    _pla_meta2 = parse_meta_from_collation(txt_pla)
                    st.session_state["__set__pla_entries_at_deadline__"] = _pla_meta2.get('entries_at_deadline')
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
    pla_df = st.data_editor(pla_df, num_rows='fixed', key='pla_spread_editor', use_container_width=True)
    st.session_state["__set__pla_df__"] = pla_df
    
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
    
    if st.button('Calculate Place Declared Dividend(s)', key='btn_pla_div_v1'):
        # Persist current input values so they don't reset on rerun
        st.session_state['gross_sales'] = float(st.session_state.get('gross_sales', 0.0))
        st.session_state['refunds'] = float(st.session_state.get('refunds', 0.0))
        st.session_state['jackpot_in'] = float(st.session_state.get('jackpot_in', 0.0))
        if not pla_sel:
            st.info('Add winning runner(s) above to calculate.')
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
        # --- DEBUG: show parameters used in calculation ---
        st.info(f"PLA debug → commission={rules.commission:.4f}, net={net:.2f}, share(per place)={net/max(1,int(pla_winners_n)):.2f}")
        st.info("Units used: " + ", ".join([f"{int(r)}={winners_units.get(int(r),0.0):.0f}" for r in pla_sel]))

        declared_map = pla_declared_dividends_with_deficiency(net, winners_units, pw, rules)
        for rn in pla_sel:
            st.success(f"Declared PLA dividend for Runner {rn}: ${declared_map.get(int(rn), 0.0)}")

    # Inputs
    colA, colB, colC, colD = st.columns([1,1,1,1])
    with colA:
        gross_sales = st.number_input("Gross Sales ($)", value=float(st.session_state.get("gross_sales", 0.00)), min_value=0.0, step=100.0, format="%.2f", key="gross_sales")
    with colB:
        refunds = st.number_input("Refunds ($)", value=float(st.session_state.get("refunds", 0.00)), min_value=0.0, step=10.0, format="%.2f", key="refunds")
    with colC:
        jackpot_in = st.number_input("Jackpot In ($)", value=float(st.session_state.get("jackpot_in", 0.00)), min_value=0.0, step=10.0, format="%.2f", key="jackpot_in")
    with colD:
        single_leg_units = st.number_input("Single-leg Winning Units (optional)", value=0.0, min_value=0.0, step=1.0, format="%.2f", key='pla_single_leg_units')
    
    st.markdown("---")
    
    st.session_state["spread_df"] = st.data_editor(
        st.session_state.get("spread_df"),
        num_rows="dynamic", key="spread_editor_main", use_container_width=True)

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
                if st.button("Calculate Approximates / WillPays", key="btn_calculate_approximates_willpays_1"):
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
                if st.button("Calculate Declared Dividends", key="btn_calculate_declared_dividends_1"):
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
    
                    
                    # --- PATCH: ensure dead-heat winners render correctly for WIN ---
                    if str(pool).upper() == "WIN":
                        def _norm_name(s):
                            return " ".join(str(s).replace("\u00A0", " ").split()).strip().lower()
                        key_map = { _norm_name(k): k for k in (spread_units or {}).keys() }
                        # resolve winners to exact spread keys and filter those with >0 units
                        winners_keys = []
                        for w in (_winners_tmp or []):
                            k = key_map.get(_norm_name(w))
                            if k and float(spread_units.get(k, 0.0) or 0.0) > 0.0:
                                winners_keys.append(k)
                        if winners_keys:
                            total_units = sum(float(spread_units.get(k, 0.0) or 0.0) for k in winners_keys)
                            base = (net / total_units) if total_units > 0 else 0.0
                            divs = {}
                            for k in winners_keys:
                                d = _apply_breakage(base, rules.break_step, rules.break_mode)
                                d = max(d, rules.min_div)
                                divs[k] = _format_display(d, rules.display_dp)
                        else:
                            divs = {}
                    else:
                        divs = dividends_from_spread(net, _winners_tmp, spread_units, rules, declare_per_winner=True)
                    # --- END PATCH ---

                    if not _winners_tmp:
                        st.warning("Please select at least one winner above before calculating declared dividends.")
                    st.session_state["net"] = net
                    st.session_state["divs"] = divs
            with c3:
                if st.button("Calculate Single-leg Dividend", key="btn_calculate_single_leg_dividend_1"):
                    gross = float(st.session_state.get('gross_sales', 0.0))
                    refunds_val = float(st.session_state.get('refunds', 0.0))
                    jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                    net = net_pool(gross, refunds_val, jackpot_val, rules.commission)
                    st.session_state["net"] = net
                    pw = int(st.session_state.get("pla_winners_n", 3)) if str(pool).upper() == "PLA" else 1
                    st.session_state["single"] = single_pool_dividend(net, single_leg_units, rules, pool=pool, place_winners=pw)
    
            # Results
    
    if 'exa_approxs' in st.session_state:
        st.subheader('EXA Approximates / WillPays ($1)')
        df_exa_app = pd.DataFrame({'Pair': list(st.session_state['exa_approxs'].keys()),
                                   'Approx ($1)': list(st.session_state['exa_approxs'].values())})
        st.dataframe(df_exa_app, width='stretch', hide_index=True)
        st.download_button('Download EXA Approximates (CSV)',
                           df_exa_app.to_csv(index=False).encode('utf-8'),
                           'exa_approximates.csv','text/csv', key='dl_auto_1')

    st.markdown("### Results")
    net_disp = st.session_state.get("net", None)
    if net_disp is not None:
        st.info(f"Net Pool: ${net_disp:,.2f}  (Commission {rules.commission:.2%})")

    if "approxs" in st.session_state:
        st.subheader("Approximates / WillPays ($1)")
        df_a = pd.DataFrame({"Runner": list(st.session_state["approxs"].keys()), "Approx ($1)": list(st.session_state["approxs"].values())})
        st.dataframe(df_a, width='stretch', hide_index=True)
        st.download_button("Download Approximates (CSV)", df_a.to_csv(index=False).encode("utf-8"), "approximates.csv", "text/csv", key='dl_auto_2')

    if "divs" in st.session_state:
        st.subheader("Declared Dividends ($1)")
        df_d = pd.DataFrame({"Declared Dividend": list(st.session_state["divs"].keys()), "Amount ($1)": list(st.session_state["divs"].values())})
        st.dataframe(df_d, width='stretch', hide_index=True)
        st.download_button("Download Declared Dividends (CSV)", df_d.to_csv(index=False).encode("utf-8"), "declared_dividends.csv", "text/csv", key='dl_auto_3')

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
        def _ffr_parse_transactions_robust(text: str) -> dict:
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
            comm = st.number_input("Commission (%)", min_value=0.0, max_value=100.0, value=float(rules.commission*100.0), step=0.05, format="%.2f", key="ffr_tx_comm_v6")
        with c3:
            break_step = st.selectbox("Breakage step", [0.10, 0.05, 0.01], index=0, key="ffr_tx_break_step_v6")
        with c4:
            dp = st.selectbox("Display DP", [2, 3], index=0, key="ffr_tx_dp_v6")

        
        if st.button("Compute FFR from transactions", key="btn_ffr_txn_compute_v6"):
            try:
                parsed = _ffr_parse_transactions_robust(txt_tx)
                tickets = parsed.get("tickets", []) or []
                gross = float(parsed.get("gross") or 0.0)
                refunds = float(parsed.get("refunds") or 0.0)
                jackpot = float(parsed.get("jackpot") or 0.0)
                percent_per_comb = float(parsed.get("percent")) if (use_percent and parsed.get("percent") is not None) else None

                # Directly parse "Percent per comb." from pasted text to avoid propagation bugs
                try:
                    if use_percent:
                        import re as _repp
                        mpp = _repp.search(r"Percent\s+per\s+comb\.?\s*([0-9.]+)%", txt_tx, _repp.I)
                        if mpp:
                            percent_per_comb = float(mpp.group(1))
                except Exception:
                    pass

                units = _ffr_units_per1((int(pick_a), int(pick_b), int(pick_c), int(pick_d)), tickets)

                F=S=T=Q=set()
                for _t in tickets:
                    try:
                        _legs = getattr(_t, "legs", {}) or {}
                    except Exception:
                        _legs = {}
                    F |= set(_legs.get("F", []))
                    S |= set(_legs.get("S", []))
                    T |= set(_legs.get("T", []))
                    Q |= set(_legs.get("Q", []))

                def _valid_combo(a,b,c,d):
                    if F and S and T and Q:
                        return (a in F) and (b in S) and (c in T) and (d in Q) and len({a,b,c,d})==4
                    return True

                if (units == 0.0 or units is None) and percent_per_comb is not None and _valid_combo(int(pick_a), int(pick_b), int(pick_c), int(pick_d)):
                    units = percent_per_comb/100.0 if percent_per_comb > 1.5 else percent_per_comb

                payout_pool = ((1.0 - (comm/100.0)) * gross + jackpot - refunds)

                div = _ffr_declared_dividend(
                    units=units,
                    payout_pool=payout_pool,
                    break_step=float(break_step),
                    display_dp=int(dp),
                    break_mode="down",
                )

                pool_label = (f"Units-per-combo {percent_per_comb/100.0:.5f}" if percent_per_comb is not None else f"Commission {comm:.2f}%")
                msg = (
                    f"**${div:,.2f}** per $1  •  "
                    f"Units {units:,.5f}  •  "
                    f"Payout pool ${payout_pool:,.2f} ({pool_label})  •  "
                    f"Gross ${gross:,.2f}  •  Refunds ${refunds:,.2f}  •  Jackpot ${jackpot:,.2f}"
                )
                st.success(msg)

                st.session_state['ffr_model_price_pending'] = float(div)
                st.session_state['single'] = float(div)
            except Exception as e:
                st.exception(e)
                st.exception(e)
    # =================== End transaction-based FFR ONLY =====================


# =================== Transaction-based EQD ONLY =====================
if pool == 'EQD':
    with st.expander("🧾 Transaction-based EQD (scan sells)", expanded=False):
            import re as _re
            from dataclasses import dataclass
            from typing import Optional, List
            from decimal import Decimal, ROUND_FLOOR
    
            @dataclass
            class _EQDTicket:
                kind: str                  # STRAIGHT | BOX | ROVER | STANDOUT
                stake: float               # total $ on this ticket
                lines: int                 # number of combinations covered
                legs: dict                 # {'F':[...], 'S':[...], 'T':[...], 'Q':[...]
                rover: Optional[int] = None
    
            def _eqd_to_ints(s: str) -> list[int]:
                return [int(x) for x in _re.findall(r"\d+", s or "")]
            def _eqd_parse_transactions(text: str) -> dict:
                """Parse EQD transaction dump.
            
                - Reads POOL TOTALS / SUB TOTALS (SELLS, optional PAID SELL) for gross and payout percent
                - Parses STRAIGHT bets (A/B/C/D) and FIELD bets F(x-y)/F(x-y)/F(x-y)/F(x-y), even when wrapped
                - Amounts may include commas and need not be EOL
                - If totals exist, do **not** add ticket stakes to gross (avoid double count)
                - If no totals, set gross = sum(stakes)
                """
                import re as _re
                from dataclasses import dataclass
                from typing import List, Optional
            
                tickets: List[_EQDTicket] = []
                gross = 0.0
                refunds = 0.0
                jackpot = 0.0
                percent: Optional[float] = None
                got_total = False
            
                if not text:
                    return dict(tickets=tickets, gross=gross, refunds=refunds, jackpot=jackpot, percent=percent)
            
                # ---- Totals ----
                m_pool = _re.search(r"(?is)POOL\s+TOTALS:\s.*?EQD\s*\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2}))", text)
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
                    if "EQD" not in up:
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
                        tickets.append(_EQDTicket("STRAIGHT", amt, 1, legs))
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
                        tickets.append(_EQDTicket("STANDOUT", amt, lines_count, dict(F=F,S=S,T=T,Q=Q)))
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
    
    
    
    
    
            def _eqd_ticket_covers(ticket: _EQDTicket, order: tuple[int,int,int,int]) -> bool:
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
    
            def _qad_units_per1(order: tuple[int,int,int,int], tickets: List[_EQDTicket]) -> float:
                u = 0.0
                for t in tickets:
                    if t.lines <= 0 or t.stake <= 0:
                        continue
                    if _eqd_ticket_covers(t, order):
                        u += (t.stake / t.lines)
                return u
    
            def _eqd_declared_from_tx(order: tuple[int,int,int,int],
                                      tickets: List[_EQDTicket],
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
    
                u = _eqd_units_per1(order, tickets)
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
            up_tx = st.file_uploader("Upload transaction dump (.txt/.log)", type=["txt", "log"], key="eqd_tx_upload_v6")
            default_tx = ""
            if up_tx is not None:
                default_tx = up_tx.read().decode("utf-8", errors="ignore")
            txt_tx = st.text_area("…or paste transaction text here", value=default_tx, height=220, key="eqd_tx_area_v6")
    
            ca, cb, cc, cd = st.columns(4)
            with ca:
                pick_a = st.number_input("First", min_value=1, value=1, step=1, key="eqd_tx_a_v6")
            with cb:
                pick_b = st.number_input("Second", min_value=1, value=2, step=1, key="eqd_tx_b_v6")
            with cc:
                pick_c = st.number_input("Third", min_value=1, value=3, step=1, key="eqd_tx_c_v6")
            with cd:
                pick_d = st.number_input("Fourth", min_value=1, value=4, step=1, key="eqd_tx_d_v6")
    
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                use_percent = st.checkbox("Use 'Percent' from file (if present)", value=False, key="eqd_tx_usepct_v6")
            with c2:
                comm = st.number_input("Commission (%)", min_value=0.0, max_value=100.0, value=float(rules.commission*100.0), step=0.05, format="%.2f", key="eqd_tx_comm_v6")
            with c3:
                break_step = st.selectbox("Breakage step", [0.10, 0.05, 0.01], index=0, key="eqd_tx_break_step_v6")
            with c4:
                dp = st.selectbox("Display DP", [2, 3], index=0, key="eqd_tx_dp_v6")
    
            if st.button("Compute EQD from transactions", key="btn_eqd_txn_compute_v6"):
                try:
                    parsed = _eqd_parse_transactions(txt_tx or "")
                    tickets = parsed["tickets"]
                    gross = parsed["gross"]
                    refunds = parsed["refunds"]
                    jackpot = parsed["jackpot"]
                    percent = parsed["percent"] if use_percent else None
    
                    units = _eqd_units_per1((int(pick_a), int(pick_b), int(pick_c), int(pick_d)), tickets)
                    payout_pool = ((percent/100.0)*gross + jackpot - refunds) if (percent is not None) else ((1.0-comm/100.0)*gross + jackpot - refunds)
                    if payout_pool < 0: payout_pool = 0.0
                    div = _eqd_declared_from_tx(
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
                    st.session_state['eqd_model_price_pending'] = float(div)
                    st.session_state['single'] = float(div)
                except Exception as e:
                    st.exception(e)
        # =================== End transaction-based EQD ONLY =====================
# =================== End transaction-based EQD ONLY =====================





    # Export scenario JSONst.markdown("---")# ===================== FFR — Transaction-based (scan sells) ONLY =====================
# =================== End transaction-based FFR ONLY ===================
st.subheader("Export / Save Scenario")
if st.button("Generate Scenario JSON", key="btn_generate_scenario_json_1"):
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
        num_legs = st.number_input("Number of legs", key="num_number_of_legs_1", value=4, min_value=2, max_value=6, step=1)
    with col_warn:
        st.write("For QAD/EQD use 4 legs, TBL 3, DD/RD 2, BIG6 6. "
                 "Large tie sets create many combinations; we cap at 5000 rows for performance.")

    # Text inputs for leg winners and last leg runners
    leg_inputs = []
    for i in range(1, int(num_legs)):
        leg_inputs.append(st.text_input(f"Leg {i} winners (comma-separated)", key=f"txt_leg_winners_{i}", value="A,B" if i == 1 else ""))
    last_leg_runners = st.text_input(f"Leg {int(num_legs)} (last) runners (comma-separated)", key="txt_leg_int_num_legs_last_runners_comma_separated_1", value="A,B,C")

    # Build grid
    if st.button("Generate Combination Grid", key="btn_generate_combination_grid_1"):
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
        combo_df = st.data_editor(st.session_state["combo_df"], num_rows="dynamic",key="combo_editor_1", use_container_width=True)
        st.session_state["combo_df"] = combo_df

        # Aggregate to spread
        if st.button("Compute Spread From Grid", key="btn_compute_spread_from_grid_1"):
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
                st.download_button("Download Combination Grid (CSV)", csv_data, "combination_grid.csv", "text/csv", key='dl_auto_4')
        with cdr:
            if st.session_state.get("spread_df") is not None:
                csv_data2 = st.session_state["spread_df"].to_csv(index=False).encode("utf-8")
                st.download_button("Download Spread (CSV)", csv_data2, "derived_spread.csv", "text/csv", key='dl_auto_5')

if "scenario_json" in st.session_state:
    st.code(st.session_state["scenario_json"], language="json")
    st.download_button("Download Scenario JSON", st.session_state["scenario_json"].encode("utf-8"), "scenario.json", "application/json", key='dl_auto_6')

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
                        st.session_state['__set__gross_sales__'] = float(meta_tri.get('total'))

                        st.session_state['__set__tri_gross_sales__'] = float(meta_tri.get('total'))
                    except Exception:
                        pass
                # Jackpot scheduling if present
                if isinstance(meta_tri, dict) and meta_tri.get('jackpot') is not None:
                    try:
                        st.session_state['__set__jackpot_in__'] = float(meta_tri.get('jackpot'))
                    except Exception:
                        pass
                # Apply on first click
                st.rerun()
                st.success(f"Loaded {len(df_tri)} triples. (TOTAL={meta_tri.get('total') if isinstance(meta_tri, dict) else None})")

    tri_df = st.session_state.get('tri_df')
    if tri_df is None:
        tri_df = pd.DataFrame([], columns=['First','Second','Third','Units'])
    tri_df = st.data_editor(
        tri_df,
        num_rows='dynamic',
        key='tri_spread_editor',
        column_config={
            'First': st.column_config.NumberColumn('First', min_value=0, step=1, format='%d'),
            'Second': st.column_config.NumberColumn('Second', min_value=0, step=1, format='%d'),
            'Third': st.column_config.NumberColumn('Third', min_value=0, step=1, format='%d'),
            'Units': st.column_config.NumberColumn('Units', min_value=0.0, step=0.1, format='%.4f'),
        }
    , use_container_width=True)
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
                                               'Approx ($1)': [round(float(v), rules.display_dp) for v in approxs_tri.values()]})
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
    """Parse EXA collation into (First, Second, Units).

    The collations often print *multiple numeric columns* per pair (e.g. host/guest, totals).
    Some lines have **0** in the first column but non‑zero later; we must not grab only the first!
    Strategy here:
      - Normalise dashes/whitespace.
      - For every occurrence of "A- B" in a line, capture *all* numeric tokens until the next pair/EOL.
      - Use the **rightmost non‑zero** numeric (or else the maximum) as the units for that pair.
    """
    import re, pandas as pd

    t = (txt or "")
    t = (t.replace("\r", "")
           .replace("\u00a0", " ")
           .replace("–", "-")
           .replace("—", "-")
           .replace("−", "-"))

    lines = [ln for ln in t.split("\n") if ln.strip()]
    out = []
    # pattern for a pair like 2- 3
    pair_re = re.compile(r"(\d+)\s*-\s*(\d+)")
    # pattern for a number, allowing commas and decimal
    num_re = re.compile(r"([0-9][0-9,]*\.?\d*)")

    for ln in lines:
        pos = 0
        while True:
            m = pair_re.search(ln, pos)
            if not m:
                break
            a = int(m.group(1)); b = int(m.group(2))
            # slice after the pair; stop at next pair or EOL
            next_m = pair_re.search(ln, m.end())
            segment = ln[m.end(): next_m.start() if next_m else len(ln)]
            nums = [float(x.replace(",", "")) for x in num_re.findall(segment)]
            units_val = 0.0
            if nums:
                # prefer the rightmost non-zero
                nz = [x for x in nums if x > 0]
                units_val = (nz[-1] if nz else nums[-1])
            units_val *= float(scale or 1.0)
            out.append((a, b, units_val))
            pos = next_m.start() if next_m else len(ln)

    df = pd.DataFrame(out, columns=["First", "Second", "Units"]) if out else pd.DataFrame([], columns=["First", "Second", "Units"])
    meta = {}
    try:
        mt = re.search(r"TOTAL\s+([0-9,]+(?:\.\d+)?)", t)
        if mt:
            meta["total"] = float(mt.group(1).replace(",", ""))
    except Exception:
        pass
    return df, meta

# [dedup removed earlier FunctionDef due_declared_dividend spanning 2789-2795]

# UI
try:
    if pool == 'EXA':
        st.subheader('Exacta (EXA)')
        # ——— Single-leg pool inputs (EXA) ———
        with st.expander('Single-leg pool inputs', expanded=False):
            colA, colB, colC, colD = st.columns([1,1,1,1])
            with colA:
                if 'exa_gross_sales' in st.session_state:
                    _exa_gross_sales = st.number_input('Gross Sales ($)', min_value=0.0, step=100.0, format='%.2f', key='exa_gross_sales')
                else:
                    _exa_gross_sales = st.number_input('Gross Sales ($)', value=float(st.session_state.get('gross_sales', 0.0)), min_value=0.0, step=100.0, format='%.2f', key='exa_gross_sales')
            with colB:
                if 'exa_refunds' in st.session_state:
                    _exa_refunds = st.number_input('Refunds ($)', min_value=0.0, step=10.0, format='%.2f', key='exa_refunds')
                else:
                    _exa_refunds = st.number_input('Refunds ($)', value=float(st.session_state.get('refunds', 0.0)), min_value=0.0, step=10.0, format='%.2f', key='exa_refunds')
            with colC:
                if 'exa_jackpot_in' in st.session_state:
                    _exa_jackpot_in = st.number_input('Jackpot In ($)', min_value=0.0, step=10.0, format='%.2f', key='exa_jackpot_in')
                else:
                    _exa_jackpot_in = st.number_input('Jackpot In ($)', value=float(st.session_state.get('jackpot_in', 0.0)), min_value=0.0, step=10.0, format='%.2f', key='exa_jackpot_in')
            with colD:
                if 'exa_num_single_leg_winning_units_optional_1' in st.session_state:
                    _exa_units = st.number_input('Single-leg Winning Units (optional)', min_value=0.0, step=1.0, format='%.2f', key='exa_num_single_leg_winning_units_optional_1')
                else:
                    _exa_units = st.number_input('Single-leg Winning Units (optional)', value=float(st.session_state.get('num_single_leg_winning_units_optional_1', 0.0)), min_value=0.0, step=1.0, format='%.2f', key='exa_num_single_leg_winning_units_optional_1')
            # Sync back to generic keys for calculations
            try:
                st.session_state['gross_sales'] = float(_exa_gross_sales)
                st.session_state['refunds'] = float(_exa_refunds)
                st.session_state['jackpot_in'] = float(_exa_jackpot_in)
                st.session_state['num_single_leg_winning_units_optional_1'] = float(_exa_units)
            except Exception:
                pass
        # ——— End single‑leg inputs (EXA) ———
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
                    try:
                        df_exa, meta_exa = parse_exa_collation_text(txt_exa, scale=exa_scale)
                    except Exception as e:
                        st.exception(e)
                        df_exa, meta_exa = pd.DataFrame([], columns=['First','Second','Units']), {}

                    # Save pairs first so they survive the refresh
                    if df_exa is not None and len(df_exa) > 0:
                        st.session_state['exa_df'] = df_exa.copy()
                    else:
                        st.warning('No pairs parsed from the text.')

                    # Schedule totals/jackpot into generic inputs (no direct widget writes)
                    try:
                        import re
                        total = (meta_exa or {}).get('total')
                        jackpot = (meta_exa or {}).get('jackpot')

                        # Fallbacks if parser meta is missing
                        if total is None:
                            m = re.search(r'(?mi)^\s*TOTAL\s*[:=]?\s*([\d,]+(?:\.\d+)?)\s*$', txt_exa or '')
                            if m: total = float(m.group(1).replace(',', ''))
                        if jackpot is None:
                            m = re.search(r'(?mi)^\s*JACKPOT(?:\s+IN)?\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*$', txt_exa or '')
                            if m: jackpot = float(m.group(1).replace(',', ''))

                        if total is not None:
                            st.session_state['__set__gross_sales__'] = float(total)
                        st.session_state['__set__exa_gross_sales__'] = float(total)
                        if jackpot is not None:
                            st.session_state['__set__jackpot_in__'] = float(jackpot)
                        st.session_state['__set__exa_jackpot_in__'] = float(jackpot)
                    except Exception:
                        pass

                    # Single refresh so scheduled values populate the top inputs
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
        exa_df = st.session_state.get('exa_df')
        if exa_df is None:
            import pandas as pd
            exa_df = pd.DataFrame([], columns=['First','Second','Units'])

        try:
            exa_df = exa_df.dropna(subset=['First','Second'])
            exa_df['Units'] = exa_df['Units'].astype(float)
        except Exception:
            pass

        exa_df = st.data_editor(exa_df, num_rows='dynamic', key='exa_df_editor', use_container_width=True)

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
                    div = qin_declared_dividend(net, units, rules)
                    approxs_exa[pair] = div
                st.session_state['exa_approxs'] = approxs_exa
                # Show results immediately below the button
                st.subheader('EXA Approximates / WillPays ($1)')
                df_exa_app = pd.DataFrame({'Pair': list(approxs_exa.keys()),
                                           'Approx ($1)': [round(float(v), rules.display_dp) for v in approxs_exa.values()]})
                st.dataframe(df_exa_app, width='stretch', hide_index=True)
                st.download_button('Download EXA Approximates (CSV)',
                                   df_exa_app.to_csv(index=False).encode('utf-8'),
                                   'exa_approximates.csv','text/csv', key='dl_auto_7')
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
                div = qin_declared_dividend(net, pair_units, rules)
                st.info(f'EXA {int(exa_first)}-{int(exa_second)} units={pair_units:.4f}  →  Declared: ${div}')
            except Exception as e:
                st.exception(e)
except Exception as _exa_err:
    st.error(f'EXA UI error: {_exa_err}')

# =================== End EXA Helpers & UI ===================


# =================== FFR Helpers (top-loaded) ===================
import re as _re_ffr
from decimal import Decimal as _D, ROUND_FLOOR as _RF


# [dedup removed earlier FunctionDef parse_ffr_collation spanning 2958-3012]

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
    due_df = st.data_editor(due_df, num_rows='dynamic', key='due_df_editor', use_container_width=True)

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
                div = qin_declared_dividend(net, units, rules)
                approxs_due[pair] = float(div)
            # Present as table
            df_due_app = pd.DataFrame([{'Pair': k, 'Approx per $1': v} for k,v in approxs_due.items()])
            st.dataframe(df_due_app, width='stretch', hide_index=True)
            st.download_button('Download DUE Approximates (CSV)',
                               df_due_app.to_csv(index=False).encode('utf-8'),
                               'due_approximates.csv','text/csv', key='dl_auto_8')
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
                div = qin_declared_dividend(net, units, rules)
                da, db = (a, b) if a <= b else (b, a)
                pool_label = f'Commission {rules.commission*100:.2f}%'
                st.success(f"**${div:,.2f}** per $1  •  Duet {da}-{db} units {units:,.5f}  •  "
                           f"Net ${net:,.2f} ({pool_label})  •  Gross ${gross:,.2f}  •  Refunds ${refunds_val:,.2f}  •  Jackpot ${jackpot_val:,.2f}")
        except Exception as e:
            st.exception(e)
# ======================= End DUE (Duet) — EXA collation format =======================


# === DUE: Declared for Result (auto-calc the 3 winning duets) ===
if pool == 'DUE':  # [dedup disabled duplicate UI block]
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
                            div = float(qin_declared_dividend(net, units, rules))
                        rows.append({'Pair': f'{x}-{y}', 'Units': units, 'Declared per $1': div})

                    out_df = pd.DataFrame(rows)
                    st.dataframe(out_df, hide_index=True, use_container_width=True)
                    st.download_button("Download DUE declared (CSV)", out_df.to_csv(index=False).encode("utf-8"),
                                       "due_declared_from_result.csv", "text/csv", key='dl_auto_9')


# ======================= Running Double (RD) — EXA collation format =======================
def rd_parse_exa_collation_text(text: str, scale: float = 1.0):
    import re as _re
    import pandas as pd
    txt = text or ""
    def _to_float(m):
        try:
            s = m.group(1) if hasattr(m, "group") else m
            return float(str(s).replace(",", ""))
        except Exception:
            return None

    m_total = _re.search(r"(?im)^\s*TOTAL\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)", txt)
    m_jp    = _re.search(r"(?im)^\s*JACKPOT\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)", txt)
    m_nr2   = _re.search(r"(?im)^\s*NUM\s+RUNNERS\s+(\d+)\s+(\d+)", txt)
    total   = _to_float(m_total) if m_total else None
    jackpot = _to_float(m_jp) if m_jp else None
    leg1 = leg2 = None
    if m_nr2:
        leg1 = int(m_nr2.group(1)); leg2 = int(m_nr2.group(2))

    line_re  = _re.compile(r"(?m)^\s*(\d+)\s*-\s*(\d+)\s+(.+)$")
    float_re = _re.compile(r"[\d,]+(?:\.\d+)?")
    rows = []
    for a, b0, tail in line_re.findall(txt):
        a = int(a); b0 = int(b0)
        vals = [float(v.replace(',', '')) for v in float_re.findall(tail)]
        if not vals: continue
        for k, v in enumerate(vals):
            b = b0 + k
            if leg2 is not None and b > leg2: break
            if leg1 is not None and a > leg1: continue
            rows.append({'First': a, 'Second': b, 'Units': float(v) * float(scale or 1.0)})
    df = pd.DataFrame(rows, columns=['First','Second','Units'])
    meta = {'total': total, 'jackpot': jackpot, 'num_runners_leg1': leg1, 'num_runners_leg2': leg2}
    return df, meta

def rd_declared_dividend(net: float, pair_units: float, rules: 'PoolRules') -> float:
    if not pair_units or pair_units <= 0:
        return 0.0
    raw = float(net) / float(pair_units)
    div = _apply_breakage(raw, rules.break_step, rules.break_mode)
    div = max(div, rules.min_div)
    return _format_display(div, rules.display_dp)

if pool == 'RD':
    import streamlit as st, pandas as pd
    st.markdown("### Running Double (RD)")
    with st.expander("📥 Import RD Collation (EXA format)"):
        up_rd = st.file_uploader("Upload .txt/.log", type=["txt","log"], key="rd_file_uploader_v1")
        _rd_text_from_file = up_rd.read().decode(errors="ignore") if up_rd else ""
        txt_rd = st.text_area("Paste RD collation dump (EXA-style pairs)", value=_rd_text_from_file, height=220, key="txt_rd")
        rd_scale = st.number_input("Units scale", value=1.00, min_value=0.0, step=0.01, key="rd_units_scale")
        if st.button("Parse & load RD pairs", key="btn_rd_parse"):
            try:
                df_rd, meta_rd = rd_parse_exa_collation_text(txt_rd, scale=rd_scale)
                st.session_state['rd_df'] = df_rd
                total = meta_rd.get('total') or 0.0
                st.session_state['gross_sales'] = float(total or 0.0)
                st.session_state['jackpot_in']  = float(meta_rd.get('jackpot') or 0.0)
                st.success(f"Loaded {len(df_rd)} pairs. (TOTAL=${(total or 0):,.2f}, R1={meta_rd.get('num_runners_leg1')}, R2={meta_rd.get('num_runners_leg2')})")
            except Exception as e:
                st.exception(e)

    with st.expander("📈 RD Approximates (per $1)", expanded=False):
        st.caption("Approximates use current pool settings (commission, breakage, min dividend).")
        if st.button("Calculate RD Approximates", key="btn_rd_approx"):
            rd_df = st.session_state.get('rd_df')
            if rd_df is None or rd_df.empty:
                st.warning("No RD pairs loaded. Import the RD collation first.")
            else:
                gross = float(st.session_state.get('gross_sales', 0.0))
                refunds_val = float(st.session_state.get('refunds', 0.0))
                jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                net = net_pool(gross, refunds_val, jackpot_val, rules.commission)
                rows = []
                for _, r in rd_df.iterrows():
                    units = float(r['Units'])
                    if units <= 0: continue
                    div = rd_declared_dividend(net, units, rules)
                    rows.append({'First': int(r['First']), 'Second': int(r['Second']), 'Units': units, 'Approx per $1': div})
                if rows:
                    out_df = pd.DataFrame(rows).sort_values(['First','Second']).reset_index(drop=True)
                    st.dataframe(out_df, hide_index=True, use_container_width=True)
                    st.download_button("Download RD approximates (CSV)", out_df.to_csv(index=False).encode("utf-8"),
                                       "rd_approximates.csv", "text/csv", key='dl_auto_10')
                else:
                    st.info("No positive-unit pairs found.")

    with st.expander("🏁 Declared from Result (RD)", expanded=False):
        st.caption("Enter the two winners (Race 1 winner, Race 2 winner), e.g., 1-7.")
        res_str = st.text_input("Result", value="1-7", key="rd_result_str_v1")
        if st.button("Calculate RD Declared", key="btn_rd_declared_for_result_v1"):
            import re as _re
            nums = [int(x) for x in _re.split(r"[^0-9]+", res_str) if x.strip().isdigit()]
            if len(nums) < 2:
                st.warning("Please enter two runners like 1-7 (first-leg winner, second-leg winner).")
            else:
                a, b = nums[0], nums[1]
                rd_df = st.session_state.get('rd_df')
                if rd_df is None or rd_df.empty:
                    st.warning("No RD pairs loaded. Import the RD collation first.")
                else:
                    gross = float(st.session_state.get('gross_sales', 0.0))
                    refunds_val = float(st.session_state.get('refunds', 0.0))
                    jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                    net = net_pool(gross, refunds_val, jackpot_val, rules.commission)
                    sel = rd_df[(rd_df['First'].astype(int)==a) & (rd_df['Second'].astype(int)==b)]
                    units = float(sel['Units'].iloc[0]) if len(sel) else 0.0
                    if units <= 0:
                        st.warning(f"No units found for pair {a}-{b}.")
                    else:
                        div = rd_declared_dividend(net, units, rules)
                        st.success(f"Declared for {a}-{b}: **${div}** (units {units:,.4f}, gross ${gross:,.2f}, commission={rules.commission:.2%}).")
# ======================= End RD =======================


# ===================== EQD — Transaction-based (scan sells) =====================
if pool == "EQD" and False:  # [dedup disabled duplicate UI block]
    import re as _re
    from dataclasses import dataclass
    from typing import Optional, List
    from decimal import Decimal, ROUND_FLOOR

    @dataclass
    class _EQDTicket:
        kind: str                  # STRAIGHT | STANDOUT (F ranges)
        stake: float               # total $ on this ticket
        lines: int                 # number of combinations covered
        legs: dict                 # {'F':[...], 'S':[...], 'T':[...], 'Q':[...]]

    def _eqd_rng(lo: int, hi: int) -> list[int]:
        lo, hi = int(lo), int(hi)
        return list(range(min(lo, hi), max(lo, hi)+1))

    def _eqd_ticket_covers(ticket: _EQDTicket, order: tuple[int,int,int,int]) -> bool:
        a,b,c,d = order
        F,S,T,Q = ticket.legs["F"], ticket.legs["S"], ticket.legs["T"], ticket.legs["Q"]
        if ticket.kind == "STRAIGHT":
            return F == [a] and S == [b] and T == [c] and Q == [d]
        # Standout/Field tickets: all legs must include and four must be distinct
        if a not in F or b not in S or c not in T or d not in Q:
            return False
        return len({a,b,c,d}) == 4

    def _eqd_units_per1(order: tuple[int,int,int,int], tickets: List[_EQDTicket]) -> float:
        u = 0.0
        for t in tickets:
            if t.lines <= 0 or t.stake <= 0:
                continue
            if _eqd_ticket_covers(t, order):
                u += (t.stake / t.lines)
        return u

    def _eqd_parse_transactions(text: str):
        """
        Parse EQD transaction dump for tickets and pool totals.
        - Reads POOL TOTALS / SUB TOTALS (SELLS, optional PAID SELL) for gross and percent
        - Parses STRAIGHT bets (A/B/C/D) and FIELD bets F(x-y)/F(x-y)/F(x-y)/F(x-y)
          with tolerant whitespace/wrapping.
        Returns dict: {tickets, gross, refunds, jackpot, percent}
        """
        tickets: List[_EQDTicket] = []
        gross = 0.0
        refunds = 0.0
        jackpot = 0.0
        percent: Optional[float] = None
        got_total = False
        if not text:
            return dict(tickets=tickets, gross=gross, refunds=refunds, jackpot=jackpot, percent=percent)

        # ---- Totals (robust) ----
        m_pool = _re.search(r"(?is)POOL\\s+TOTALS:\\s.*?EQD\\s*\\$\\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\\.[0-9]{2}))", text)
        if m_pool:
            try:
                gross = float(m_pool.group(1).replace(',', ''))
                got_total = True
            except Exception:
                pass
        m_sub = _re.search(r"(?is)SUB\\s+TOTALS:\\s.*?SELLS\\s*\\$\\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\\.[0-9]{2}))(?:.*?PAID\\s+SELL\\s*\\$\\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\\.[0-9]{2})))?", text)
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
        lines = text.splitlines()
        i = 0
        while i < len(lines):
            raw = lines[i]
            up  = " ".join(raw.upper().split())
            if "FFR" not in up:
                i += 1; continue

            # Amount (stake) = last currency-like number on the line
            m_amt = _re.findall(r"([0-9]{1,3}(?:,[0-9]{3})*(?:\\.[0-9]{2}))", raw)
            amt = float(m_amt[-1].replace(',', '')) if m_amt else None

            # Join next lines to capture wrapped field/straight spec
            joined = " ".join([raw, (lines[i+1] if i+1 < len(lines) else ""), (lines[i+2] if i+2 < len(lines) else "")])

            # STRAIGHT e.g. "1/2/3/4"
            m_st = _re.search(r"\\b(\\d+)[-/](\\d+)[-/](\\d+)[-/](\\d+)\\b", joined)
            if m_st and amt is not None:
                a,b,c,d = [int(x) for x in m_st.groups()]
                legs = dict(F=[a], S=[b], T=[c], Q=[d])
                tickets.append(_EQDTicket("STRAIGHT", amt, 1, legs))
                if not got_total: gross += amt
                i += 1; continue

            # FIELD F(x-y)/F(..)/F(..)/F(..)
            m_field = _re.search(r"F\\(\\s*(\\d+)\\s*-\\s*(\\d+)\\)\\s*/\\s*F\\(\\s*(\\d+)\\s*-\\s*(\\d+)\\)\\s*/\\s*F\\(\\s*(\\d+)\\s*-\\s*(\\d+)\\)\\s*/\\s*F\\(\\s*(\\d+)\\s*-\\s*(\\d+)\\)", joined)
            if m_field and amt is not None:
                a1,b1,a2,b2,a3,b3,a4,b4 = [int(x) for x in m_field.groups()]
                F = _eqd_rng(a1,b1); S = _eqd_rng(a2,b2); T = _eqd_rng(a3,b3); Q = _eqd_rng(a4,b4)
                ahead = "\\n".join(lines[i+1:i+6])
                m_combs = _re.search(r"No\\.\\s*of\\s*combs\\s*=\\s*(\\d+)", ahead, flags=_re.I)
                lines_count = int(m_combs.group(1)) if m_combs else 0
                if lines_count <= 0:
                    n = len(F) if (len(F)==len(S)==len(T)==len(Q) and F==S==T==Q) else 0
                    lines_count = n*(n-1)*(n-2)*(n-3) if n >= 4 else 0
                tickets.append(_EQDTicket("STANDOUT", amt, lines_count, dict(F=F,S=S,T=T,Q=Q)))
                if not got_total: gross += amt
                i += 1; continue

            i += 1

        # Percent override if present anywhere
        m_pct = _re.search(r"(?i)\\bPERCENT\\s+([0-9]+(?:\\.[0-9]+)?)\\b", text)
        if m_pct:
            try:
                percent = float(m_pct.group(1))
            except Exception:
                pass

        # If we never found totals, fall back to sum of stakes
        if (not got_total) and (not gross) and tickets:
            try:
                gross = float(sum(getattr(t, 'stake', 0.0) for t in tickets))
            except Exception:
                pass

        return dict(tickets=tickets, gross=gross, refunds=refunds, jackpot=jackpot, percent=percent)

    def _eqd_declared_from_tx(order: tuple[int,int,int,int],
                              tickets: List[_EQDTicket],
                              commission: float,
                              gross: float,
                              refunds: float,
                              jackpot: float,
                              percent: Optional[float],
                              break_step: float,
                              display_dp: int,
                              break_mode: str = "down") -> float:
        base = max((gross or 0.0) - (refunds or 0.0), 0.0) + max(jackpot or 0.0, 0.0)
        pct = (percent / 100.0) if percent is not None else (1.0 - float(commission))
        net = base * pct

        u = _eqd_units_per1(order, tickets)
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

    # --- UI ---
    # Legacy EQD v1 controls fully removed

# =================== End Running Double =======================


# =================== Daily Double (DD) — EXA collation format =======================
def dd_parse_exa_collation_text(text: str, scale: float = 1.0):
    # Identical structure to RD parsing; supports 'Collation ... D-D from ALL' format
    import re as _re
    import pandas as pd
    txt = text or ""
    def _to_float(m):
        try:
            s = m.group(1) if hasattr(m, "group") else m
            return float(str(s).replace(",", ""))
        except Exception:
            return None

    m_total = _re.search(r"(?im)^\s*TOTAL\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)", txt)
    m_jp    = _re.search(r"(?im)^\s*JACKPOT\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)", txt)
    m_nr2   = _re.search(r"(?im)^\s*NUM\s+RUNNERS\s+(\d+)\s+(\d+)", txt)
    total   = _to_float(m_total) if m_total else None
    jackpot = _to_float(m_jp) if m_jp else 0.0
    leg1    = int(m_nr2.group(1)) if m_nr2 else None
    leg2    = int(m_nr2.group(2)) if m_nr2 else None

    # Lines like "  1- 7   41.333332  41.333332 ..." -> sum across columns for units
    rows = []
    for m in _re.finditer(r"(?m)^\s*(\d+)-\s*(\d+)\s+(.+)$", txt):
        a = int(m.group(1)); b = int(m.group(2)); rest = m.group(3).strip()
        vals = [_to_float(x) for x in _re.findall(r"([-\d,]+\.\d+)", rest)]
        units = sum([v for v in vals if v is not None])
        if units and units > 0:
            rows.append((a, b, units * float(scale or 1.0)))

    import pandas as pd
    df = pd.DataFrame(rows, columns=['First','Second','Units'])
    meta = {'total': total, 'jackpot': jackpot, 'num_runners_leg1': leg1, 'num_runners_leg2': leg2}
    return df, meta

def dd_declared_dividend(net: float, pair_units: float, rules: 'PoolRules') -> float:
    if not pair_units or pair_units <= 0:
        return 0.0
    raw = float(net) / float(pair_units)
    div = _apply_breakage(raw, rules.break_step, rules.break_mode)
    div = max(div, rules.min_div)
    return _format_display(div, rules.display_dp)

if pool == 'DD':
    import streamlit as st, pandas as pd
    st.markdown("### Daily Double (DD)")
    with st.expander("📥 Import DD Collation (EXA format)"):
        up_dd = st.file_uploader("Upload .txt/.log", type=["txt","log"], key="dd_file_uploader_v1")
        _dd_text_from_file = up_dd.read().decode(errors="ignore") if up_dd else ""
        txt_dd = st.text_area("Paste DD collation dump (EXA-style pairs)", value=_dd_text_from_file, height=220, key="txt_dd")
        dd_scale = st.number_input("Units scale", value=1.00, min_value=0.0, step=0.01, key="dd_units_scale")
        if st.button("Parse & load DD pairs", key="btn_dd_parse"):
            try:
                df_dd, meta_dd = dd_parse_exa_collation_text(txt_dd, scale=dd_scale)
                st.session_state['dd_df'] = df_dd
                total = meta_dd.get('total') or 0.0
                st.session_state['gross_sales'] = float(total or 0.0)
                st.session_state['jackpot_in']  = float(meta_dd.get('jackpot') or 0.0)
                st.success(f"Loaded {len(df_dd)} pairs. (TOTAL=${total:.2f}, JACKPOT=${float(meta_dd.get('jackpot') or 0.0):.2f}, R1={meta_dd.get('num_runners_leg1')}, R2={meta_dd.get('num_runners_leg2')})")
            except Exception as e:
                st.error(f"Failed to parse: {{e}}")

    
    with st.expander("📈 DD Approximates (per $1)", expanded=False):
        st.caption("Approximates use current pool settings (commission, breakage, min dividend).")
        if st.button("Calculate DD Approximates", key="btn_dd_approx"):
            dd_df = st.session_state.get('dd_df')
            if dd_df is None or dd_df.empty:
                st.warning("No DD pairs loaded. Import the DD collation first.")
            else:
                gross = float(st.session_state.get('gross_sales', 0.0))
                refunds_val = float(st.session_state.get('refunds', 0.0))
                jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                rules = PoolRules(**POOL_PRESETS['DD'])
                net = net_pool(gross, refunds_val, jackpot_val, rules.commission)
                rows = []
                for _, r in dd_df.iterrows():
                    units = float(r['Units'])
                    if units <= 0: 
                        continue
                    div = dd_declared_dividend(net, units, rules)
                    rows.append({'First': int(r['First']), 'Second': int(r['Second']), 'Units': units, 'Approx per $1': div})
                if rows:
                    out_df = pd.DataFrame(rows).sort_values(['First','Second']).reset_index(drop=True)
                    st.dataframe(out_df, hide_index=True, use_container_width=True)
                    st.download_button("Download DD approximates (CSV)", out_df.to_csv(index=False).encode("utf-8"),
                                       "dd_approximates.csv", "text/csv", key='dl_auto_11')
                else:
                    st.info("No positive-unit pairs found.")

    with st.expander("🏁 Declared from Result (DD)", expanded=False):
        st.caption("Enter the two leg winners (e.g., 1 and 7) or use the selectors below.")
        ca, cb = st.columns([1,1])
        with ca:
            pick_a = st.number_input("First leg winner", min_value=1, value=1, step=1, key="dd_win_a_decl")
        with cb:
            pick_b = st.number_input("Second leg winner", min_value=1, value=1, step=1, key="dd_win_b_decl")
        if st.button("Calculate DD Declared", key="btn_dd_declared"):
            dd_df = st.session_state.get('dd_df')
            if dd_df is None or dd_df.empty:
                st.warning("No DD pairs loaded. Import the DD collation first.")
            else:
                rules = PoolRules(**POOL_PRESETS['DD'])
                gross = float(st.session_state.get('gross_sales', 0.0))
                refunds_val = float(st.session_state.get('refunds', 0.0))
                jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                net = net_pool(gross, refunds_val, jackpot_val, rules.commission)
                win_row = dd_df[(dd_df['First']==pick_a) & (dd_df['Second']==pick_b)]
                units = float(win_row['Units'].iloc[0]) if len(win_row)>0 else 0.0
                declared = dd_declared_dividend(net, units, rules)
                st.success(f"**${declared}**  *per $1* • Units {units:.6f} • Payout pool ${net:,.2f} (Commission {rules.commission*100:.2f}%, Gross ${gross:,.2f} • Refunds {refunds_val:.2f} • Jackpot {jackpot_val:.2f})")

# =================== End Daily Double (DD) =======================




if pool == 'TBL':
    import pandas as pd
    st.markdown("### Treble (TBL)")
    with st.expander("📥 Import TBL Collation (EXA format)"):
        up_tbl = st.file_uploader("Upload .txt/.log", type=["txt","log"], key="tbl_file_uploader_v1")
        _tbl_text_from_file = up_tbl.read().decode(errors="ignore") if up_tbl else ""
        txt_tbl = st.text_area("Paste TBL collation dump (EXA-style triples)", value=_tbl_text_from_file, height=220, key="txt_tbl")
        tbl_scale = st.number_input("Units scale", value=1.00, min_value=0.0, step=0.01, key="tbl_units_scale")
        if st.button("Parse & load TBL triples", key="btn_tbl_parse"):
            try:
                df_tbl, meta_tbl = tbl_parse_exa_collation_text(txt_tbl, scale=tbl_scale)
                st.session_state['tbl_df'] = df_tbl
                total = meta_tbl.get('total') or 0.0
                st.session_state['gross_sales'] = float(total or 0.0)
                st.session_state['jackpot_in']  = float(meta_tbl.get('jackpot') or 0.0)
                st.success(f"Loaded {len(df_tbl)} triples. (TOTAL=${total:.2f}, JACKPOT=${float(meta_tbl.get('jackpot') or 0.0):.2f}, R1={meta_tbl.get('num_runners_leg1')}, R2={meta_tbl.get('num_runners_leg2')}, R3={meta_tbl.get('num_runners_leg3')})")
            except Exception as e:
                st.error(f"Failed to parse: {e}")

    with st.expander("📈 TBL Approximates (per $1)", expanded=False):
        st.caption("Approximates use current pool settings (commission, breakage, min dividend).")
        if st.button("Calculate TBL Approximates", key="btn_tbl_approx"):
            tbl_df = st.session_state.get('tbl_df')
            if tbl_df is None or tbl_df.empty:
                st.warning("No TBL triples loaded. Import the TBL collation first.")
            else:
                gross = float(st.session_state.get('gross_sales', 0.0))
                refunds_val = float(st.session_state.get('refunds', 0.0))
                jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                rules_tbl = PoolRules(**POOL_PRESETS['TBL'])
                net = net_pool(gross, refunds_val, jackpot_val, rules_tbl.commission)
                rows = []
                for _, r in tbl_df.iterrows():
                    units = float(r['Units'])
                    if units <= 0:
                        continue
                    div = tbl_declared_dividend(net, units, rules_tbl)
                    rows.append({'First': int(r['First']), 'Second': int(r['Second']), 'Third': int(r['Third']), 'Units': units, 'Approx per $1': div})
                if rows:
                    out_df = pd.DataFrame(rows).sort_values(['First','Second','Third']).reset_index(drop=True)
                    st.dataframe(out_df, hide_index=True, use_container_width=True)
                    st.download_button("Download TBL approximates (CSV)", out_df.to_csv(index=False).encode("utf-8"),
                                       "tbl_approximates.csv", "text/csv", key='dl_auto_12')
                else:
                    st.info("No positive-unit triples found.")

    with st.expander("🏁 Declared from Result (TBL)", expanded=False):
        st.caption("Enter the three leg winners (e.g., 1, 7 and 13) or use the selectors below.")
        ca, cb, cc = st.columns([1,1,1])
        with ca:
            pick_a = st.number_input("First leg winner", min_value=1, value=1, step=1, key="tbl_win_a")
        with cb:
            pick_b = st.number_input("Second leg winner", min_value=1, value=1, step=1, key="tbl_win_b")
        with cc:
            pick_c = st.number_input("Third leg winner", min_value=1, value=1, step=1, key="tbl_win_c")
        if st.button("Calculate TBL Declared", key="btn_tbl_declared"):
            tbl_df = st.session_state.get('tbl_df')
            if tbl_df is None or tbl_df.empty:
                st.warning("No TBL triples loaded. Import the TBL collation first.")
            else:
                rules_tbl = PoolRules(**POOL_PRESETS['TBL'])
                gross = float(st.session_state.get('gross_sales', 0.0))
                refunds_val = float(st.session_state.get('refunds', 0.0))
                jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                net = net_pool(gross, refunds_val, jackpot_val, rules_tbl.commission)
                win_row = tbl_df[(tbl_df['First']==pick_a) & (tbl_df['Second']==pick_b) & (tbl_df['Third']==pick_c)]
                units = float(win_row['Units'].iloc[0]) if len(win_row)>0 else 0.0
                declared = tbl_declared_dividend(net, units, rules_tbl)
                st.success(f"**${declared}** per $1 • Units {units:.6f} • Net pool ${net:,.2f} (Commission {rules_tbl.commission*100:.2f}%, Gross ${gross:,.2f} • Refunds ${refunds_val:,.2f} • Jackpot ${jackpot_val:,.2f})")

if pool == "QAD":
    # ======================= Begin QAD =======================
    with st.expander("🧾 Transaction-based QAD (scan sells)", expanded=False):
                import re as _re
                from dataclasses import dataclass
                from typing import Optional, List
                from decimal import Decimal, ROUND_FLOOR
    
                @dataclass
                class _QADTicket:
                    kind: str                  # STRAIGHT | BOX | ROVER | STANDOUT
                    stake: float               # total $ on this ticket
                    lines: int                 # number of combinations covered
                    legs: dict                 # {'F':[...], 'S':[...], 'T':[...], 'Q':[...]
                    rover: Optional[int] = None
    
                def _qad_to_ints(s: str) -> list[int]:
                    return [int(x) for x in _re.findall(r"\d+", s or "")]
                def _qad_parse_transactions(text: str) -> dict:
                    """Parse QAD transaction dump.
            
                    - Reads POOL TOTALS / SUB TOTALS (SELLS, optional PAID SELL) for gross and payout percent
                    - Parses STRAIGHT bets (A/B/C/D) and FIELD bets F(x-y)/F(x-y)/F(x-y)/F(x-y), even when wrapped
                    - Amounts may include commas and need not be EOL
                    - If totals exist, do **not** add ticket stakes to gross (avoid double count)
                    - If no totals, set gross = sum(stakes)
                    """
                    import re as _re
                    from dataclasses import dataclass
                    from typing import List, Optional
            
                    tickets: List[_QADTicket] = []
                    gross = 0.0
                    refunds = 0.0
                    jackpot = 0.0
                    percent: Optional[float] = None
                    got_total = False
            
                    if not text:
                        return dict(tickets=tickets, gross=gross, refunds=refunds, jackpot=jackpot, percent=percent)
            
                    # ---- Totals ----
                    m_pool = _re.search(r"(?is)POOL\s+TOTALS:\s.*?EQD\s*\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2}))", text)
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
                        if "QAD" not in up:
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
                            tickets.append(_QADTicket("STRAIGHT", amt, 1, legs))
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
                            tickets.append(_QADTicket("STANDOUT", amt, lines_count, dict(F=F,S=S,T=T,Q=Q)))
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
    
    
    
    
    
                def _qad_ticket_covers(ticket: _QADTicket, order: tuple[int,int,int,int]) -> bool:
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
    
                def _qad_units_per1(order: tuple[int,int,int,int], tickets: List[_QADTicket]) -> float:
                    u = 0.0
                    for t in tickets:
                        if t.lines <= 0 or t.stake <= 0:
                            continue
                        if _eqd_ticket_covers(t, order):
                            u += (t.stake / t.lines)
                    return u
    
                def _qad_declared_from_tx(order: tuple[int,int,int,int],
                                          tickets: List[_QADTicket],
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
    
                    u = _eqd_units_per1(order, tickets)
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
                up_tx = st.file_uploader("Upload transaction dump (.txt/.log)", type=["txt", "log"], key="eqd_tx_upload_v6")
                default_tx = ""
                if up_tx is not None:
                    default_tx = up_tx.read().decode("utf-8", errors="ignore")
                txt_tx = st.text_area("…or paste transaction text here", value=default_tx, height=220, key="eqd_tx_area_v6")
    
                ca, cb, cc, cd = st.columns(4)
                with ca:
                    pick_a = st.number_input("First", min_value=1, value=1, step=1, key="eqd_tx_a_v6")
                with cb:
                    pick_b = st.number_input("Second", min_value=1, value=2, step=1, key="eqd_tx_b_v6")
                with cc:
                    pick_c = st.number_input("Third", min_value=1, value=3, step=1, key="eqd_tx_c_v6")
                with cd:
                    pick_d = st.number_input("Fourth", min_value=1, value=4, step=1, key="eqd_tx_d_v6")
    
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    use_percent = st.checkbox("Use 'Percent' from file (if present)", value=False, key="eqd_tx_usepct_v6")
                with c2:
                    comm = st.number_input("Commission (%)", min_value=0.0, max_value=100.0, value=float(rules.commission*100.0), step=0.05, format="%.2f", key="eqd_tx_comm_v6")
                with c3:
                    break_step = st.selectbox("Breakage step", [0.10, 0.05, 0.01], index=0, key="eqd_tx_break_step_v6")
                with c4:
                    dp = st.selectbox("Display DP", [2, 3], index=0, key="eqd_tx_dp_v6")
                # ---- Minimal EQD/QAD transactions parser (guarded) ----
                if '_eqd_parse_transactions' not in globals():
                    import re as _re
                    from dataclasses import dataclass
                    from typing import Optional, List

                    @dataclass
                    class _EQDTicket:
                        kind: str                  # STRAIGHT | BOX | ROVER | STANDOUT
                        stake: float               # total $ on this ticket
                        lines: int                 # number of combinations covered
                        legs: dict                 # {'F':[...], 'S':[...], 'T':[...], 'Q':[...]} for QAD
                        rover: Optional[int] = None

                    def _eqd_to_ints(s: str) -> list[int]:
                        return [int(x) for x in _re.findall(r"\d+", s or "")]

                    def _eqd_parse_transactions(text: str) -> dict:
                        """Parse EQD/QAD transaction dump and return tickets + pool totals.
                        Very tolerant: picks up 'POOL TOTALS'/'SUB TOTALS' blocks and simple STRAIGHT lines like
                        '1-2-3-4   $5.00'. Returns dict: tickets, gross, refunds, jackpot, percent.
                        """
                        tickets: List[_EQDTicket] = []
                        gross = 0.0
                        refunds = 0.0
                        jackpot = 0.0
                        percent: Optional[float] = None
                        got_total = False

                        if not text:
                            return dict(tickets=tickets, gross=gross, refunds=refunds, jackpot=jackpot, percent=percent)

                        # Totals (accept EQD or QAD labels)
                        m_pool = _re.search(r"(?is)POOL\s+TOTALS:\s.*?(?:EQD|QAD)\s*\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2}))", text)
                        if m_pool:
                            try:
                                gross = float(m_pool.group(1).replace(',', ''))
                                got_total = True
                            except Exception:
                                pass

                        m_sub = _re.search(
                            r"(?is)SUB\s+TOTALS:\s.*?SELLS\s*\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2}))"
                            r"(?:.*?PAID\s+SELL\s*\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2})))?",
                            text
                        )
                        if m_sub:
                            try:
                                sells = float(m_sub.group(1).replace(',', ''))
                                if gross <= 0:
                                    gross = sells
                                if m_sub.group(2):
                                    paid = float(m_sub.group(2).replace(',', ''))
                                    if sells > 0:
                                        percent = (paid / sells) * 100.0
                            except Exception:
                                pass

                        m_ref = _re.search(r"(?is)REFUNDS?\s*\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2}))", text)
                        if m_ref:
                            try:
                                refunds = float(m_ref.group(1).replace(',', ''))
                            except Exception:
                                pass

                        m_jp = _re.search(r"(?is)JACKPOT(?:\s+IN)?\s*\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2}))", text)
                        if m_jp:
                            try:
                                jackpot = float(m_jp.group(1).replace(',', ''))
                            except Exception:
                                pass

                        # Tickets — simple STRAIGHT lines with optional label
                        for m in _re.finditer(r"(?im)^(?:STRAIGHT|STANDOUT|ROVER|BOX)?\s*([0-9F][^$]*)\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2}))", text):
                            pick_str = m.group(1)
                            stake = float(m.group(2).replace(',', ''))
                            # split legs by '-' or '/'
                            parts = [p.strip() for p in _re.split(r"[-/]", pick_str) if p.strip()]
                            legs = {'F': [], 'S': [], 'T': [], 'Q': []}
                            try:
                                if len(parts) >= 4:
                                    legs['F'] = _eqd_to_ints(parts[0])
                                    legs['S'] = _eqd_to_ints(parts[1])
                                    legs['T'] = _eqd_to_ints(parts[2])
                                    legs['Q'] = _eqd_to_ints(parts[3])
                                elif len(parts) == 3:
                                    legs['F'] = _eqd_to_ints(parts[0])
                                    legs['S'] = _eqd_to_ints(parts[1])
                                    legs['T'] = _eqd_to_ints(parts[2])
                                else:
                                    continue
                                tickets.append(_EQDTicket(kind="STRAIGHT", stake=stake, lines=1, legs=legs))
                                if not got_total:
                                    gross += stake
                            except Exception:
                                continue

                        return dict(tickets=tickets, gross=gross, refunds=refunds, jackpot=jackpot, percent=percent)
                # ---- end parser ----

    
                if st.button("Compute QAD from transactions", key="btn_eqd_txn_compute_v6"):
                    try:
                        parsed = _eqd_parse_transactions(txt_tx or "")
                        tickets = parsed["tickets"]
                        gross = parsed["gross"]
                        refunds = parsed["refunds"]
                        jackpot = parsed["jackpot"]
                        percent = parsed["percent"] if use_percent else None
    
                        units = _eqd_units_per1((int(pick_a), int(pick_b), int(pick_c), int(pick_d)), tickets)
                        payout_pool = ((percent/100.0)*gross + jackpot - refunds) if (percent is not None) else ((1.0-comm/100.0)*gross + jackpot - refunds)
                        if payout_pool < 0: payout_pool = 0.0
                        div = _eqd_declared_from_tx(
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
                        st.session_state['eqd_model_price_pending'] = float(div)
                        st.session_state['single'] = float(div)
                    except Exception as e:
                        st.exception(e)
            # =================== End transaction-based EQD ONLY =====================
    # =================== End transaction-based EQD ONLY =====================





        # Export scenario JSONst.markdown("---")# ===================== FFR — Transaction-based (scan sells) ONLY =====================
    # =================== End transaction-based FFR ONLY ===================
    st.subheader("Export / Save Scenario")
    if st.button("Generate Scenario JSON", key="btn_generate_scenario_json_2"):
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
            num_legs = st.number_input("Number of legs", key="num_number_of_legs_2", value=4, min_value=2, max_value=6, step=1)
        with col_warn:
            st.write("For QAD/EQD use 4 legs, TBL 3, DD/RD 2, BIG6 6. "
                     "Large tie sets create many combinations; we cap at 5000 rows for performance.")

        # Text inputs for leg winners and last leg runners
        leg_inputs = []
        for i in range(1, int(num_legs)):
            leg_inputs.append(st.text_input(f"Leg {i} winners (comma-separated)", key=f"txt_leg_winners_{i}_2", value="A,B" if i == 1 else ""))
        last_leg_runners = st.text_input(f"Leg {int(num_legs)} (last) runners (comma-separated)", key="txt_leg_int_num_legs_last_runners_comma_separated_2", value="A,B,C")

        # Build grid
        if st.button("Generate Combination Grid", key="btn_generate_combination_grid_2"):
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
            combo_df = st.data_editor(st.session_state["combo_df"], num_rows="dynamic",key="combo_editor_2", use_container_width=True)
            st.session_state["combo_df"] = combo_df

            # Aggregate to spread
            if st.button("Compute Spread From Grid", key="btn_compute_spread_from_grid_2"):
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
                    st.download_button("Download Combination Grid (CSV)", csv_data, "combination_grid.csv", "text/csv", key='dl_auto_13')
            with cdr:
                if st.session_state.get("spread_df") is not None:
                    csv_data2 = st.session_state["spread_df"].to_csv(index=False).encode("utf-8")
                    st.download_button("Download Spread (CSV)", csv_data2, "derived_spread.csv", "text/csv", key='dl_auto_14')

    if "scenario_json" in st.session_state:
        st.code(st.session_state["scenario_json"], language="json")
        st.download_button("Download Scenario JSON", st.session_state["scenario_json"].encode("utf-8"), "scenario.json", "application/json", key='dl_auto_15')

    st.markdown("---")
    st.caption("Dead-heat Matrix lets you enter ties in earlier legs and assign units per combination; we then aggregate to a last-leg spread. "
               "Presets: commissions (aligned to QLD), MIN_DIV_BREAK=10c, min-div 1.04 (most pools) / 1.00 (BIG6), MAXDIV caps (EQD/QAD 8; BIG6 12).")








    # === End improved parser ===




    # === End clean parser ===




    # === End parser ===



    # =====================
    # TRIFECTA (TRI) Section
    # =====================
    if pool == 'TRI' and False:  # [dedup disabled duplicate UI block]
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
                            st.session_state['__set__gross_sales__'] = float(meta_tri.get('total'))

                            st.session_state['__set__tri_gross_sales__'] = float(meta_tri.get('total'))
                        except Exception:
                            pass
                    st.success(f"Loaded {len(df_tri)} triples. (TOTAL={meta_tri.get('total') if isinstance(meta_tri, dict) else None})")

        tri_df = st.session_state.get('tri_df')
        if tri_df is None:
            tri_df = pd.DataFrame([], columns=['First','Second','Third','Units'])
        tri_df = st.data_editor(
            tri_df,
            num_rows='dynamic',
            key='tri_spread_editor',
            column_config={
                'First': st.column_config.NumberColumn('First', min_value=0, step=1, format='%d'),
                'Second': st.column_config.NumberColumn('Second', min_value=0, step=1, format='%d'),
                'Third': st.column_config.NumberColumn('Third', min_value=0, step=1, format='%d'),
                'Units': st.column_config.NumberColumn('Units', min_value=0.0, step=0.1, format='%.4f'),
            }
        , use_container_width=True)
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
                                                   'Approx ($1)': [round(float(v), rules.display_dp) for v in approxs_tri.values()]})
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
        if pool == 'EXA' and False:  # [dedup disabled duplicate UI block]
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

            exa_df = st.data_editor(exa_df, num_rows='dynamic', key='exa_df_editor', use_container_width=True)

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
                        div = qin_declared_dividend(net, units, rules)
                        approxs_exa[pair] = div
                    st.session_state['exa_approxs'] = approxs_exa
                    # Show results immediately below the button
                    st.subheader('EXA Approximates / WillPays ($1)')
                    df_exa_app = pd.DataFrame({'Pair': list(approxs_exa.keys()),
                                               'Approx ($1)': [round(float(v), rules.display_dp) for v in approxs_exa.values()]})
                    st.dataframe(df_exa_app, width='stretch', hide_index=True)
                    st.download_button('Download EXA Approximates (CSV)',
                                       df_exa_app.to_csv(index=False).encode('utf-8'),
                                       'exa_approximates.csv','text/csv', key='dl_auto_16')
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
                    div = qin_declared_dividend(net, pair_units, rules)
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

    if pool == 'DUE':  # [dedup disabled duplicate UI block]
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
        due_df = st.data_editor(due_df, num_rows='dynamic', key='due_df_editor', use_container_width=True)

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
                    div = qin_declared_dividend(net, units, rules)
                    approxs_due[pair] = float(div)
                # Present as table
                df_due_app = pd.DataFrame([{'Pair': k, 'Approx per $1': v} for k,v in approxs_due.items()])
                st.dataframe(df_due_app, width='stretch', hide_index=True)
                st.download_button('Download DUE Approximates (CSV)',
                                   df_due_app.to_csv(index=False).encode('utf-8'),
                                   'due_approximates.csv','text/csv', key='dl_auto_17')
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
                    div = qin_declared_dividend(net, units, rules)
                    pool_label = f'Commission {rules.commission*100:.2f}%'
                    st.success(f"**${{div:,.2f}}** per $1  •  Pair {{a}}-{{b}} units {{units:,.5f}}  •  "
                               f"Net ${{net:,.2f}} ({{pool_label}})  •  Gross ${{gross:,.2f}}  •  Refunds ${{refunds_val:,.2f}}  •  Jackpot ${{jackpot_val:,.2f}}")
            except Exception as e:
                st.exception(e)
    # ======================= End DUE (Duet) — EXA collation format =======================


    # === DUE: Declared for Result (auto-calc the 3 winning duets) ===
    if pool == 'DUE':  # [dedup disabled duplicate UI block]
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
                                div = float(qin_declared_dividend(net, units, rules))
                            rows.append({'Pair': f'{x}-{y}', 'Units': units, 'Declared per $1': div})

                        out_df = pd.DataFrame(rows)
                        st.dataframe(out_df, hide_index=True, use_container_width=True)
                        st.download_button("Download DUE declared (CSV)", out_df.to_csv(index=False).encode("utf-8"),
                                           "due_declared_from_result.csv", "text/csv", key='dl_auto_18')


    # ======================= Running Double (RD) — EXA collation format =======================
    def rd_parse_exa_collation_text(text: str, scale: float = 1.0):
        import re as _re
        import pandas as pd
        txt = text or ""
        def _to_float(m):
            try:
                s = m.group(1) if hasattr(m, "group") else m
                return float(str(s).replace(",", ""))
            except Exception:
                return None

        m_total = _re.search(r"(?im)^\s*TOTAL\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)", txt)
        m_jp    = _re.search(r"(?im)^\s*JACKPOT\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)", txt)
        m_nr2   = _re.search(r"(?im)^\s*NUM\s+RUNNERS\s+(\d+)\s+(\d+)", txt)
        total   = _to_float(m_total) if m_total else None
        jackpot = _to_float(m_jp) if m_jp else None
        leg1 = leg2 = None
        if m_nr2:
            leg1 = int(m_nr2.group(1)); leg2 = int(m_nr2.group(2))

        line_re  = _re.compile(r"(?m)^\s*(\d+)\s*-\s*(\d+)\s+(.+)$")
        float_re = _re.compile(r"[\d,]+(?:\.\d+)?")
        rows = []
        for a, b0, tail in line_re.findall(txt):
            a = int(a); b0 = int(b0)
            vals = [float(v.replace(',', '')) for v in float_re.findall(tail)]
            if not vals: continue
            for k, v in enumerate(vals):
                b = b0 + k
                if leg2 is not None and b > leg2: break
                if leg1 is not None and a > leg1: continue
                rows.append({'First': a, 'Second': b, 'Units': float(v) * float(scale or 1.0)})
        df = pd.DataFrame(rows, columns=['First','Second','Units'])
        meta = {'total': total, 'jackpot': jackpot, 'num_runners_leg1': leg1, 'num_runners_leg2': leg2}
        return df, meta

    def rd_declared_dividend(net: float, pair_units: float, rules: 'PoolRules') -> float:
        if not pair_units or pair_units <= 0:
            return 0.0
        raw = float(net) / float(pair_units)
        div = _apply_breakage(raw, rules.break_step, rules.break_mode)
        div = max(div, rules.min_div)
        return _format_display(div, rules.display_dp)

    if pool == 'RD':
        import streamlit as st, pandas as pd
        st.markdown("### Running Double (RD)")
        with st.expander("📥 Import RD Collation (EXA format)"):
            up_rd = st.file_uploader("Upload .txt/.log", type=["txt","log"], key="rd_file_uploader_v1")
            _rd_text_from_file = up_rd.read().decode(errors="ignore") if up_rd else ""
            txt_rd = st.text_area("Paste RD collation dump (EXA-style pairs)", value=_rd_text_from_file, height=220, key="txt_rd")
            rd_scale = st.number_input("Units scale", value=1.00, min_value=0.0, step=0.01, key="rd_units_scale")
            if st.button("Parse & load RD pairs", key="btn_rd_parse"):
                try:
                    df_rd, meta_rd = rd_parse_exa_collation_text(txt_rd, scale=rd_scale)
                    st.session_state['rd_df'] = df_rd
                    total = meta_rd.get('total') or 0.0
                    st.session_state['gross_sales'] = float(total or 0.0)
                    st.session_state['jackpot_in']  = float(meta_rd.get('jackpot') or 0.0)
                    st.success(f"Loaded {len(df_rd)} pairs. (TOTAL=${(total or 0):,.2f}, R1={meta_rd.get('num_runners_leg1')}, R2={meta_rd.get('num_runners_leg2')})")
                except Exception as e:
                    st.exception(e)

        with st.expander("📈 RD Approximates (per $1)", expanded=False):
            st.caption("Approximates use current pool settings (commission, breakage, min dividend).")
            if st.button("Calculate RD Approximates", key="btn_rd_approx"):
                rd_df = st.session_state.get('rd_df')
                if rd_df is None or rd_df.empty:
                    st.warning("No RD pairs loaded. Import the RD collation first.")
                else:
                    gross = float(st.session_state.get('gross_sales', 0.0))
                    refunds_val = float(st.session_state.get('refunds', 0.0))
                    jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                    net = net_pool(gross, refunds_val, jackpot_val, rules.commission)
                    rows = []
                    for _, r in rd_df.iterrows():
                        units = float(r['Units'])
                        if units <= 0: continue
                        div = rd_declared_dividend(net, units, rules)
                        rows.append({'First': int(r['First']), 'Second': int(r['Second']), 'Units': units, 'Approx per $1': div})
                    if rows:
                        out_df = pd.DataFrame(rows).sort_values(['First','Second']).reset_index(drop=True)
                        st.dataframe(out_df, hide_index=True, use_container_width=True)
                        st.download_button("Download RD approximates (CSV)", out_df.to_csv(index=False).encode("utf-8"),
                                           "rd_approximates.csv", "text/csv", key='dl_auto_19')
                    else:
                        st.info("No positive-unit pairs found.")

        with st.expander("🏁 Declared from Result (RD)", expanded=False):
            st.caption("Enter the two winners (Race 1 winner, Race 2 winner), e.g., 1-7.")
            res_str = st.text_input("Result", value="1-7", key="rd_result_str_v1")
            if st.button("Calculate RD Declared", key="btn_rd_declared_for_result_v1"):
                import re as _re
                nums = [int(x) for x in _re.split(r"[^0-9]+", res_str) if x.strip().isdigit()]
                if len(nums) < 2:
                    st.warning("Please enter two runners like 1-7 (first-leg winner, second-leg winner).")
                else:
                    a, b = nums[0], nums[1]
                    rd_df = st.session_state.get('rd_df')
                    if rd_df is None or rd_df.empty:
                        st.warning("No RD pairs loaded. Import the RD collation first.")
                    else:
                        gross = float(st.session_state.get('gross_sales', 0.0))
                        refunds_val = float(st.session_state.get('refunds', 0.0))
                        jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                        net = net_pool(gross, refunds_val, jackpot_val, rules.commission)
                        sel = rd_df[(rd_df['First'].astype(int)==a) & (rd_df['Second'].astype(int)==b)]
                        units = float(sel['Units'].iloc[0]) if len(sel) else 0.0
                        if units <= 0:
                            st.warning(f"No units found for pair {a}-{b}.")
                        else:
                            div = rd_declared_dividend(net, units, rules)
                            st.success(f"Declared for {a}-{b}: **${div}** (units {units:,.4f}, gross ${gross:,.2f}, commission={rules.commission:.2%}).")
    # ======================= End RD =======================


    # ===================== EQD — Transaction-based (scan sells) =====================
    if pool == "EQD" and False:  # [dedup disabled duplicate UI block]
        import re as _re
        from dataclasses import dataclass
        from typing import Optional, List
        from decimal import Decimal, ROUND_FLOOR

        @dataclass
        class _QADTicket:
            kind: str                  # STRAIGHT | STANDOUT (F ranges)
            stake: float               # total $ on this ticket
            lines: int                 # number of combinations covered
            legs: dict                 # {'F':[...], 'S':[...], 'T':[...], 'Q':[...]]

        def _eqd_rng(lo: int, hi: int) -> list[int]:
            lo, hi = int(lo), int(hi)
            return list(range(min(lo, hi), max(lo, hi)+1))

        def _qad_ticket_covers(ticket: _QADTicket, order: tuple[int,int,int,int]) -> bool:
            a,b,c,d = order
            F,S,T,Q = ticket.legs["F"], ticket.legs["S"], ticket.legs["T"], ticket.legs["Q"]
            if ticket.kind == "STRAIGHT":
                return F == [a] and S == [b] and T == [c] and Q == [d]
            # Standout/Field tickets: all legs must include and four must be distinct
            if a not in F or b not in S or c not in T or d not in Q:
                return False
            return len({a,b,c,d}) == 4

        def _qad_units_per1(order: tuple[int,int,int,int], tickets: List[_QADTicket]) -> float:
            u = 0.0
            for t in tickets:
                if t.lines <= 0 or t.stake <= 0:
                    continue
                if _eqd_ticket_covers(t, order):
                    u += (t.stake / t.lines)
            return u

        def _qad_parse_transactions(text: str):
            """
            Parse QAD transaction dump for tickets and pool totals.
            - Reads POOL TOTALS / SUB TOTALS (SELLS, optional PAID SELL) for gross and percent
            - Parses STRAIGHT bets (A/B/C/D) and FIELD bets F(x-y)/F(x-y)/F(x-y)/F(x-y)
              with tolerant whitespace/wrapping.
            Returns dict: {tickets, gross, refunds, jackpot, percent}
            """
            tickets: List[_QADTicket] = []
            gross = 0.0
            refunds = 0.0
            jackpot = 0.0
            percent: Optional[float] = None
            got_total = False
            if not text:
                return dict(tickets=tickets, gross=gross, refunds=refunds, jackpot=jackpot, percent=percent)

            # ---- Totals (robust) ----
            m_pool = _re.search(r"(?is)POOL\\s+TOTALS:\\s.*?EQD\\s*\\$\\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\\.[0-9]{2}))", text)
            if m_pool:
                try:
                    gross = float(m_pool.group(1).replace(',', ''))
                    got_total = True
                except Exception:
                    pass
            m_sub = _re.search(r"(?is)SUB\\s+TOTALS:\\s.*?SELLS\\s*\\$\\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\\.[0-9]{2}))(?:.*?PAID\\s+SELL\\s*\\$\\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\\.[0-9]{2})))?", text)
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
            lines = text.splitlines()
            i = 0
            while i < len(lines):
                raw = lines[i]
                up  = " ".join(raw.upper().split())
                if "FFR" not in up:
                    i += 1; continue

                # Amount (stake) = last currency-like number on the line
                m_amt = _re.findall(r"([0-9]{1,3}(?:,[0-9]{3})*(?:\\.[0-9]{2}))", raw)
                amt = float(m_amt[-1].replace(',', '')) if m_amt else None

                # Join next lines to capture wrapped field/straight spec
                joined = " ".join([raw, (lines[i+1] if i+1 < len(lines) else ""), (lines[i+2] if i+2 < len(lines) else "")])

                # STRAIGHT e.g. "1/2/3/4"
                m_st = _re.search(r"\\b(\\d+)[-/](\\d+)[-/](\\d+)[-/](\\d+)\\b", joined)
                if m_st and amt is not None:
                    a,b,c,d = [int(x) for x in m_st.groups()]
                    legs = dict(F=[a], S=[b], T=[c], Q=[d])
                    tickets.append(_QADTicket("STRAIGHT", amt, 1, legs))
                    if not got_total: gross += amt
                    i += 1; continue

                # FIELD F(x-y)/F(..)/F(..)/F(..)
                m_field = _re.search(r"F\\(\\s*(\\d+)\\s*-\\s*(\\d+)\\)\\s*/\\s*F\\(\\s*(\\d+)\\s*-\\s*(\\d+)\\)\\s*/\\s*F\\(\\s*(\\d+)\\s*-\\s*(\\d+)\\)\\s*/\\s*F\\(\\s*(\\d+)\\s*-\\s*(\\d+)\\)", joined)
                if m_field and amt is not None:
                    a1,b1,a2,b2,a3,b3,a4,b4 = [int(x) for x in m_field.groups()]
                    F = _eqd_rng(a1,b1); S = _eqd_rng(a2,b2); T = _eqd_rng(a3,b3); Q = _eqd_rng(a4,b4)
                    ahead = "\\n".join(lines[i+1:i+6])
                    m_combs = _re.search(r"No\\.\\s*of\\s*combs\\s*=\\s*(\\d+)", ahead, flags=_re.I)
                    lines_count = int(m_combs.group(1)) if m_combs else 0
                    if lines_count <= 0:
                        n = len(F) if (len(F)==len(S)==len(T)==len(Q) and F==S==T==Q) else 0
                        lines_count = n*(n-1)*(n-2)*(n-3) if n >= 4 else 0
                    tickets.append(_QADTicket("STANDOUT", amt, lines_count, dict(F=F,S=S,T=T,Q=Q)))
                    if not got_total: gross += amt
                    i += 1; continue

                i += 1

            # Percent override if present anywhere
            m_pct = _re.search(r"(?i)\\bPERCENT\\s+([0-9]+(?:\\.[0-9]+)?)\\b", text)
            if m_pct:
                try:
                    percent = float(m_pct.group(1))
                except Exception:
                    pass

            # If we never found totals, fall back to sum of stakes
            if (not got_total) and (not gross) and tickets:
                try:
                    gross = float(sum(getattr(t, 'stake', 0.0) for t in tickets))
                except Exception:
                    pass

            return dict(tickets=tickets, gross=gross, refunds=refunds, jackpot=jackpot, percent=percent)

        def _qad_declared_from_tx(order: tuple[int,int,int,int],
                                  tickets: List[_QADTicket],
                                  commission: float,
                                  gross: float,
                                  refunds: float,
                                  jackpot: float,
                                  percent: Optional[float],
                                  break_step: float,
                                  display_dp: int,
                                  break_mode: str = "down") -> float:
            base = max((gross or 0.0) - (refunds or 0.0), 0.0) + max(jackpot or 0.0, 0.0)
            pct = (percent / 100.0) if percent is not None else (1.0 - float(commission))
            net = base * pct

            u = _eqd_units_per1(order, tickets)
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

        # --- UI ---
        # Legacy EQD v1 controls fully removed

    # =================== End Running Double =======================


    # =================== Daily Double (DD) — EXA collation format =======================
    def dd_parse_exa_collation_text(text: str, scale: float = 1.0):
        # Identical structure to RD parsing; supports 'Collation ... D-D from ALL' format
        import re as _re
        import pandas as pd
        txt = text or ""
        def _to_float(m):
            try:
                s = m.group(1) if hasattr(m, "group") else m
                return float(str(s).replace(",", ""))
            except Exception:
                return None

        m_total = _re.search(r"(?im)^\s*TOTAL\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)", txt)
        m_jp    = _re.search(r"(?im)^\s*JACKPOT\s*[:=]?\s*\$?\s*([\d,]+(?:\.\d+)?)", txt)
        m_nr2   = _re.search(r"(?im)^\s*NUM\s+RUNNERS\s+(\d+)\s+(\d+)", txt)
        total   = _to_float(m_total) if m_total else None
        jackpot = _to_float(m_jp) if m_jp else 0.0
        leg1    = int(m_nr2.group(1)) if m_nr2 else None
        leg2    = int(m_nr2.group(2)) if m_nr2 else None

        # Lines like "  1- 7   41.333332  41.333332 ..." -> sum across columns for units
        rows = []
        for m in _re.finditer(r"(?m)^\s*(\d+)-\s*(\d+)\s+(.+)$", txt):
            a = int(m.group(1)); b = int(m.group(2)); rest = m.group(3).strip()
            vals = [_to_float(x) for x in _re.findall(r"([-\d,]+\.\d+)", rest)]
            units = sum([v for v in vals if v is not None])
            if units and units > 0:
                rows.append((a, b, units * float(scale or 1.0)))

        import pandas as pd
        df = pd.DataFrame(rows, columns=['First','Second','Units'])
        meta = {'total': total, 'jackpot': jackpot, 'num_runners_leg1': leg1, 'num_runners_leg2': leg2}
        return df, meta

    def dd_declared_dividend(net: float, pair_units: float, rules: 'PoolRules') -> float:
        if not pair_units or pair_units <= 0:
            return 0.0
        raw = float(net) / float(pair_units)
        div = _apply_breakage(raw, rules.break_step, rules.break_mode)
        div = max(div, rules.min_div)
        return _format_display(div, rules.display_dp)

    if pool == 'DD' and False:  # [dedup disabled duplicate UI block]
        import streamlit as st, pandas as pd
        st.markdown("### Daily Double (DD)")
        with st.expander("📥 Import DD Collation (EXA format)"):
            up_dd = st.file_uploader("Upload .txt/.log", type=["txt","log"], key="dd_file_uploader_v1")
            _dd_text_from_file = up_dd.read().decode(errors="ignore") if up_dd else ""
            txt_dd = st.text_area("Paste DD collation dump (EXA-style pairs)", value=_dd_text_from_file, height=220, key="txt_dd")
            dd_scale = st.number_input("Units scale", value=1.00, min_value=0.0, step=0.01, key="dd_units_scale")
            if st.button("Parse & load DD pairs", key="btn_dd_parse"):
                try:
                    df_dd, meta_dd = dd_parse_exa_collation_text(txt_dd, scale=dd_scale)
                    st.session_state['dd_df'] = df_dd
                    total = meta_dd.get('total') or 0.0
                    st.session_state['gross_sales'] = float(total or 0.0)
                    st.session_state['jackpot_in']  = float(meta_dd.get('jackpot') or 0.0)
                    st.success(f"Loaded {len(df_dd)} pairs. (TOTAL=${total:.2f}, JACKPOT=${float(meta_dd.get('jackpot') or 0.0):.2f}, R1={meta_dd.get('num_runners_leg1')}, R2={meta_dd.get('num_runners_leg2')})")
                except Exception as e:
                    st.error(f"Failed to parse: {{e}}")

    
        with st.expander("📈 DD Approximates (per $1)", expanded=False):
            st.caption("Approximates use current pool settings (commission, breakage, min dividend).")
            if st.button("Calculate DD Approximates", key="btn_dd_approx"):
                dd_df = st.session_state.get('dd_df')
                if dd_df is None or dd_df.empty:
                    st.warning("No DD pairs loaded. Import the DD collation first.")
                else:
                    gross = float(st.session_state.get('gross_sales', 0.0))
                    refunds_val = float(st.session_state.get('refunds', 0.0))
                    jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                    rules = PoolRules(**POOL_PRESETS['DD'])
                    net = net_pool(gross, refunds_val, jackpot_val, rules.commission)
                    rows = []
                    for _, r in dd_df.iterrows():
                        units = float(r['Units'])
                        if units <= 0: 
                            continue
                        div = dd_declared_dividend(net, units, rules)
                        rows.append({'First': int(r['First']), 'Second': int(r['Second']), 'Units': units, 'Approx per $1': div})
                    if rows:
                        out_df = pd.DataFrame(rows).sort_values(['First','Second']).reset_index(drop=True)
                        st.dataframe(out_df, hide_index=True, use_container_width=True)
                        st.download_button("Download DD approximates (CSV)", out_df.to_csv(index=False).encode("utf-8"),
                                           "dd_approximates.csv", "text/csv", key='dl_auto_20')
                    else:
                        st.info("No positive-unit pairs found.")

        with st.expander("🏁 Declared from Result (DD)", expanded=False):
            st.caption("Enter the two leg winners (e.g., 1 and 7) or use the selectors below.")
            ca, cb = st.columns([1,1])
            with ca:
                pick_a = st.number_input("First leg winner", min_value=1, value=1, step=1, key="dd_win_a_decl")
            with cb:
                pick_b = st.number_input("Second leg winner", min_value=1, value=1, step=1, key="dd_win_b_decl")
            if st.button("Calculate DD Declared", key="btn_dd_declared"):
                dd_df = st.session_state.get('dd_df')
                if dd_df is None or dd_df.empty:
                    st.warning("No DD pairs loaded. Import the DD collation first.")
                else:
                    rules = PoolRules(**POOL_PRESETS['DD'])
                    gross = float(st.session_state.get('gross_sales', 0.0))
                    refunds_val = float(st.session_state.get('refunds', 0.0))
                    jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                    net = net_pool(gross, refunds_val, jackpot_val, rules.commission)
                    win_row = dd_df[(dd_df['First']==pick_a) & (dd_df['Second']==pick_b)]
                    units = float(win_row['Units'].iloc[0]) if len(win_row)>0 else 0.0
                    declared = dd_declared_dividend(net, units, rules)
                    st.success(f"**${declared}**  *per $1* • Units {units:.6f} • Payout pool ${net:,.2f} (Commission {rules.commission*100:.2f}%, Gross ${gross:,.2f} • Refunds {refunds_val:.2f} • Jackpot {jackpot_val:.2f})")

    # =================== End Daily Double (DD) =======================




    if pool == 'TBL':
        import pandas as pd
        st.markdown("### Treble (TBL)")
        with st.expander("📥 Import TBL Collation (EXA format)"):
            up_tbl = st.file_uploader("Upload .txt/.log", type=["txt","log"], key="tbl_file_uploader_v1")
            _tbl_text_from_file = up_tbl.read().decode(errors="ignore") if up_tbl else ""
            txt_tbl = st.text_area("Paste TBL collation dump (EXA-style triples)", value=_tbl_text_from_file, height=220, key="txt_tbl")
            tbl_scale = st.number_input("Units scale", value=1.00, min_value=0.0, step=0.01, key="tbl_units_scale")
            if st.button("Parse & load TBL triples", key="btn_tbl_parse"):
                try:
                    df_tbl, meta_tbl = tbl_parse_exa_collation_text(txt_tbl, scale=tbl_scale)
                    st.session_state['tbl_df'] = df_tbl
                    total = meta_tbl.get('total') or 0.0
                    st.session_state['gross_sales'] = float(total or 0.0)
                    st.session_state['jackpot_in']  = float(meta_tbl.get('jackpot') or 0.0)
                    st.success(f"Loaded {len(df_tbl)} triples. (TOTAL=${total:.2f}, JACKPOT=${float(meta_tbl.get('jackpot') or 0.0):.2f}, R1={meta_tbl.get('num_runners_leg1')}, R2={meta_tbl.get('num_runners_leg2')}, R3={meta_tbl.get('num_runners_leg3')})")
                except Exception as e:
                    st.error(f"Failed to parse: {e}")

        with st.expander("📈 TBL Approximates (per $1)", expanded=False):
            st.caption("Approximates use current pool settings (commission, breakage, min dividend).")
            if st.button("Calculate TBL Approximates", key="btn_tbl_approx"):
                tbl_df = st.session_state.get('tbl_df')
                if tbl_df is None or tbl_df.empty:
                    st.warning("No TBL triples loaded. Import the TBL collation first.")
                else:
                    gross = float(st.session_state.get('gross_sales', 0.0))
                    refunds_val = float(st.session_state.get('refunds', 0.0))
                    jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                    rules_tbl = PoolRules(**POOL_PRESETS['TBL'])
                    net = net_pool(gross, refunds_val, jackpot_val, rules_tbl.commission)
                    rows = []
                    for _, r in tbl_df.iterrows():
                        units = float(r['Units'])
                        if units <= 0:
                            continue
                        div = tbl_declared_dividend(net, units, rules_tbl)
                        rows.append({'First': int(r['First']), 'Second': int(r['Second']), 'Third': int(r['Third']), 'Units': units, 'Approx per $1': div})
                    if rows:
                        out_df = pd.DataFrame(rows).sort_values(['First','Second','Third']).reset_index(drop=True)
                        st.dataframe(out_df, hide_index=True, use_container_width=True)
                        st.download_button("Download TBL approximates (CSV)", out_df.to_csv(index=False).encode("utf-8"),
                                           "tbl_approximates.csv", "text/csv", key='dl_auto_21')
                    else:
                        st.info("No positive-unit triples found.")

        with st.expander("🏁 Declared from Result (TBL)", expanded=False):
            st.caption("Enter the three leg winners (e.g., 1, 7 and 13) or use the selectors below.")
            ca, cb, cc = st.columns([1,1,1])
            with ca:
                pick_a = st.number_input("First leg winner", min_value=1, value=1, step=1, key="tbl_win_a")
            with cb:
                pick_b = st.number_input("Second leg winner", min_value=1, value=1, step=1, key="tbl_win_b")
            with cc:
                pick_c = st.number_input("Third leg winner", min_value=1, value=1, step=1, key="tbl_win_c")
            if st.button("Calculate TBL Declared", key="btn_tbl_declared"):
                tbl_df = st.session_state.get('tbl_df')
                if tbl_df is None or tbl_df.empty:
                    st.warning("No TBL triples loaded. Import the TBL collation first.")
                else:
                    rules_tbl = PoolRules(**POOL_PRESETS['TBL'])
                    gross = float(st.session_state.get('gross_sales', 0.0))
                    refunds_val = float(st.session_state.get('refunds', 0.0))
                    jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
                    net = net_pool(gross, refunds_val, jackpot_val, rules_tbl.commission)
                    win_row = tbl_df[(tbl_df['First']==pick_a) & (tbl_df['Second']==pick_b) & (tbl_df['Third']==pick_c)]
                    units = float(win_row['Units'].iloc[0]) if len(win_row)>0 else 0.0
                    declared = tbl_declared_dividend(net, units, rules_tbl)
                    st.success(f"**${declared}** per $1 • Units {units:.6f} • Net pool ${net:,.2f} (Commission {rules_tbl.commission*100:.2f}%, Gross ${gross:,.2f} • Refunds ${refunds_val:,.2f} • Jackpot ${jackpot_val:,.2f})")
# ======================= End QAD =======================


# ===================== BIG6 — Final Pool (QAD-style, 6 legs) =====================
if pool == "BIG6":
    st.markdown('---')
    st.subheader("BIG6 Final Pool (QAD-style) — 6 Legs")

    # BG6 importer
    with st.expander("📥 Import BG6 Collation (POOL TOTALS → Gross Sales)", expanded=False):
        up_bg6 = st.file_uploader("Upload BG6 .txt/.log", type=["txt","log"], key="bg6_collation_upload_strict")
        default_bg6_text = up_bg6.read().decode("utf-8", errors="ignore") if up_bg6 else ""
        txt_bg6 = st.text_area("…or paste a BG6 dump here", value=default_bg6_text, height=220, key="bg6_collation_area_strict")

        if st.button("Parse & set Gross Sales", key="btn_parse_bg6_strict"):
            try: tot = parse_bg6_totals(txt_bg6)
            except Exception: tot = None
            if tot is None:
                st.error("Could not find POOL TOTALS / SELLS in the pasted BG6 dump.")
            else:
                st.session_state["gross_sales"] = float(tot)
                st.success(f"Gross Sales set from BG6 POOL TOTALS: ${tot:,.2f}")

        jack_in = st.number_input("Jackpot In (carryover added to pool)", min_value=0.0, step=0.01, value=float(st.session_state.get("jackpot_in", 0.0)), key="bg6_jackpot_in_input_strict")
        st.session_state["jackpot_in"] = float(jack_in)

        st.markdown("**Jackpot semantics**")
        sem_choice = st.radio(
            "How should Jackpot be treated?",
            options=[
                "Apply to Major (add to Major pool)",
                "Apply to Supp (not added to Major)",
                "Carry over only (Out)",
            ],
            index=1,
            key="bg6_jackpot_semantics_choice",
        )
        sem_key = {"Apply to Major (add to Major pool)": "major",
                   "Apply to Supp (not added to Major)": "supp",
                   "Carry over only (Out)": "out"}[sem_choice]
        st.session_state["bg6_jackpot_semantics"] = sem_key

    # MUL importer
    with st.expander("📥 Import Multi‑Leg Approximates (MUL)", expanded=False):
        st.caption("Parses 'MULTI‑LEG APPROXIMATES FOR $1 INVESTMENT' to infer per‑runner units and (optionally) jackpot.")
        up_mul = st.file_uploader("Upload MUL .txt", type=["txt"], key="bg6_mul_upload_strict")
        default_mul = up_mul.read().decode("utf-8", errors="ignore") if up_mul else ""
        txt_mul = st.text_area("…or paste a MUL dump here", value=default_mul, height=240, key="bg6_mul_area_strict")
        use_mul_jackpot = st.checkbox("When calculating, prefer Jackpot from MUL for the selected winner(s)", value=False, key="bg6_mul_use_jackpot_strict")

        if st.button("Parse MUL & compute implied units", key="btn_parse_mul_units_strict"):
            import re as _re
            t = (txt_mul or "").replace("\r\n","\n").replace("\r","\n")
            m_comm = _re.search(r"Commission\s+Rate\s+Used\s+([\d.]+)", t, flags=_re.I)
            mul_commission = float(m_comm.group(1))/100.0 if m_comm else None
            m_pool = _re.search(r"POOL\s+TOTAL\s+\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2}))", t, flags=_re.I)
            mul_gross = float(m_pool.group(1).replace(",","")) if m_pool else None

            rows = [
                (int(m.group(1)), float(m.group(2).replace(',', '')), float(m.group(3).replace(',', '')))
                for m in _re.finditer(r"(?:^|\s)(\d{1,3})\s+\$?((?:\d|,)+(?:\.\d{2})?)\s+\$?((?:\d|,)+(?:\.\d{2})?)", t, flags=_re.M)
            ]

            if not rows or mul_gross is None:
                st.error("Could not parse runners or pool total from MUL text.")
            else:
                if mul_commission is None:
                    mul_commission = float(rules.commission)
                net_mul = mul_gross * (1.0 - float(mul_commission))

                implied = {}
                jp_map = {}
                sem = st.session_state.get("bg6_jackpot_semantics", "supp")
                for r,div,jp in rows:
                    jp_map[r] = jp
                    pool_for_units = (net_mul + jp) if sem == "major" else net_mul
                    implied[r] = (pool_for_units / div) if div > 0 else 0.0

                st.session_state["bg6_mul_data"] = {
                    "commission": float(mul_commission),
                    "gross": float(mul_gross),
                    "net": float(net_mul),
                    "implied_units": implied,
                    "jackpots": jp_map,
                    "use_mul_jackpot": bool(use_mul_jackpot),
                }

                import pandas as _pd
                rows_out = [[f"Runner {r}", div, jp, implied.get(r,0.0)] for (r,div,jp) in rows]
                st.dataframe(_pd.DataFrame(rows_out, columns=["Runner","Dividend ($1)","Jackpot","Implied Units"]), use_container_width=True, hide_index=True)

                if st.button("Apply implied units to Runner/Units table", key="btn_apply_mul_units_strict"):
                    data = {"Runner": [], "Units": []}
                    for r,_,_ in rows:
                        data["Runner"].append(f"Runner {r}")
                        data["Units"].append(float(implied.get(r,0.0)))
                    st.session_state["spread_df"] = _pd.DataFrame(data)
                    st.success("Applied implied units to Runner/Units table.")

    # Winners by leg
    leg_cols = st.columns(3)
    winners_by_leg = {}
    for i in range(1, 7):
        with leg_cols[(i-1) % 3]:
            winners_by_leg[i] = st.text_input(f"Leg {i} winners (comma-separated)", key=f"txt_bg6_leg_{i}_winners_strict", value="" if i < 6 else "1")

    with st.expander("Consolation (5 of 6) options", expanded=False):
        enable_consol = st.checkbox("Enable 5/6 consolation dividend", value=False, key="bg6_enable_consol_strict")
        consol_pct = st.number_input("Consolation pool percentage of Final Net Pool (0–100)", min_value=0.0, max_value=100.0, step=0.5, value=0.0, key="bg6_consol_pct_strict")

    def _find_runner_key(n, spread_units: dict):
        import re as _re
        n = int(n)
        cand = f"Runner {n}"
        if cand in spread_units: return cand
        for k in spread_units.keys():
            m = _re.search(r"(\d+)", str(k))
            if m and int(m.group(1)) == n: return k
        if str(n) in spread_units: return str(n)
        return cand

    # Info banners
    counts = {}
    for i in range(1, 7):
        raw = winners_by_leg[i]
        toks = [t.strip() for t in (raw or "").split(",") if t.strip()]
        nums = []
        import re as _re
        for t in toks:
            m = _re.search(r"(\d+)", t)
            if m: nums.append(int(m.group(1)))
        counts[i] = len(set(nums)) if nums else 0
    total_combos = 1
    for i in range(1,7): total_combos *= max(1, counts[i])

    c1,c2 = st.columns([1,1])
    with c1:
        st.info(f"Dead-heat combinations (informational): {total_combos:,d}")
    with c2:
        gross = float(st.session_state.get('gross_sales', 0.0))
        refunds_val = float(st.session_state.get('refunds', 0.0))
        jackpot_val = float(st.session_state.get('jackpot_in', 0.0))
        mul = st.session_state.get("bg6_mul_data")
        if mul and mul.get("commission") is not None:
            try: rules.commission = float(mul["commission"])
            except Exception: pass
        net = net_pool(gross, refunds_val, 0.0, rules.commission)  # banner shows Net without semantics
        st.info(f"Final Net Pool (after commission): ${net:,.2f}  •  Commission: {rules.commission*100:.2f}%")

    # Calculate
    if st.button("Calculate BIG6 Declared Dividend(s)", key="btn_bg6_declared_divs_strict"):
        _df = st.session_state.get("spread_df")
        spread_units = {}
        try:
            for _, r in _df.iterrows():
                k = str(r.get("Runner", "")).strip()
                if k and k.lower() != "runner":
                    spread_units[k] = float(r.get("Units", 0.0))
        except Exception:
            pass

        last_raw = winners_by_leg[6]
        toks = [t.strip() for t in (last_raw or "").split(",") if t.strip()]
        import re as _re
        last_nums = []
        for t in toks:
            m = _re.search(r"(\d+)", t)
            if m: last_nums.append(int(m.group(1)))
        last_nums = sorted(set(last_nums))

        if not last_nums:
            st.warning("Enter at least one numeric runner for Leg 6 (last leg).")
        else:
            winners_keys = [_find_runner_key(n, spread_units) for n in last_nums]

            gross = float(st.session_state.get('gross_sales', 0.0))
            refunds_val = float(st.session_state.get('refunds', 0.0))
            jackpot_val = float(st.session_state.get('jackpot_in', 0.0))

            mul = st.session_state.get("bg6_mul_data")
            if mul and mul.get("use_mul_jackpot"):
                try:
                    jps = []
                    for wk in winners_keys:
                        m = _re.search(r"(\d+)", str(wk))
                        if m:
                            rn = int(m.group(1))
                            if rn in mul.get("jackpots", {}):
                                jps.append(float(mul["jackpots"][rn]))
                    if jps: jackpot_val = float(sum(jps)/len(jps))
                except Exception:
                    pass

            # Apply semantics only here
            sem = st.session_state.get("bg6_jackpot_semantics", "supp")
            jackpot_effective = float(jackpot_val) if sem == "major" else 0.0
            jackpot_out_amount = float(jackpot_val) if sem != "major" else 0.0

            net = net_pool(gross, refunds_val, jackpot_effective, rules.commission)

            total_units_for_winners = sum(float(spread_units.get(wk, 0.0) or 0.0) for wk in winners_keys)
            if total_units_for_winners <= 0.0:
                used_mul_units = False
                if mul and mul.get("implied_units"):
                    for wk in winners_keys:
                        m = _re.search(r"(\d+)", str(wk))
                        if m:
                            rn = int(m.group(1))
                            iu = float(mul["implied_units"].get(rn, 0.0))
                            if iu > 0:
                                spread_units[wk] = iu
                                used_mul_units = True
                if not used_mul_units:
                    for wk in winners_keys: spread_units[wk] = 1.0
                    st.warning("No units found for last‑leg winners. Assuming 1 unit per winner.", icon="⚠️")
                else:
                    st.info("Filled winners' units from MUL implied units.")

            p = max(0.0, min(100.0, float(consol_pct))) if enable_consol else 0.0
            major_pool = net * (1.0 - p/100.0)
            consol_pool = net * (p/100.0)

            winners_units_used = sum(float(spread_units.get(wk, 0.0) or 0.0) for wk in winners_keys)
            st.info(f"Major Pool: ${major_pool:,.2f}  •  Consolation Pool: ${consol_pool:,.2f}  •  Winners' Units used: {winners_units_used:.4f}  •  Jackpot mode: {sem}  •  Jackpot used in Major: ${jackpot_effective:,.2f}  •  Jackpot out: ${jackpot_out_amount:,.2f}")

            divs_major = dividends_from_spread(major_pool, winners_keys, spread_units, rules, declare_per_winner=True)

            def _is_winner_key(k): return k in winners_keys
            U_total = 0.0; U_win = 0.0
            for k, u in list(spread_units.items()):
                u = float(u or 0.0)
                if u <= 0: continue
                U_total += u
                if _is_winner_key(k): U_win += u
            U_lose = max(0.0, U_total - U_win)

            out_rows = [("Major — " + str(k), float(amt)) for k, amt in divs_major.items()]
            if enable_consol and U_lose > 0.0:
                out_rows.append(("Consolation (5 of 6)", float(consol_pool)/float(U_lose)))

            import pandas as _pd
            df_out = _pd.DataFrame(out_rows, columns=["Declared Dividend", "Amount ($1)"])
            st.subheader("BIG6 Declared Dividends ($1)")
            st.dataframe(df_out, use_container_width=True, hide_index=True)
            st.download_button("Download BIG6 Declared Dividends (CSV)", df_out.to_csv(index=False).encode("utf-8"), "big6_declared_dividends.csv", "text/csv", key='dl_auto_22')

    st.markdown('---')
# ===================== End BIG6 =====================


# === TRI Single‑leg Inputs (added) ===
if pool == 'TRI' and False:  # [dedup disabled duplicate UI block]
    st.subheader('Trifecta (TRI)')
    with st.expander('Single-leg pool inputs', expanded=False):
        colA, colB, colC, colD = st.columns([1,1,1,1])
        with colA:
            if 'tri_gross_sales' in st.session_state:
                _tri_gross_sales = st.number_input('Gross Sales ($)', min_value=0.0, step=100.0, format='%.2f', key='tri_gross_sales')
            else:
                _tri_gross_sales = st.number_input('Gross Sales ($)', value=float(st.session_state.get('gross_sales', 0.0)), min_value=0.0, step=100.0, format='%.2f', key='tri_gross_sales')
        with colB:
            if 'tri_refunds' in st.session_state:
                _tri_refunds = st.number_input('Refunds ($)', min_value=0.0, step=10.0, format='%.2f', key='tri_refunds')
            else:
                _tri_refunds = st.number_input('Refunds ($)', value=float(st.session_state.get('refunds', 0.0)), min_value=0.0, step=10.0, format='%.2f', key='tri_refunds')
        with colC:
            if 'tri_jackpot_in' in st.session_state:
                _tri_jackpot_in = st.number_input('Jackpot In ($)', min_value=0.0, step=10.0, format='%.2f', key='tri_jackpot_in')
            else:
                _tri_jackpot_in = st.number_input('Jackpot In ($)', value=float(st.session_state.get('jackpot_in', 0.0)), min_value=0.0, step=10.0, format='%.2f', key='tri_jackpot_in')
        with colD:
            if 'tri_num_single_leg_winning_units_optional_1' in st.session_state:
                _tri_units = st.number_input('Single-leg Winning Units (optional)', min_value=0.0, step=1.0, format='%.2f', key='tri_num_single_leg_winning_units_optional_1')
            else:
                _tri_units = st.number_input('Single-leg Winning Units (optional)', value=float(st.session_state.get('num_single_leg_winning_units_optional_1', 0.0)), min_value=0.0, step=1.0, format='%.2f', key='tri_num_single_leg_winning_units_optional_1')
        # Keep generic keys in sync (used by calculations elsewhere)
        try:
            st.session_state['gross_sales'] = float(_tri_gross_sales)
            st.session_state['refunds'] = float(_tri_refunds)
            st.session_state['jackpot_in'] = float(_tri_jackpot_in)
            st.session_state['num_single_leg_winning_units_optional_1'] = float(_tri_units)
        except Exception:
            pass
    st.markdown('---')
# === End TRI Single‑leg Inputs (added) ===


# === QAD MLA v3 PATCH BEGIN ===
# Quaddie (QAD) – MLA importer + TX compute using MLA units override
# This block is self‑contained and guarded to avoid collisions elsewhere.
import re as _re

def _qad__to_money(s):
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return 0.0

def qad_units_from_mla_v3(mla_text: str, fourth_runner: int = 1, gross_override: float | None = None, commission_override: float | None = None):
    """
    Parse the '022 MULTI‑LEG APPROXIMATES FOR $1 (QAD)' block and derive units per $1
    for the selected last‑leg runner.
    Returns: dict(units, approx_dividend, approx_jackpot, gross, net, commission).
    """
    t = (mla_text or "").replace("\r", "")
    # Gross (POOL TOTAL)
    m_pool = _re.search(r"(?im)^\s*POOL\s+TOTAL\s+([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2}))\s*$", t)
    gross = _qad__to_money(m_pool.group(1)) if m_pool else 0.0
    if gross_override is not None and gross_override > 0:
        gross = float(gross_override)

    # Commission (Commission Rate Used  20.500000)
    m_comm = _re.search(r"(?im)Commission\s+Rate\s+Used\s+([0-9.]+)", t)
    commission = float(m_comm.group(1))/100.0 if m_comm else 0.205
    if commission_override is not None:
        commission = float(commission_override)

    # Runner dividend/jackpot rows
    rows = []
    for ln in t.splitlines():
        ln = " ".join(ln.strip().split())
        m = _re.match(r"^(\d+)\s+([0-9,]+(?:\.\d+)?)\s+([0-9,]+(?:\.\d+)?)$", ln)
        if m:
            idx = int(m.group(1))
            div = _qad__to_money(m.group(2))
            jp  = _qad__to_money(m.group(3))
            rows.append((idx, div, jp))

    sel = next(((i,d,j) for (i,d,j) in rows if i == int(fourth_runner)), None)
    approx_dividend = float(sel[1]) if sel else 0.0
    approx_jackpot  = float(sel[2]) if sel else 0.0

    net = max(gross, 0.0) * (1.0 - float(commission))
    units = (net / approx_dividend) if (approx_dividend and net > 0) else 0.0
    return {
        "units": round(float(units), 6),
        "approx_dividend": float(approx_dividend),
        "approx_jackpot": float(approx_jackpot),
        "gross": float(gross),
        "net": float(net),
        "commission": float(commission),
    }

# ---- UI wiring (Streamlit) ----
try:
    import streamlit as st
    _HAS_ST_QADV3 = True
except Exception:
    _HAS_ST_QADV3 = False




def _is_qad_pool():
    # Return True only when current pool selection is QAD
    try:
        import streamlit as st  # type: ignore
    except Exception:
        return False
    val = ""
    for k in ("pool","pool_sel","pool_choice","selected_pool","pool_name","pool_select","tab_pool","pool_dropdown"):
        v = st.session_state.get(k)
        if v:
            val = str(v)
            break
    val = (val or "").strip().upper()
    if not val:
        v = st.session_state.get("pool")
        if isinstance(v, (list, tuple)) and v:
            val = str(v[0]).upper()
    return "QAD" in val
if _HAS_ST_QADV3 and _is_qad_pool():
    with st.expander("📥 Import Multi‑Leg Approximates (QAD)", expanded=False):
        st.caption("Paste the **022 MULTI‑LEG APPROXIMATES FOR $1** block. We'll derive **units** for the chosen last‑leg runner and keep it as an override for QAD.")
        mla_text = st.text_area("Paste MLA text", key="qad_mla_text_v3", height=220)
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        with c1:
            fourth_runner = st.number_input("Fourth‑leg runner", min_value=1, max_value=200, value=1, step=1, key="qad_mla_fourth_v3")
        with c2:
            mla_comm = st.number_input("Commission (0–1)", min_value=0.0, max_value=0.99, value=0.205, step=0.001, format="%.3f", key="qad_mla_comm_v3")
        with c3:
            gross_override = st.number_input("Gross override ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f", key="qad_mla_gross_over_v3")
        with c4:
            go = st.button("Use MLA → Set Units", key="btn_qad_use_mla_v3")
        if go and (mla_text or "").strip():
            try:
                info = qad_units_from_mla_v3(
                    mla_text,
                    fourth_runner=int(fourth_runner),
                    gross_override=(gross_override if gross_override > 0 else None),
                    commission_override=float(mla_comm),
                )
                st.session_state["qad_units_override"] = float(info["units"])
                st.session_state["qad_jackpot_out"] = float(info["approx_jackpot"])
                st.session_state["qad_mla_gross"] = float(info["gross"])
                st.session_state["qad_mla_comm"] = float(info["commission"])
                st.success(f"Units set from MLA: {info['units']:.6f} • Gross ${info['gross']:.2f} • Approx ${info['approx_dividend']:.2f} • Jackpot out ${info['approx_jackpot']:.2f}")
            except Exception as e:
                st.error(f"MLA parse failed: {e}")
        chips = []
        if st.session_state.get("qad_units_override", 0.0) > 0:
            chips.append(f"MLA units {st.session_state['qad_units_override']:.6f}")
        if st.session_state.get("qad_mla_gross", 0.0) > 0:
            chips.append(f"MLA gross ${st.session_state['qad_mla_gross']:.2f}")
        if chips:
            st.caption(" • ".join(chips))

    with st.expander("⚙️ Compute QAD (TX + MLA units override)", expanded=False):
        st.caption("Paste the **Transaction Report** (scan sells). We use **MLA units** if set; otherwise fallback to **file percent** or **ticket coverage**.")
        qad_tx_text = st.text_area("Paste/confirm TX text here (same as above)", key="qad_tx_compute_v3", height=160)
        colA, colB, colC = st.columns([1,1,1])
        with colA:
            commission_pct = st.number_input("Commission (%)", min_value=0.0, max_value=99.0, value=20.50, step=0.01, format="%.2f", key="qad_comm_pct_compute_v3")
        with colB:
            break_step = st.number_input("Breakage step ($)", min_value=0.0, max_value=1.0, value=0.10, step=0.01, format="%.2f", key="qad_break_compute_v3")
        with colC:
            dp = st.number_input("Display DP", min_value=0, max_value=4, value=2, step=1, key="qad_dp_compute_v3")

        go2 = st.button("Compute (TX + overrides)", key="btn_qad_compute_mla_v3")
        if go2:
            try:
                # Use existing TX parser (already present in your file for EQD/QAD)
                res = _eqd_parse_transactions(qad_tx_text)
                tickets = res.get("tickets", [])
                gross_tx   = float(res.get("gross", 0.0) or 0.0)
                refunds = float(res.get("refunds", 0.0) or 0.0)
                jackpot_in = float(res.get("jackpot", 0.0) or 0.0)  # default 0

                # Units source preference: MLA override > file percent > ticket coverage
                u_override = float(st.session_state.get("qad_units_override", 0.0) or 0.0)
                if u_override > 0:
                    units_used = u_override
                    units_source = "MLA override"
                else:
                    pct = res.get("percent", None)
                    if pct is not None:
                        units_used = (float(pct)/100.0) if pct > 1.5 else float(pct)
                        units_source = "file percent"
                    else:
                        units_used = _eqd_units_per1((1,1,1,1), tickets)
                        units_source = "ticket coverage"

                # Gross: prefer TX; if missing, fallback to MLA gross saved earlier
                gross = gross_tx if gross_tx > 0 else float(st.session_state.get("qad_mla_gross", 0.0) or 0.0)

                # Net pool = (gross - refunds + jackpot_in) * (1 - commission)
                comm = float(commission_pct)/100.0
                base = max((gross or 0.0) - (refunds or 0.0), 0.0) + max(jackpot_in or 0.0, 0.0)
                net_pool_val = base * (1.0 - comm)

                # Declared with breakage "down"
                from decimal import Decimal
                if units_used > 0 and net_pool_val > 0:
                    val = float(net_pool_val) / float(units_used)
                    cents = Decimal(val) * Decimal(100)
                    step = Decimal(str(break_step)) * Decimal(100)
                    cents = cents - (cents % step)
                    declared = float((cents / Decimal(100)).quantize(Decimal(10) ** -int(dp)))
                else:
                    declared = 0.0

                st.success(f"Declared **${declared:,.2f}** per $1 • Units **{units_used:.6f}** • Payout pool **${net_pool_val:,.2f}** (Gross ${gross:,.2f} • Comm {commission_pct:.2f}% • Refunds ${refunds:,.2f} • Jackpot-in ${jackpot_in:.2f})")
                st.caption(f"Units source: {units_source}")
                jp_out = st.session_state.get("qad_jackpot_out", None)
                if jp_out is not None:
                    st.info(f"Jackpot out: **${float(jp_out):,.2f}** (display-only; not included in dividend)")
            except Exception as e:
                st.error(f"Compute failed: {e}")
# === QAD MLA v3 PATCH END ===



# =========================
# QAD — MLA importer + TX compute (QAD-only UI)
# =========================
try:
    import streamlit as st  # already imported above
except Exception:
    st = None

def _qad_is_active_pool():
    try:
        p = globals().get("pool") or ""
        p = str(p).strip().upper()
        return "QAD" in p
    except Exception:
        return False


def _qad_parse_mla(text: str) -> dict:
    # Robust parser for 022 MULTI‑LEG APPROXIMATES FOR $1 (handles 1- or 2-column rows,
    # 'Scratched' entries, and 'POOL TOTAL' appearing mid-line).
    out = dict(gross=None, commission_used=None, dividends={}, jackpots={})
    if not text:
        return out
    t = text.replace("\r", "").replace("\u00a0"," ")
    # Gross (POOL TOTAL can appear after other words on the line)
    m = re.search(r"(?i)POOL\s+TOTAL\s+([0-9,]+(?:\.\d{2})?)", t)
    if m:
        try:
            out["gross"] = float(m.group(1).replace(",", ""))
        except Exception:
            pass
    # Commission Rate Used
    m = re.search(r"(?i)Commission\s+Rate\s+Used\s+([0-9.]+)", t)
    if m:
        try:
            out["commission_used"] = float(m.group(1)) / 100.0
        except Exception:
            pass
    # Runner rows — accept either two-column lines or single-column lines.
    for line in t.splitlines():
        if "cratched" in line:  # catches 'Scratched'
            continue
        for m in re.finditer(r"(\d{1,2})\s+([0-9,]+(?:\.\d{2})?)\s+([0-9,]+(?:\.\d{2})?)", line):
            try:
                r = int(m.group(1))
                d = float(m.group(2).replace(",",""))
                j = float(m.group(3).replace(",",""))
                out["dividends"][r] = d
                out["jackpots"][r] = j
            except Exception:
                pass
    return out
    t = text.replace("\r", "").replace("\u00a0"," ")
    # Pool total / gross
    m = re.search(r"(?im)^\s*POOL\s+TOTAL\s+([0-9,]+(?:\.\d+)?)\s*$", t)
    if m:
        try:
            out["gross"] = float(m.group(1).replace(",", ""))
        except Exception:
            pass
    # Commission Rate Used
    m = re.search(r"(?i)Commission\s+Rate\s+Used\s+([0-9.]+)", t)
    if m:
        try:
            out["commission_used"] = float(m.group(1)) / 100.0
        except Exception:
            pass
    # Runner rows like: "1   3,180.00   1,080.14" possibly with columns twice per row
    row_rx = re.compile(r"(?m)^(\\s*(\\d{1,2})\\s+([0-9,]+(?:\\.\\d+)?)\\s+([0-9,]+(?:\\.\\d+)?))(?:\\s+(\\d{1,2})\\s+([0-9,]+(?:\\.\\d+)?)\\s+([0-9,]+(?:\\.\\d+)?))?\\s*$")
    for m in row_rx.finditer(t):
        try:
            r1 = int(m.group(2)); d1 = float(m.group(3).replace(",","")); j1 = float(m.group(4).replace(",",""))
            out["dividends"][r1] = d1
            out["jackpots"][r1] = j1
        except Exception:
            pass
        try:
            if m.group(5):
                r2 = int(m.group(5)); d2 = float(m.group(6).replace(",","")); j2 = float(m.group(7).replace(",",""))
                out["dividends"][r2] = d2
                out["jackpots"][r2] = j2
        except Exception:
            pass
    return out

def _qad_units_from_mla(text: str, fourth_leg_runner: int, commission_fallback: float):
    # Compute units override from MLA block.
    # Returns (units, gross, jackpot_out_for_runner)
    d = _qad_parse_mla(text or "")
    gross = float(d.get("gross") or 0.0)
    used_comm = d.get("commission_used")
    commission = used_comm if (used_comm is not None) else float(commission_fallback or 0.0)
    dividends = d.get("dividends") or {}
    jackpots = d.get("jackpots") or {}
    div = float(dividends.get(int(fourth_leg_runner), 0.0))
    jp = float(jackpots.get(int(fourth_leg_runner), 0.0))
    if gross <= 0 or div <= 0:
        return (0.0, gross, jp)
    net = gross * (1.0 - max(commission, 0.0))
    units = net / div
    return (units, gross, jp)

def _qad_declared_from_tx_with_units(tx_text: str, units: float, rules):
    # Compute declared using TX dump for gross/refunds/jackpot, but with units override (e.g., from MLA).
    # Returns (declared_dividend, payout_pool_net)
    if not tx_text or units <= 0:
        return (0.0, 0.0)
    t = tx_text.replace("\r","")
    # Gross (POOL TOTALS ... QAD $ X) or SUB TOTALS ... SELLS $ X
    m = (re.search(r"(?is)POOL\\s+TOTALS\\s*:.*?QAD\\s*\\$\\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\\.\\d{2}))", t)
         or re.search(r"(?is)SUB\\s+TOTALS\\s*:.*?SELLS\\s*\\$\\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\\.\\d{2}))", t))
    gross = float(m.group(1).replace(",","")) if m else 0.0
    # Refunds
    if gross <= 0.0:
        try:
            gross = float(gross_override_ui or 0.0)
        except Exception:
            gross = 0.0
    # Refunds (optional)
    m = re.search(r"(?im)^\\s*REFUNDS?\\s*\\$\\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\\.\\d{2}))", t)
    refunds = float(m.group(1).replace(",","")) if m else 0.0
    # Jackpot in (optional)
    m = re.search(r"(?im)^\\s*JACKPOT\\s*IN\\s*\\$\\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\\.\\d{2}))", t)
    jp_in = float(m.group(1).replace(",","")) if m else 0.0
    # Commission from sidebar rules
    commission = float(getattr(rules, "commission", 0.205))
    base = max(gross + jp_in - refunds, 0.0)
    net = base * (1.0 - commission)
    # Declared with breakage/min-div
    div = 0.0 if units <= 0 or net <= 0 else (net / float(units))
    # Use helpers from main app if available
    try:
        div = _apply_breakage(div, rules.break_step, rules.break_mode)
        div = max(div, rules.min_div)
        disp = _format_display(div, rules.display_dp)
    except Exception:
        disp = f"{div:.2f}"
    return (disp, net)

if st is not None and _qad_is_active_pool():
    with st.expander("📥 Import Multi‑Leg Approximates (QAD)"):
        st.markdown("Paste the **022 MULTI‑LEG APPROXIMATES FOR $1** block.")
        mla_txt = st.text_area("Paste MLA text", value=st.session_state.get("qad_mla_text",""), height=220, key="qad_mla_text")
        runner = st.number_input("Fourth‑leg runner", min_value=1, max_value=99, value=int(st.session_state.get("qad_mla_runner", 1)))
        mla_comm = float(getattr(globals().get("rules", None), "commission", 0.205))
        st.caption(f"Using commission fallback = {mla_comm:.3f} unless 'Commission Rate Used' present in the block.")
        if st.button("Use MLA → Set Units", key="btn_qad_use_mla"):
            units, gross, jp = _qad_units_from_mla(mla_txt, runner, mla_comm)
            st.session_state["qad_mla_units"] = float(units or 0.0)
            st.session_state["qad_mla_jackpot_out"] = float(jp or 0.0)
            st.session_state["qad_mla_runner"] = int(runner)
            st.session_state["qad_mla_gross"] = float(gross or 0.0)
            if units > 0:
                st.success(f"MLA set: Units override = {units:.6f} per $1. (Gross={gross:,.2f}; Jackpot-out display={jp:,.2f})")
            else:
                st.error("Could not derive units from the MLA block (check Pool Total and Dividend rows).")

    with st.expander("⚙️ Compute QAD (TX + MLA units override)"):
        st.markdown("Paste the **Transaction Report** (TX) block. Uses MLA units if set; otherwise falls back to file percent.")
        tx_txt = st.text_area("Paste/confirm TX text here (same as above)", value=st.session_state.get("qad_tx_text", ""), height=180, key="qad_tx_text")
        col1, col2, col3 = st.columns(3)
        with col1:
            comm_ui = st.number_input("Commission (%)", min_value=0.0, max_value=99.0, step=0.01,
                                      value=round(float(getattr(globals().get("rules", None), "commission", 0.205)) * 100.0, 2),
                                      key="qad_comm_percent_ui")
        with col2:
            break_step_ui = st.number_input("Breakage step ($)", value=float(getattr(globals().get("rules", None), "break_step", 0.10)), step=0.01, format="%.2f")
        with col3:
            dp_ui = st.number_input("Display DP", min_value=0, max_value=4, step=1, value=int(getattr(globals().get("rules", None), "display_dp", 2)))
        # Optional gross override; default to MLA gross if we have it
        gross_default = float(st.session_state.get("qad_mla_gross", 0.0) or 0.0)
        gross_override_ui = st.number_input("Gross override ($)", min_value=0.0, value=gross_default, step=0.01, format="%.2f")
        do_compute = st.button("Compute (TX + overrides)", key="btn_qad_compute_tx_override")
        if do_compute:
            # derive units: prefer MLA; else TX percent per comb
            u = float(st.session_state.get("qad_mla_units", 0.0) or 0.0)
            if u <= 0.0:
                m = re.search(r"Percent\s+per\s+comb\.?\s*([0-9.]+)\s*%", tx_txt or "", re.I)
                if m:
                    try:
                        u = float(m.group(1)) / 100.0
                    except Exception:
                        u = 0.0
            class PoolRules:
                def __init__(self, commission, min_div, break_step, break_mode, max_dividends, display_dp):
                    self.commission = commission
                    self.min_div = min_div
                    self.break_step = break_step
                    self.break_mode = break_mode
                    self.max_dividends = max_dividends
                    self.display_dp = display_dp
            local_rules = PoolRules(
                commission=float(comm_ui)/100.0,
                min_div=getattr(globals().get("rules", None), "min_div", 1.04),
                break_step=float(break_step_ui),
                break_mode=getattr(globals().get("rules", None), "break_mode", "down"),
                max_dividends=8,
                display_dp=int(dp_ui),
            )
            declared, net = _qad_declared_from_tx_with_units(tx_txt, u, local_rules)
            st.success(f"**{declared} × *per1*** • Units {u:.6f} • Payout pool **{net:,.2f}** • Jackpot-out (display only): **{float(st.session_state.get('qad_mla_jackpot_out',0.0)):.2f}**")


# === EQD MLA v3 PATCH BEGIN ===
# Early Quaddie (EQD) – MLA importer + TX compute using MLA units override
# Mirrors the QAD flow but scoped to EQD with unique widget keys.

try:
    import streamlit as st  # type: ignore
    _HAS_ST_EQDV3 = True
except Exception:
    _HAS_ST_EQDV3 = False

def _is_eqd_pool() -> bool:
    try:
        import streamlit as st
    except Exception:
        # Fallback to globals only
        v0 = str(globals().get("pool", "")).upper()
        return ("EQD" in v0) or ("EARLY" in v0 and "QUAD" in v0)

    ss = getattr(st, "session_state", {})
    cand = []

    # Add common keys used across builds
    for k in ("pool","pool_sel","pool_choice","selected_pool","pool_name","pool_select",
              "tab_pool","pool_dropdown","pool_type","pool_display","selected_pool_name"):
        try:
            val = ss.get(k)
            if isinstance(val, (list, tuple)):
                cand.extend([str(x) for x in val])
            elif val is not None:
                cand.append(str(val))
        except Exception:
            pass

    # Also include any global variable named `pool`
    try:
        gpool = globals().get("pool", "")
        if gpool:
            cand.append(str(gpool))
    except Exception:
        pass

    val = " ".join(cand).upper()
    return ("EQD" in val) or ("EARLY" in val and "QUAD" in val)


def _eqd__to_money(s):
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return 0.0

def eqd_units_from_mla_v3(mla_text: str, fourth_runner: int = 1, gross_override: float | None = None, commission_override: float | None = None):
    """
    Parse the '022 MULTI‑LEG APPROXIMATES FOR $1 (EQD)' block and derive units per $1
    for the selected last‑leg runner.
    Returns: dict(units, approx_dividend, approx_jackpot, gross, net, commission).
    """
    t = (mla_text or "").replace("\r", "")
    # Gross (POOL TOTAL) - tolerate extra content on the line
    m_pool = re.search(r"(?im)POOL\s+TOTAL\s+([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2}))", t)
    gross = _eqd__to_money(m_pool.group(1)) if m_pool else 0.0
    if gross_override is not None and gross_override > 0:
        gross = float(gross_override)

    # Commission (Commission Rate Used  20.500000)
    m_comm = re.search(r"(?im)Commission\s+Rate\s+Used\s+([0-9.]+)", t)
    commission = float(m_comm.group(1))/100.0 if m_comm else 0.205
    if commission_override is not None:
        commission = float(commission_override)

    # Runner dividend/jackpot rows — handle two columns per line
    rows = []
    for ln in t.splitlines():
        ln = " ".join(ln.strip().split())
        for m in re.finditer(r"(\d+)\s+([0-9,]+(?:\.\d+)?)\s+([0-9,]+(?:\.\d+)?)", ln):
            idx = int(m.group(1))
            div = _eqd__to_money(m.group(2))
            jp  = _eqd__to_money(m.group(3))
            rows.append((idx, div, jp))

    sel = next(((i,d,j) for (i,d,j) in rows if i == int(fourth_runner)), None)
    approx_dividend = float(sel[1]) if sel else 0.0
    approx_jackpot  = float(sel[2]) if sel else 0.0

    net = max(gross, 0.0) * (1.0 - float(commission))
    units = (net / approx_dividend) if (approx_dividend and net > 0) else 0.0
    return {
        "units": round(float(units), 6),
        "approx_dividend": float(approx_dividend),
        "approx_jackpot": float(approx_jackpot),
        "gross": float(gross),
        "net": float(net),
        "commission": float(commission),
    }

# Gate relaxed to avoid order/session issues

__gate_eqd = False

try:

    __gate_eqd = _is_eqd_pool()

except Exception:

    __gate_eqd = False

if _HAS_ST_EQDV3 and _is_eqd_pool():
    with st.expander("📥 Import Multi‑Leg Approximates (EQD)", expanded=False):
        st.markdown("Paste the **022 MULTI‑LEG APPROXIMATES FOR $1** block. We'll derive **units** for the chosen last‑leg runner and keep it as an override for EQD.")
        eqd_mla_text = st.text_area("Paste MLA text", key="eqd_mla_text_v3", height=220)
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        with c1:
            eqd_fourth_runner = st.number_input("Fourth‑leg runner", min_value=1, max_value=200, value=1, step=1, key="eqd_mla_fourth_v3")
        with c2:
            eqd_mla_comm = st.number_input("Commission (0–1)", min_value=0.0, max_value=0.99, value=0.205, step=0.001, format="%.3f", key="eqd_mla_comm_v3")
        with c3:
            eqd_gross_override = st.number_input("Gross override ($)", min_value=0.0, value=0.0, step=0.01, format="%.2f", key="eqd_mla_gross_over_v3")
        with c4:
            go_eqd = st.button("Use MLA → Set Units", key="btn_eqd_use_mla_v3")
        if go_eqd and (eqd_mla_text or "").strip():
            try:
                info = eqd_units_from_mla_v3(
                    eqd_mla_text,
                    fourth_runner=int(eqd_fourth_runner),
                    gross_override=(eqd_gross_override if eqd_gross_override > 0 else None),
                    commission_override=float(eqd_mla_comm),
                )
                st.session_state["eqd_units_override"] = float(info["units"])
                st.session_state["eqd_jackpot_out"] = float(info["approx_jackpot"])
                st.session_state["eqd_mla_gross"] = float(info["gross"])
                st.session_state["eqd_mla_comm"] = float(info["commission"])
                st.success(f"MLA set: Units override = {info['units']:.6f} per $1. (Gross={info['gross']:,.2f}; Jackpot‑out display={info['approx_jackpot']:,.2f})")
            except Exception as e:
                st.error(f"MLA parse failed: {e}")
        # Chips
        eqd_chips = []
        if st.session_state.get("eqd_units_override", 0.0) > 0:
            eqd_chips.append(f"MLA units {st.session_state['eqd_units_override']:.6f}")
        if st.session_state.get("eqd_mla_gross", 0.0) > 0:
            eqd_chips.append(f"MLA gross ${st.session_state['eqd_mla_gross']:.2f}")
        if eqd_chips:
            st.caption(" • ".join(eqd_chips))

    with st.expander("⚙️ Compute EQD (TX + MLA units override)", expanded=False):
        st.caption("Paste the **Transaction Report** (scan sells). We use **MLA units** if set; otherwise fallback to **file percent** or **ticket coverage**.")
        eqd_tx_text = st.text_area("Paste/confirm TX text here (same as above)", key="eqd_tx_compute_v3", height=160)
        colA, colB, colC = st.columns([1,1,1])
        with colA:
            eqd_commission_pct = st.number_input("Commission (%)", min_value=0.0, max_value=99.0, value=20.50, step=0.01, format="%.2f", key="eqd_comm_pct_compute_v3")
        with colB:
            eqd_break_step = st.number_input("Breakage step ($)", min_value=0.0, max_value=1.0, value=0.10, step=0.01, format="%.2f", key="eqd_break_compute_v3")
        with colC:
            eqd_dp = st.number_input("Display DP", min_value=0, max_value=4, value=2, step=1, key="eqd_dp_compute_v3")

        go2_eqd = st.button("Compute (TX + overrides)", key="btn_eqd_compute_mla_v3")
        if go2_eqd:
            try:
                # If available elsewhere in the GOLD, we reuse; otherwise, guard.
                res = _eqd_parse_transactions(eqd_tx_text) if '_eqd_parse_transactions' in globals() else {}
                tickets = res.get("tickets", [])
                gross_tx   = float(res.get("gross", 0.0) or 0.0)
                refunds = float(res.get("refunds", 0.0) or 0.0)
                jackpot_in = float(res.get("jackpot", 0.0) or 0.0)  # default 0

                # Units source preference: MLA override > file percent > ticket coverage
                u_override = float(st.session_state.get("eqd_units_override", 0.0) or 0.0)
                if u_override > 0:
                    units_used = u_override
                    units_source = "MLA override"
                else:
                    pct = res.get("percent", None)
                    if pct is not None:
                        units_used = (float(pct)/100.0) if pct > 1.5 else float(pct)
                        units_source = "file percent"
                    else:
                        # Ticket coverage requires a helper similar to QAD's; if absent, fallback to 0
                        units_used = 0.0
                        units_source = "ticket coverage"

                # Gross: prefer TX; if missing, fallback to MLA gross saved earlier
                gross = gross_tx if gross_tx > 0 else float(st.session_state.get("eqd_mla_gross", 0.0) or 0.0)

                # Net pool = (gross - refunds + jackpot_in) * (1 - commission)
                comm = float(eqd_commission_pct)/100.0
                base = max((gross or 0.0) - (refunds or 0.0), 0.0) + max(jackpot_in or 0.0, 0.0)
                net_pool_val = base * (1.0 - comm)

                # Declared with simple break-down rule (down) using Decimal to avoid FP artefacts
                from decimal import Decimal
                if units_used > 0 and net_pool_val > 0:
                    val = float(net_pool_val) / float(units_used)
                    cents = Decimal(val) * Decimal(100)
                    step = Decimal(str(eqd_break_step)) * Decimal(100)
                    cents = cents - (cents % step)
                    declared = float((cents / Decimal(100)).quantize(Decimal(10) ** -int(eqd_dp)))
                else:
                    declared = 0.0

                st.success(f"Declared {declared:.{eqd_dp}f} * *per1 • Units {units_used:.6f} ({units_source}) • Net ${net_pool_val:,.2f} • (Gross ${gross:,.2f} • Commission {eqd_commission_pct:.2f}% • Refunds ${refunds:,.2f} • Jackpot − in ${jackpot_in:,.2f})")
            except Exception as e:
                st.error(f"EQD compute failed: {e}")

# === EQD MLA v3 PATCH END ===