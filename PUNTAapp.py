import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

PUNTA_URL = "http://10.13.236.57/api/renderer/report/byEnvironment/Zambeze/standard"
HEADERS = {"Host": "punta.corpad.net.local"}

r = requests.get(PUNTA_URL, headers=HEADERS, timeout=30)
r.raise_for_status()
html = r.text


st.set_page_config(page_title="Punta – App Versions", layout="wide")

st.title("Punta – App Versions (Zambeze / standard)")
st.caption("Fields: Environment, App Name, Version, Last change")

# --- Data fetch + parse (cached) ---
@st.cache_data(ttl=30, show_spinner=False)
def fetch_and_parse() -> pd.DataFrame:
    r = requests.get(PUNTA_URL, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table")
    if not table:
        return pd.DataFrame(columns=["Environment", "App Name", "Version", "Last change"])

    headers = [th.get_text(strip=True) for th in table.find_all("th")]

    def idx(name: str):
        try:
            return headers.index(name)
        except ValueError:
            return None

    i_env = idx("Environment")
    i_app = idx("App Name")
    i_ver = idx("Version")
    i_last = idx("Last change")

    if None in (i_env, i_app, i_ver, i_last):
        return pd.DataFrame(columns=["Environment", "App Name", "Version", "Last change"])

    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        cols = [td.get_text(" ", strip=True) for td in tds]
        if max(i_env, i_app, i_ver, i_last) >= len(cols):
            continue

        rows.append({
            "Environment": cols[i_env],
            "App Name": cols[i_app],
            "Version": cols[i_ver],
            "Last change": cols[i_last],
        })

    df = pd.DataFrame(rows)
    return df


# --- Controls ---
left, right = st.columns([3, 1])
with left:
    q = st.text_input("Filter by App Name", placeholder="e.g. betty, fossil, foo")
with right:
    if st.button("Refresh now"):
        fetch_and_parse.clear()
        st.rerun()

# --- Load data ---
try:
    df = fetch_and_parse()
except Exception as e:
    st.error(f"Failed to fetch/parse Punta page: {e}")
    st.stop()

# --- Filter ---
if q.strip():
    df = df[df["App Name"].str.contains(q.strip(), case=False, na=False)]

# --- Display ---
st.write(f"Rows: {len(df)}")
st.dataframe(df.sort_values(["App Name", "Environment"]), use_container_width=True, hide_index=True)

# --- Export ---
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv_bytes, file_name="punta_apps.csv", mime="text/csv")
