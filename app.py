# app.py — Google Drive CSV (>100MB) dengan downloader kuat + cache + fix low_memory/pyarrow

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches

# ============================ CONFIG ============================
st.set_page_config(
    page_title="Analyst Indonesia Investment Realization With Modelling SARIMA",
    layout="wide",
)

# >>> Ganti ke FILE_ID CSV-mu (Drive harus Anyone with the link)
FILE_ID = "1auW1x7VA_e2PWVoBA8klWdMDgRAEx8zK"

# Nama file lokal; dukung .csv atau .csv.gz
LOCAL_FILE = Path("data.csv")  # kalau file-mu .csv.gz, ubah ke Path("data.csv.gz")
CACHE_TTL  = 24 * 60 * 60  # 1 hari

# ============================ UTILS ============================
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(" ", "_", regex=False)
          .str.replace("-", "_", regex=False)
    )
    return df.drop_duplicates()

def _file_fingerprint(p: Path) -> Tuple[int, int]:
    s = p.stat()
    return (int(s.st_mtime_ns), int(s.st_size))

def _looks_like_csv(p: Path) -> bool:
    """
    Heuristik: file ada, > 0 byte, bukan HTML, dan ada delimiter umum.
    Untuk .csv.gz kita terima berdasar ekstensi.
    """
    try:
        if not p.exists() or p.stat().st_size < 8:
            return False
        if p.suffix.lower() == ".gz" and p.with_suffix("").suffix.lower() == ".csv":
            return True
        head = p.read_text(errors="ignore")[:800].lower()
        if "<html" in head or "google drive" in head or "download_warning" in head:
            return False
        return ("," in head) or (";" in head) or ("\t" in head)
    except Exception:
        return False

def _download_from_drive(file_id: str, dest: Path) -> None:
    """
    Unduh dari Google Drive; tangani file besar (confirm page). Simpan ke 'dest'.
    """
    import gdown, requests, re

    dest.unlink(missing_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"

    # 1) Coba gdown (dengan cookies & progress) — handle confirm token otomatis
    out = gdown.download(url=url, output=str(dest), quiet=False, use_cookies=True, fuzzy=True)
    if out and Path(out).exists() and _looks_like_csv(Path(out)):
        return

    # 2) Fallback manual pakai requests (handle confirm token)
    SESSION = requests.Session()
    r = SESSION.get(url, timeout=60)
    if r.headers.get("content-disposition"):
        with open(dest, "wb") as f:
            f.write(r.content)
        if _looks_like_csv(dest):
            return

    # Cari link dengan confirm token di HTML
    m = re.search(r'href="(/uc\?export=download[^"]+confirm=[^"&]+[^"]+)"', r.text)
    if not m:
        try:
            dest_html = dest.with_suffix(".html")
            dest_html.write_text(r.text, encoding="utf-8", errors="ignore")
        except Exception:
            pass
        raise RuntimeError("Google Drive meminta konfirmasi & token tidak ditemukan. Pastikan file publik.")
    confirm_url = "https://drive.google.com" + m.group(1).replace("&amp;", "&")

    with SESSION.get(confirm_url, stream=True, timeout=60) as rr, open(dest, "wb") as f:
        for chunk in rr.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)

    if not _looks_like_csv(dest):
        try:
            txt = dest.read_text(errors="ignore")[:400].lower()
            if "<html" in txt:
                dest.rename(dest.with_suffix(".html"))
        except Exception:
            pass
        raise RuntimeError("File yang terunduh tidak terlihat seperti CSV. Cek izin & format di Drive.")

def _prepare_local_file(file_id: str, force_download: bool = False) -> Path:
    """
    Pastikan data CSV tersedia.
    - Jika force_download=True, selalu ambil ulang dari Drive.
    - Jika sudah ada & 'kelihatan CSV', langsung pakai.
    """
    if force_download or not LOCAL_FILE.exists():
        with st.spinner(f"Mengunduh {LOCAL_FILE.name} dari Google Drive…"):
            _download_from_drive(file_id, LOCAL_FILE)
    else:
        if not _looks_like_csv(LOCAL_FILE):
            with st.spinner("File lokal tidak valid, unduh ulang dari Google Drive…"):
                _download_from_drive(file_id, LOCAL_FILE)
    return LOCAL_FILE

@st.cache_data(show_spinner=True, ttl=CACHE_TTL)
def load_table(path_str: str, fingerprint: Optional[Tuple[int, int]]) -> pd.DataFrame:
    """
    Loader tabel generik: CSV/CSV.GZ (prioritas), Parquet (opsional).
    - Jika pyarrow tersedia: gunakan engine 'pyarrow' (tanpa low_memory).
    - Jika tidak: gunakan parser default (low_memory=False).
    """
    p = Path(path_str)
    if not p.exists():
        st.error(f"File tidak ditemukan: {p.resolve()}")
        return pd.DataFrame()

    ext = p.suffix.lower()
    try:
        if ext in {".csv", ".gz"}:
            use_pyarrow = False
            try:
                import pyarrow  # noqa
                use_pyarrow = True
            except Exception:
                use_pyarrow = False

            if use_pyarrow:
                df = pd.read_csv(p, engine="pyarrow")
                # kalau mau dtype Arrow:
                # df = pd.read_csv(p, engine="pyarrow", dtype_backend="pyarrow")
            else:
                df = pd.read_csv(p, low_memory=False)

        elif ext in {".parquet", ".pq"}:
            import pyarrow  # pastikan ada
            df = pd.read_parquet(p)

        else:
            # fallback: coba CSV heuristik
            if _looks_like_csv(p):
                use_pyarrow = False
                try:
                    import pyarrow  # noqa
                    use_pyarrow = True
                except Exception:
                    pass
                if use_pyarrow:
                    df = pd.read_csv(p, engine="pyarrow")
                else:
                    df = pd.read_csv(p, low_memory=False)
            else:
                st.error(f"Ekstensi {ext} belum didukung dan file tidak terdeteksi sebagai CSV.")
                return pd.DataFrame()

    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return pd.DataFrame()

    return _normalize_cols(df)


# ============================ LOAD DATA ============================
data_path = _prepare_local_file(FILE_ID, force_download=False)
fingerprint = _file_fingerprint(Path(data_path)) if Path(data_path).exists() else None
df = load_table(str(data_path), fingerprint)

if df.empty:
    st.stop()

# ============================ DETEKSI KOLOM & UTIL ============================
def find_col(candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

COL_YEAR   = find_col(["year", "tahun"])
COL_QTR    = find_col(["quarter", "kuartal", "quartal", "q"])
COL_CNTRY  = find_col(["country", "negara"])
COL_REGION = find_col(["region", "wilayah", "area"])
COL_SECTOR = find_col(["main_sector", "mainsector", "main_category", "kategori", "category", "sector", "sektor"])
COL_VALUE  = find_col(["investment_idr_million", "nilai_investasi_idr", "amount_idr_million", "nilai", "pendapatan"])
COL_VALUE_USD = find_col(["investment_usd_thousand", "nilai_investasi_usd", "amount_usd_thousand"])

def safe_sum(df_in: pd.DataFrame, col: Optional[str]) -> float:
    if col and col in df_in.columns:
        return pd.to_numeric(df_in[col].astype(str).str.replace(",", ""), errors="coerce").sum()
    return np.nan

# ============================ SIDEBAR FILTERS ============================
with st.sidebar:
    st.subheader("Filter Dashboard")

    if COL_YEAR:
        year_opts = sorted([x for x in df[COL_YEAR].dropna().unique()])
        selected_years = st.multiselect("Year", options=year_opts, default=[], placeholder="Pilih satu/lebih tahun")
    else:
        selected_years = []
        st.warning("Kolom 'year' tidak ditemukan.")

    if COL_CNTRY:
        country_opts = sorted(df[COL_CNTRY].dropna().unique())
        selected_country = st.multiselect("Country", options=country_opts, default=[])
    else:
        selected_country = []
        st.warning("Kolom 'country' tidak ditemukan.")

    if COL_REGION:
        region_opts = sorted(df[COL_REGION].dropna().unique())
        selected_region = st.multiselect("Region", options=region_opts, default=[])
    else:
        selected_region = []
        st.warning("Kolom 'region' tidak ditemukan.")

    if COL_SECTOR:
        sector_opts = sorted(df[COL_SECTOR].dropna().unique())
        selected_sector = st.multiselect("Main Sector", options=sector_opts, default=[])
    else:
        selected_sector = []
        st.warning("Kolom 'main_sector'/'sector' tidak ditemukan.")

# Terapkan filter
df_f = df.copy()
if COL_YEAR and selected_years:
    df_f = df_f[df_f[COL_YEAR].isin(selected_years)]
if COL_CNTRY and selected_country:
    df_f = df_f[df_f[COL_CNTRY].isin(selected_country)]
if COL_REGION and selected_region:
    df_f = df_f[df_f[COL_REGION].isin(selected_region)]
if COL_SECTOR and selected_sector:
    df_f = df_f[df_f[COL_SECTOR].isin(selected_sector)]

# ============================ HEADER & KPI ============================
st.title("Dashboard Analyst Indonesia Investment Realization 2010–2025 (SARIMA)")

st.markdown("""
<style>
.card {
    background-color: #d4f7d4;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 1px 1px 6px rgba(0,0,0,0.1);
}
.card h3 { font-size: 16px; color: #333; margin-bottom: 8px; }
.card p  { font-size: 24px; font-weight: bold; color: #000; margin: 0; }
</style>
""", unsafe_allow_html=True)

col_pad_l, col_kpi1, col_kpi2, col_kpi3, col_pad_r = st.columns([0.5, 4, 4, 4, 0.5])

with col_kpi1:
    total_idr = safe_sum(df_f, COL_VALUE)
    st.markdown(f"""
        <div class="card">
            <h3>Total Invest (IDR Million)</h3>
            <p>{(0 if np.isnan(total_idr) else total_idr):,.2f}</p>
        </div>
    """, unsafe_allow_html=True)

with col_kpi2:
    total_usd = safe_sum(df_f, COL_VALUE_USD)
    st.markdown(f"""
        <div class="card">
            <h3>Total Invest (USD Thousand)</h3>
            <p>{(0 if np.isnan(total_usd) else total_usd):,.2f}</p>
        </div>
    """, unsafe_allow_html=True)

with col_kpi3:
    total_records = len(df_f)
    st.markdown(f"""
        <div class="card">
            <h3>Jumlah Proyek / Records</h3>
            <p>{total_records:,}</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================ TABEL DATA ============================
st.subheader("Data Investment Indonesia 2010–2025")
st.dataframe(df_f, use_container_width=True)
st.markdown("---")

# ============================ CHARTS: REGION & SECTOR ============================
tab_reg, tab_sec = st.tabs(["Total per Region", "Distribusi per Main Sector"])

with tab_reg:
    st.subheader("Total per Region")
    if COL_REGION and not df_f.empty:
        if COL_VALUE and COL_VALUE in df_f.columns:
            g = df_f.copy()
            g[COL_VALUE] = pd.to_numeric(g[COL_VALUE].astype(str).str.replace(",", ""), errors="coerce")
            g = g.groupby(COL_REGION, as_index=False)[COL_VALUE].sum()
            fig = px.bar(g, x=COL_REGION, y=COL_VALUE)
        else:
            g = df_f[COL_REGION].value_counts().reset_index()
            g.columns = [COL_REGION, "count"]
            fig = px.bar(g, x=COL_REGION, y="count")
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Data/kolom region belum tersedia.")

with tab_sec:
    st.subheader("Distribusi per Main Sector")
    if COL_SECTOR and not df_f.empty:
        counts = df_f[COL_SECTOR].value_counts()
        topn = 10
        counts_top = counts.nlargest(topn)
        if len(counts) > topn:
            counts_top.loc["Others"] = counts.iloc[topn:].sum()
        pie_df = counts_top.reset_index()
        pie_df.columns = [COL_SECTOR, "count"]
        fig2 = px.pie(pie_df, names=COL_SECTOR, values="count", hole=0.3)
        fig2.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=30))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Data/kolom main sector belum tersedia.")

st.markdown("---")

# ============================ SERI KUARTALAN & SARIMA ============================
@st.cache_data(ttl=CACHE_TTL)
def build_quarter_series_cached(df_src: pd.DataFrame,
                                col_year: Optional[str],
                                col_qtr: Optional[str],
                                col_value: Optional[str]) -> Optional[pd.DataFrame]:
    if not (col_year and col_qtr and col_value):
        return None

    _qmap = {"1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4",
             "Q1": "Q1", "Q2": "Q2", "Q3": "Q3", "Q4": "Q4"}

    tmp = df_src.copy()
    tmp[col_qtr] = tmp[col_qtr].astype(str).str.strip().str.upper().map(_qmap)
    tmp[col_value] = pd.to_numeric(tmp[col_value].astype(str).str.replace(",", ""), errors="coerce")
    tmp[col_year]  = pd.to_numeric(tmp[col_year], errors="coerce").astype("Int64")

    tmp = tmp.dropna(subset=[col_year, col_qtr, col_value])
    tmp = tmp[(tmp[col_year] > 0) & (tmp[col_qtr].isin(["Q1","Q2","Q3","Q4"]))]

    if tmp.empty:
        return None

    tmp["period"] = tmp[col_year].astype(int).astype(str) + "-" + tmp[col_qtr]
    g = (
        tmp.groupby("period", as_index=False)[col_value].sum()
          .drop_duplicates(subset=["period"])
          .sort_values("period")
    )

    dt_idx = pd.PeriodIndex(g["period"], freq="Q-DEC").to_timestamp(how="end")
    g = g.set_index(dt_idx).sort_index()
    g.index.name = "date"
    g = g.rename(columns={col_value: "investment_idr_million"})

    ts = g["investment_idr_million"].asfreq("Q")
    g["investment_idr_million"] = ts.interpolate(limit_direction="both")
    return g

df_group = build_quarter_series_cached(df, COL_YEAR, COL_QTR, COL_VALUE)

tab_trend, tab_model = st.tabs(["Trend Analysis Investasi 2010–2025", "Baseline SARIMA + Evaluasi"])

with tab_trend:
    st.subheader("Trend Analysis Investasi 2010–2025")
    if df_group is None or df_group.empty:
        st.info("Tidak bisa membentuk seri kuartalan (butuh kolom year, quarter, dan nilai investasi).")
    else:
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df_group.index, df_group['investment_idr_million'], marker='o', linewidth=2)
        ax1.set_title("Tren Investasi Kuartalan di Indonesia (2010–2025)", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Tahun", fontsize=12)
        ax1.set_ylabel("Nilai Investasi (Juta Rupiah)", fontsize=12)
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.grid(True, linestyle='--', alpha=0.7)
        fig1.tight_layout()
        st.pyplot(fig1)

with tab_model:
    st.subheader("Forecast SARIMA (Best AIC)")
    if df_group is None or df_group.empty:
        st.info("Data kuartalan belum tersedia, tidak bisa menjalankan SARIMA.")
    else:
        import statsmodels.api as sm  # lazy import

        ts = df_group["investment_idr_million"]
        if ts.isna().all() or len(ts.dropna()) < 8:
            st.warning("Data kuartalan terlalu pendek untuk SARIMA (min 8 titik). Tampilkan tren saja di tab sebelumnya.")
        else:
            TEST_STEPS = 4 if len(ts) < 16 else min(8, max(1, len(ts)//4))
            y_train = ts.iloc[:-TEST_STEPS]
            y_test  = ts.iloc[-TEST_STEPS:]
            st.caption(f"Train n={len(y_train)} | Test n={len(y_test)} (TEST_STEPS={TEST_STEPS})")

            p = d = q = range(0, 2)
            P = D = Q = range(0, 2)
            s = 4

            best_aic = np.inf
            best_order = best_seasonal = None
            best_res = None

            for pi in p:
                for di in d:
                    for qi in q:
                        for Pi in P:
                            for Di in D:
                                for Qi in Q:
                                    try:
                                        model = sm.tsa.statespace.SARIMAX(
                                            y_train,
                                            order=(pi, di, qi),
                                            seasonal_order=(Pi, Di, Qi, s),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False,
                                        )
                                        res = model.fit(disp=False)
                                        if res.aic < best_aic:
                                            best_aic = res.aic
                                            best_order = (pi, di, qi)
                                            best_seasonal = (Pi, Di, Qi, s)
                                            best_res = res
                                    except Exception:
                                        continue

            if best_res is None:
                st.warning("Tidak ada kombinasi SARIMA yang valid ditemukan. Coba periksa data atau perluas grid.")
            else:
                fc = best_res.get_forecast(steps=len(y_test))
                best_yhat = fc.predicted_mean
                ci = fc.conf_int()
                conf_low = ci.iloc[:, 0].reindex(best_yhat.index)
                conf_high = ci.iloc[:, 1].reindex(best_yhat.index)
                y_test_aligned = y_test.reindex(best_yhat.index)

                fig2, ax2 = plt.subplots(figsize=(12, 6))
                ax2.plot(y_train, label="Train", marker="o", linewidth=2)
                ax2.plot(y_test_aligned, label="Actual (Test)", marker="o", linewidth=2)
                ax2.plot(best_yhat.index, best_yhat.values, label="Forecast (Best AIC)", marker="o", linewidth=2)
                ax2.fill_between(best_yhat.index, conf_low.values, conf_high.values, alpha=0.2, label="95% CI")

                ax2.set_xlabel("Periode (Tahun–Kuartal)")
                ax2.set_ylabel("Nilai Investasi (Juta Rupiah)")
                ax2.grid(True, linestyle="--", alpha=0.6)
                ax2.legend()

                title_text = f"Best SARIMA (AIC={best_aic:.2f})  order={best_order}  seasonal={best_seasonal}"
                bar_x, bar_y, bar_w, bar_h = 0.02, 0.965, 0.004, 0.06
                bar = patches.FancyBboxPatch((bar_x, bar_y - bar_h), bar_w, bar_h,
                                             boxstyle="square,pad=0", transform=fig2.transFigure,
                                             facecolor="#f59e0b", edgecolor="none")
                fig2.patches.append(bar)
                fig2.text(bar_x + bar_w + 0.01, bar_y - bar_h/2,
                          title_text, ha="left", va="center",
                          fontsize=12, fontweight="bold", color="#111827")

                fig2.subplots_adjust(top=0.88)
                fig2.tight_layout(rect=[0, 0, 1, 0.88])
                st.pyplot(fig2)
