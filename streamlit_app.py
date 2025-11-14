# streamlit_app.py
# -*- coding: utf-8 -*-
import json, re, os, hashlib, platform
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import sklearn


# =======================
# Artifacts / Data Load
# =======================
@st.cache_resource
def load_artifacts():
    """
    統合データは気候入りを最優先で読む（with_climate → enriched → clusters）。
    使った CSV のパスも返す（Fingerprint表示で使用）。
    """
    candidates = [
        "data/city_master_enriched_with_climate.csv",
        "data/city_master_enriched.csv",
        "data/city_master_enriched_with_clusters.csv",
    ]
    df_path = next((p for p in candidates if Path(p).exists()), None)
    if not df_path:
        raise FileNotFoundError("統合CSVが見つかりません。with_climate/enriched/clusters のいずれかを用意してください。")
    df = pd.read_csv(df_path)

    scaler = joblib.load("models/scaler.pkl")
    kmeans = joblib.load("models/kmeans.pkl")
    nn = joblib.load("models/nn.pkl")
    with open("models/features.json", "r") as f:
        meta = json.load(f)

    feats_model = meta.get("feature_columns", [])
    if not feats_model:
        candidates = ["nature_score","food_score","safety_score","cost_score","beach_score","climate_score"]
        feats_model = [c for c in candidates if c in df.columns]
    return df, scaler, kmeans, nn, feats_model, df_path


# =======================
# Fingerprint helpers
# =======================
def _sha12_from_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]

def compute_fingerprint(df: pd.DataFrame, feats_model: list[str], df_path: str | None, scaler) -> dict:
    """現在の構成が“前回と同じか”を目視で確かめやすい指紋を返す。"""
    cols_used = [c for c in feats_model if c in df.columns]
    values = df[cols_used].copy().fillna(0).clip(0, 100)

    # データのハッシュ（サイズが大きくても安定するようCSV文字列化）
    h_df = _sha12_from_bytes(values.to_csv(index=False).encode("utf-8"))
    # スケーラーのハッシュ（平均・スケールを連結）
    mean_ = getattr(scaler, "mean_", np.array([])).astype(float)
    scale_ = getattr(scaler, "scale_", np.array([])).astype(float)
    scaler_blob = np.concatenate([mean_, scale_]).tobytes()
    h_scaler = _sha12_from_bytes(scaler_blob)
    # 特徴量セットのハッシュ（順序も含めて固定）
    h_feats = _sha12_from_bytes(",".join(cols_used).encode("utf-8"))

    # CSVメタ
    mtime = None
    try:
        if df_path and os.path.exists(df_path):
            mtime = os.path.getmtime(df_path)
    except Exception:
        pass

    return {
        "app_version": "v4-session-fixed-pca",
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "data": {
            "path": df_path,
            "rows": int(len(df)),
            "cols_used": cols_used,
            "file_mtime": mtime,     # 秒（UNIX time）
            "hash_df_12": h_df,      # 値が変わると変化
        },
        "scaler": {
            "hash_scaler_12": h_scaler,
            "mean_len": int(mean_.size),
            "scale_len": int(scale_.size),
        },
        "features": {
            "count": len(cols_used),
            "hash_feats_12": h_feats,
        },
    }


# =======================
# Climate helpers
# =======================
def _has_temp_cols(cols):
    toks = ["tavg","tmean","temp","temperature"]
    return any(any(t in str(c).lower() for t in toks) for c in cols)

def _norm_key(s):
    if pd.isna(s): return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _detect_city_country(df):
    low = {c: c.lower() for c in df.columns}
    city = next((c for c in df.columns if "city" in low[c]), None)
    ctry = next((c for c in df.columns if "country" in low[c] or "nation" in low[c]), None)
    return city, ctry

def _detect_monthly_cols(cols):
    month_keys = {
        1:["jan","january","01","_1","-1","m1"], 2:["feb","february","02","_2","-2","m2"],
        3:["mar","march","03","_3","-3","m3"],   4:["apr","april","04","_4","-4","m4"],
        5:["may","05","_5","-5","m5"],          6:["jun","june","06","_6","-6","m6"],
        7:["jul","july","07","_7","-7","m7"],   8:["aug","august","08","_8","-8","m8"],
        9:["sep","sept","september","09","_9","-9","m9"],
        10:["oct","october","10","_10","-10","m10"],
        11:["nov","november","11","_11","-11","m11"],
        12:["dec","december","12","_12","-12","m12"],
    }
    temp_tokens = ["tavg","tmean","temp","temperature","avg_temp","mean_temp"]
    low = {c: c.lower() for c in cols}
    by_month = {m: None for m in range(1,13)}
    for m, keys in month_keys.items():
        best, best_score = None, -1
        for c in cols:
            lc = low[c]
            if not any(tok in lc for tok in temp_tokens): continue
            score = sum(1 for k in keys if k in lc)
            if score > best_score:
                best, best_score = c, score
        if best_score > 0:
            by_month[m] = best
    if all(v is None for v in by_month.values()):
        return None
    return by_month

def force_attach_climate(main_df):
    """main_df に気温列が無い場合、候補ファイルから city/country でマージ。"""
    df = main_df.copy()
    if _has_temp_cols(df.columns):
        return df

    candidates = [
        "data/city_master_enriched_with_climate.csv",
        "data/cities_with_climate_filled_knn.csv",
        "data/cities_with_climate.csv",
    ]
    clim = None
    for p in candidates:
        try:
            tmp = pd.read_csv(p)
            if len(tmp) > 0:
                clim = tmp; break
        except Exception:
            continue
    if clim is None:
        return df

    b_city, b_cty = _detect_city_country(df)
    c_city, c_cty = _detect_city_country(clim)
    if not all([b_city, b_cty, c_city, c_cty]):
        return df

    df["_ck_city"] = df[b_city].map(_norm_key)
    df["_ck_cty"]  = df[b_cty].map(_norm_key)
    clim["_ck_city"] = clim[c_city].map(_norm_key)
    clim["_ck_cty"]  = clim[c_cty].map(_norm_key)

    monthly = _detect_monthly_cols(clim.columns)
    keep = ["_ck_city","_ck_cty"]
    if monthly:
        keep += [c for c in monthly.values() if c]
    keep += [c for c in clim.columns if any(t in c.lower() for t in
             ["temp_c","avg_temp_c","tavg","tmean","mean_temp","mean_temperature"])]

    clim2 = clim[keep].drop_duplicates(["_ck_city","_ck_cty"])
    out = df.merge(clim2, on=["_ck_city","_ck_cty"], how="left")

    if monthly:
        for m, src in monthly.items():
            if src and src in out.columns:
                out[f"tavg_{m:02d}"] = pd.to_numeric(out[src], errors="coerce")

    out = out.drop(columns=[c for c in out.columns if c.startswith("_ck_")], errors="ignore")
    return out

def _find_monthly_temp_column_names(columns):
    months = {}
    for m in range(1,13):
        col = f"tavg_{m:02d}"
        if col in columns: months[m] = col
    if len(months) == 12:
        return months
    return _detect_monthly_cols(columns)

def temperature_for_month(df: pd.DataFrame, month: int) -> pd.Series:
    monthly = _find_monthly_temp_column_names(df.columns)
    if monthly and monthly.get(month):
        return pd.to_numeric(df[monthly[month]], errors="coerce")
    for key in ["temp_c","avg_temp_c","tavg","tmean","temperature_c","mean_temp","mean_temperature"]:
        k = next((c for c in df.columns if c.lower() == key), None)
        if k: return pd.to_numeric(df[k], errors="coerce")
    return pd.Series([np.nan]*len(df), index=df.index)

def climate_fit_score_from_range(temp_series: pd.Series, tmin: float, tmax: float) -> pd.Series:
    temp = pd.to_numeric(temp_series, errors="coerce")
    if tmax < tmin: tmin, tmax = tmax, min(tmin, tmax)
    mid = (tmin + tmax) / 2.0
    half = max((tmax - tmin) / 2.0, 1.5)
    sigma = half / 0.68
    fit = np.exp(-0.5 * ((temp - mid)/sigma)**2) * 100.0
    return np.clip(fit, 0, 100)


# =======================
# Ranking helpers
# =======================
RANK_CLIMATE_COL = "climate_score_dynamic"

def make_rank_dataframe(base_df, use_climate_pref, dep_month, pref_temp):
    df_r = base_df.copy()
    if use_climate_pref:
        t = temperature_for_month(df_r, int(dep_month))
        df_r[RANK_CLIMATE_COL] = climate_fit_score_from_range(t, pref_temp[0], pref_temp[1])
    else:
        if "climate_score" in df_r.columns:
            df_r[RANK_CLIMATE_COL] = pd.to_numeric(df_r["climate_score"], errors="coerce").fillna(0)
        else:
            df_r[RANK_CLIMATE_COL] = 0.0
    df_r[RANK_CLIMATE_COL] = df_r[RANK_CLIMATE_COL].fillna(0).clip(0, 100)
    return df_r

def make_rank_features(feats_model, df_for_rank):
    feats = list(feats_model)
    if RANK_CLIMATE_COL not in feats:
        feats.append(RANK_CLIMATE_COL)
    return [c for c in feats if c in df_for_rank.columns]

def make_rank_weights(raw_weights, rank_feats):
    w = dict(raw_weights)
    if "climate_score" in w:
        w[RANK_CLIMATE_COL] = float(w.get("climate_score", 0.0))
    return {k: float(w.get(k, 0.0)) for k in rank_feats}

def normalize_weights_pos(weights: dict, feats: list[str]) -> np.ndarray:
    raw = np.array([float(weights.get(f, 0.0)) for f in feats], dtype=float)
    pos = np.clip(raw, 0, None)
    s = pos.sum()
    return pos / s if s > 0 else pos

def filter_with_mins(df: pd.DataFrame, mins: dict) -> pd.DataFrame:
    D = df.copy()
    for k, v in (mins or {}).items():
        if k in D.columns and v is not None:
            D = D[D[k] >= v]
    return D

def weighted_score(df: pd.DataFrame, feats: list[str], weights: dict, mins: dict | None = None,
                   standardize: bool = True) -> pd.DataFrame:
    D = filter_with_mins(df, mins)
    feats = [c for c in feats if c in D.columns]
    if len(D) == 0 or len(feats) == 0:
        return D.assign(recommend_score=np.nan)
    X = D[feats].astype(float).clip(0, 100)
    if standardize:
        mu = X.mean(axis=0); sd = X.std(axis=0).replace(0, 1.0)
        X = (X - mu) / sd
    w = normalize_weights_pos(weights, feats)
    score = np.round((X.values * w).sum(axis=1), 6)  # 小数第6位まで保持
    return D.assign(recommend_score=score).sort_values("recommend_score", ascending=False)

def preference_vector_from_weights(weights: dict, feats: list[str]) -> np.ndarray:
    w = normalize_weights_pos(weights, feats)
    return w * 100.0


# =======================
# PCA (fit-once & reuse)
# =======================
def _summarize_axis(loadings_df: pd.DataFrame, col, top_k=2):
    s = loadings_df[col].sort_values(ascending=False)
    pos = [f.replace("_score","") for f in s.head(top_k).index]
    neg = [f.replace("_score","") for f in s.tail(top_k).index]
    return pos, neg

def ensure_pca_once(df: pd.DataFrame, feats: list[str], scaler, kmeans):
    """
    同一セッション内で PC1/PC2 を固定するため、
    PCA を一度だけfit→Session Stateに保存し、Clusters/Neighborsで共有する。
    """
    feats = [c for c in feats if c in df.columns]
    key = "pca_state"
    if key not in st.session_state:
        X = df[feats].fillna(0).clip(0, 100).values
        Xs = scaler.transform(X)
        pca = PCA(n_components=2, random_state=42).fit(Xs)
        Z = pca.transform(Xs)
        centers_2d = pca.transform(kmeans.cluster_centers_)
        loadings = pd.DataFrame(pca.components_.T, index=feats, columns=["PC1","PC2"])
        evr = pca.explained_variance_ratio_[:2]
        # バナー用テキストを手動指定
        pc1_text = "PC1 = 自然豊か → ← 食・都市度が高い"
        pc2_text = "PC2 = 安全・快適 ↑ ↓ 物価が安い"
        st.session_state[key] = {
            "pca": pca, "Z": Z, "centers_2d": centers_2d, "feats": feats,
            "loadings": loadings, "evr": evr, "pc1_text": pc1_text, "pc2_text": pc2_text
        }
    return st.session_state[key]


# ================
# Cluster labeling
# ================
PALETTE = {i: cm.get_cmap("tab20")(i) for i in range(20)}  # 色固定

def cluster_nickname(df, feats, cl):
    sub = df[df["cluster"] == cl]
    if len(sub) == 0:
        return "n/a"
    means = sub[feats].mean()
    lift = (means - df[feats].mean()).sort_values(ascending=False)
    ups  = [f.replace("_score","") for f in lift.head(2).index]
    downs= [f.replace("_score","") for f in lift.tail(1).index]
    return f"{' & '.join(ups)} high, {downs[0]} low"


# =========
# UI / App
# =========
def main():
    st.set_page_config(page_title="Destination Suggester", layout="wide")
    st.title("🌏 Destination Suggester — City Recommender")

    try:
        df, scaler, kmeans, nn, feats_model, df_path = load_artifacts()
    except Exception as e:
        st.error(f"Artifacts not found: {e}")
        st.stop()

    with st.sidebar:
        st.header("Preferences (Weights)")
        w_nature  = st.slider("Nature",  0.0, 1.0, 0.6, 0.05)
        w_food    = st.slider("Food",    0.0, 1.0, 0.5, 0.05)
        w_safety  = st.slider("Safety",  0.0, 1.0, 0.5, 0.05)
        w_cost    = st.slider("Cost (cheaper=better)", 0.0, 1.0, 0.4, 0.05)
        w_beach   = st.slider("Beach",   0.0, 1.0, 0.3, 0.05)
        w_climate = st.slider("Climate", 0.0, 1.0, 0.5, 0.05)
        weights = {
            "nature_score": w_nature, "food_score": w_food, "safety_score": w_safety,
            "cost_score": w_cost, "beach_score": w_beach, "climate_score": w_climate,
        }
        
                # --- ソフト支配モード：特定スライダーがほぼ最大のとき、その指標を90%重視 ---
        def apply_soft_dominance(weights: dict, floor: float = 0.90, thresh: float = 0.95) -> dict:
            """
            weights: {"nature_score": float, "food_score": float, ...} 0.0〜1.0
            floor:   支配的な指標に最低でも配分する割合（例：0.90 = 90%）
            thresh:  「ほぼ最大」とみなすしきい値（例：0.95）
            ルール:  最大の指標が thresh を超えたら、その指標に floor を配分。
                     残り 1 - floor を他指標に元の比率で再配分（全他指標が0なら 100% 支配）。
            """
            if not weights:
                return weights

            # 最大のキーを決定
            max_key = max(weights, key=lambda k: float(weights.get(k, 0.0)))
            max_val = float(weights.get(max_key, 0.0))

            if max_val < thresh:
                # しきい値未満なら何もしない
                return weights

            # 残り10%を他に比例配分
            others = {k: max(float(v), 0.0) for k, v in weights.items() if k != max_key}
            others_sum = sum(others.values())

            new_w = dict(weights)
            new_w[max_key] = floor
            if others_sum > 0:
                for k, v in others.items():
                    new_w[k] = (1.0 - floor) * (v / others_sum)
            else:
                # 他が全部0なら、完全に max_key に 100% を配分
                for k in others.keys():
                    new_w[k] = 0.0
                new_w[max_key] = 1.0
            return new_w

        # --- ここで適用 ---
        weights = apply_soft_dominance(weights)


        st.markdown("---")
        st.subheader("Climate preference")
        dep_month = st.selectbox("Departure month", list(range(1,13)), format_func=lambda m: f"{m:02d}")
        preset = st.selectbox("Temperature preset",
                              ["Custom","Cool (10–18°C)","Mild (18–24°C)","Warm (24–30°C)","Hot (30–35°C)"])
        preset_ranges = {
            "Cool (10–18°C)": (10,18), "Mild (18–24°C)": (18,24),
            "Warm (24–30°C)": (24,30), "Hot (30–35°C)": (30,35),
        }
        if preset == "Custom":
            c1, c2 = st.columns(2)
            with c1: tmin = st.number_input("Min °C", -20, 45, 18, 1)
            with c2: tmax = st.number_input("Max °C", -20, 45, 24, 1)
            pref_temp = (min(tmin,tmax), max(tmin,tmax))
        else:
            pref_temp = preset_ranges[preset]
        use_climate_pref = st.checkbox("Use preferred temperature to override climate score", value=True)

        st.markdown("---")
        st.subheader("Minimum thresholds")
        min_nat   = st.slider("Min Nature", 0,100,0,5)
        min_food  = st.slider("Min Food",   0,100,0,5)
        min_safe  = st.slider("Min Safety", 0,100,0,5)
        min_cost  = st.slider("Min CostScore", 0,100,0,5)
        min_beach = st.slider("Min BeachScore", 0,100,0,5)
        min_clim  = st.slider("Min Climate", 0,100,0,5)

        st.markdown("---")
        st.subheader("Filters / Options")
        countries = sorted(df["country"].dropna().unique().tolist())
        sel_countries = st.multiselect("Countries", countries)
        topn = st.number_input("Top N", 5, 100, 20, 5)
        standardize = False

        # ===== Fingerprint（構成ハッシュ）表示 =====
        st.markdown("---")
        if st.checkbox("Show session fingerprint (構成ハッシュの確認)", value=False):
            fp = compute_fingerprint(df, feats_model, df_path, scaler)
            st.code(json.dumps(fp, indent=2, ensure_ascii=False))

    # Base filter
    base = df.copy()
    if sel_countries:
        base = base[base["country"].isin(sel_countries)]

    # 気候列を必ず抱かせる（必要時）
    base = force_attach_climate(base)

    # Ranking DF/features/weights
    df_for_rank = make_rank_dataframe(base, use_climate_pref, dep_month, pref_temp)
    rank_feats = make_rank_features(feats_model, df_for_rank)
    rank_weights = make_rank_weights(weights, rank_feats)

    mins = {
        "nature_score": min_nat, "food_score": min_food, "safety_score": min_safe,
        "cost_score": min_cost, "beach_score": min_beach, RANK_CLIMATE_COL: min_clim,
    }

    # ===== Tabs =====
    tab_reco, tab_cluster, tab_nn = st.tabs(["Recommendations","Clusters (PCA 2D)","Neighbors (PCA 2D)"])

    # ----------------- Recommendations -----------------
    with tab_reco:
        st.subheader("Recommendations")

        is_climate_only = (
            RANK_CLIMATE_COL in rank_feats
            and rank_weights.get(RANK_CLIMATE_COL, 0.0) >= 0.999
            and np.all(np.array([rank_weights.get(f,0.0) for f in rank_feats if f != RANK_CLIMATE_COL]) <= 1e-6)
        )
        if is_climate_only:
            ranked = filter_with_mins(df_for_rank, mins).sort_values(RANK_CLIMATE_COL, ascending=False)
            mode_note = "（climate単独ソート）"
        else:
            ranked = weighted_score(df_for_rank, rank_feats, rank_weights, mins=mins, standardize=standardize)
            mode_note = ""

        if len(ranked) == 0:
            st.warning("⚠️ 0件になったため閾値なしで表示します。")
            if is_climate_only:
                ranked = df_for_rank.sort_values(RANK_CLIMATE_COL, ascending=False)
            else:
                ranked = weighted_score(df_for_rank, rank_feats, rank_weights, mins=None, standardize=standardize)

        show_cols = ["city","country"] + [c for c in ["recommend_score"] + rank_feats + ["cluster"] if c in ranked.columns]
        st.caption(f"Ranking{mode_note}")
        st.dataframe(ranked.head(topn)[show_cols].reset_index(drop=True), width="stretch")

        # CSV ダウンロード
        csv_bytes = ranked.head(topn)[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download ranking (CSV)", data=csv_bytes,
                           file_name="recommendations.csv", mime="text/csv")

        with st.expander("Debug: climate & filters"):
            s = df_for_rank.get(RANK_CLIMATE_COL, pd.Series([], dtype=float))
            st.write({
                "rows_all": int(len(df)),
                "rows_after_country_filter": int(len(base)),
                "rows_after_thresholds": int(len(filter_with_mins(df_for_rank, mins))),
                f"{RANK_CLIMATE_COL}.count": int(s.count()) if len(s) else 0,
                f"{RANK_CLIMATE_COL}.unique": int(s.nunique()) if len(s) else 0,
                f"{RANK_CLIMATE_COL}.min": float(s.min()) if s.count() else None,
                f"{RANK_CLIMATE_COL}.max": float(s.max()) if s.count() else None,
                "rank_feats": rank_feats,
                "is_climate_only": bool(is_climate_only),
                "standardize": standardize,
            })
            mapping = _find_monthly_temp_column_names(df_for_rank.columns)
            st.write({"monthly_cols_detected": mapping})
            st.dataframe(
                df_for_rank[["city","country",RANK_CLIMATE_COL]].sample(min(5,len(df_for_rank)), random_state=1),
                width="stretch"
            )

    # ======== PCA を一度だけfitして両タブで共有 ========
    pca_state = ensure_pca_once(df, [c for c in feats_model if c in df.columns], scaler, kmeans)
    Z = pca_state["Z"]; centers_2d = pca_state["centers_2d"]
    feats_used = pca_state["feats"]; loadings = pca_state["loadings"]
    evr = pca_state["evr"]; pc1_text = pca_state["pc1_text"]; pc2_text = pca_state["pc2_text"]
    X_std_all = scaler.transform(df[feats_used].fillna(0).clip(0,100).values)

    # ----------------- Clusters (PCA) -----------------
    with tab_cluster:
        st.subheader("KMeans Clusters — 2D (PCA)")
        try:
            fig = plt.figure(figsize=(7,5))
            if "cluster" in df.columns:
                for cl in sorted(df["cluster"].dropna().unique()):
                    idx = df["cluster"] == cl
                    color = PALETTE[int(cl)]
                    plt.scatter(Z[idx,0], Z[idx,1], s=12, alpha=0.8, color=color, label=f"cluster {int(cl)}")
                plt.scatter(centers_2d[:,0], centers_2d[:,1], s=150, marker="X", color="green")
                plt.legend(markerscale=1.5, fontsize=8)
            else:
                plt.scatter(Z[:,0], Z[:,1], s=12, alpha=0.7)

            plt.title("PCA projection of standardized features")
            plt.xlabel(f"PC1  (explained {evr[0]*100:.1f}%)")
            plt.ylabel(f"PC2  (explained {evr[1]*100:.1f}%)")

            x_min, x_max = Z[:,0].min(), Z[:,0].max()
            y_min, y_max = Z[:,1].min(), Z[:,1].max()
            pad_x = 0.02*(x_max-x_min+1e-9)
            pad_y = 0.02*(y_max-y_min+1e-9)
            plt.text(x_max - pad_x, y_max - pad_y, pc1_text, ha="right", va="top",
                     fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.6))
            plt.text(x_min + pad_x, y_min + pad_y, pc2_text, ha="left", va="bottom",
                     fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.6))

            st.pyplot(fig)

            legend_labels = []
            if "cluster" in df.columns:
                for cl in sorted(df["cluster"].dropna().unique()):
                    legend_labels.append(
                        f"cluster {int(cl)} — {cluster_nickname(df, feats_used, int(cl))}"
                    )
            if legend_labels:
                st.write("**Cluster legend (color ↔ meaning)**")
                st.write("\n".join(f"- {lbl}" for lbl in legend_labels))

            with st.expander("PC1 / PC2 loadings (現在のPCA)"):
                st.write({"explained_variance_ratio": [round(float(e),3) for e in evr]})
                st.dataframe(loadings.round(3), width="stretch")

            with st.expander("Cluster representatives (Top-3 cities nearest to centroid)"):
                if "cluster" in df.columns:
                    reps_all = []
                    centers_std = kmeans.cluster_centers_
                    for cl in sorted(df["cluster"].dropna().unique()):
                        idx = np.where(df["cluster"].values == cl)[0]
                        if len(idx) == 0: continue
                        d = np.linalg.norm(X_std_all[idx] - centers_std[int(cl)], axis=1)
                        best = df.iloc[idx].copy()
                        best = best.assign(distance_to_centroid=np.round(d, 3)).sort_values("distance_to_centroid")
                        reps_all.append(best[["city","country","distance_to_centroid"]].head(3).assign(cluster=int(cl)))
                    if reps_all:
                        out = pd.concat(reps_all, ignore_index=True)
                        st.dataframe(out[["cluster","city","country","distance_to_centroid"]], width="stretch")
                else:
                    st.info("クラスタ列が見つからないため代表都市は表示できません。")

        except Exception as e:
            st.error(f"Cluster visualization error: {e}")

    # ----------------- Neighbors (PCA) -----------------
    with tab_nn:
        st.subheader("Nearest Neighbors — 2D (PCA)")
        try:
            mode = st.radio("Query mode", ["Pick a city","Use current sidebar weights"], horizontal=True)
            if mode == "Pick a city":
                sel_city = st.selectbox("City", df["city"].tolist(), index=0)
                q_idx = df.index[df["city"] == sel_city][0]
                q_vec = X_std_all[q_idx:q_idx+1]
            else:
                feats_for_nn = feats_used
                pref = preference_vector_from_weights(weights, feats_for_nn)
                pref_df = pd.DataFrame([pref], columns=feats_for_nn)
                q_vec = scaler.transform(pref_df.values)

            dists, idxs = nn.kneighbors(q_vec, n_neighbors=15, return_distance=True)
            idxs = idxs[0].tolist(); dists = dists[0].tolist()
            cols_show = ["city","country"] + feats_used + (["cluster"] if "cluster" in df.columns else [])
            nn_df = df.iloc[idxs][cols_show].copy()
            nn_df.insert(0, "distance", np.round(dists, 3))
            st.dataframe(nn_df.reset_index(drop=True), width="stretch")

            # 散布図（Clustersと同じ PCA を使用）＋解釈テキストも表示
            fig = plt.figure(figsize=(7,5))
            plt.scatter(Z[:,0], Z[:,1], s=8, alpha=0.2)
            Z_nn = Z[idxs,:]
            plt.scatter(Z_nn[:,0], Z_nn[:,1], s=60, marker="o")
            if mode == "Pick a city":
                plt.scatter(Z[q_idx,0], Z[q_idx,1], s=160, marker="*", linewidths=1)
            else:
                plt.scatter(Z_nn[0,0], Z_nn[0,1], s=160, marker="*", linewidths=1)

            plt.title("Nearest neighbors in PCA space")
            plt.xlabel(f"PC1  (explained {evr[0]*100:.1f}%)")
            plt.ylabel(f"PC2  (explained {evr[1]*100:.1f}%)")

            # 同じ解釈ラベルを注釈
            x_min, x_max = Z[:,0].min(), Z[:,0].max()
            y_min, y_max = Z[:,1].min(), Z[:,1].max()
            pad_x = 0.02*(x_max-x_min+1e-9)
            pad_y = 0.02*(y_max-y_min+1e-9)
            plt.text(x_max - pad_x, y_max - pad_y, pc1_text, ha="right", va="top",
                     fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.6))
            plt.text(x_min + pad_x, y_min + pad_y, pc2_text, ha="left", va="bottom",
                     fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.6))

            st.pyplot(fig)

            # 地図（lat/lon があれば）
            lat_col = next((c for c in df.columns if c.lower() in ["lat","latitude"]), None)
            lon_col = next((c for c in df.columns if c.lower() in ["lon","lng","long","longitude"]), None)
            if lat_col and lon_col:
                map_base = df.iloc[idxs][["city","country", lat_col, lon_col]].rename(
                    columns={lat_col: "lat", lon_col: "lon"}
                ).dropna().head(15)
                if not map_base.empty:
                    st.map(map_base)
                else:
                    st.info("近傍候補に座標がありませんでした。")
            else:
                st.info("lat/lon が見つからないため地図は省略しました。")

        except Exception as e:
            st.error(f"NN visualization error: {e}")


if __name__ == "__main__":
    main()
