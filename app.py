"""
🏠 Airbnb Austin – Dashboard Analisis, Clustering & Prediksi Harga
Streamlit App | Data: Airbnb Listings Austin Texas
"""

import ast, pickle, warnings
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42

# ════════════════════════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🏠 Airbnb Austin Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .kpi-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 18px 12px; border-radius: 12px; color: white;
        text-align: center; margin: 4px 0;
    }
    .kpi-val  { font-size: 1.8rem; font-weight: 700; line-height: 1.2; }
    .kpi-lbl  { font-size: 0.78rem; opacity: 0.88; margin-top: 4px; }
    .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 600; }
    h1 { color: #FF5A5F; }
    .section-title { font-size: 1.1rem; font-weight: 700;
                     border-left: 4px solid #FF5A5F;
                     padding-left: 10px; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# KONSTANTA
# ════════════════════════════════════════════════════════════════

TOP_AMENITIES = [
    'wifi','kitchen','air conditioning','heating','washer','dryer',
    'free parking on premises','free street parking','pool','hot tub',
    'gym','elevator','tv','dedicated workspace','self check-in',
    'pets allowed','smoking allowed','bathtub','dishwasher',
    'refrigerator','microwave','coffee maker','hair dryer','iron',
    'fire extinguisher','first aid kit','carbon monoxide alarm',
    'smoke alarm','patio or balcony','bbq grill','backyard',
    'outdoor furniture','long term stays allowed','bed linens',
    'extra pillows and blankets','shampoo','private entrance',
    'baby crib','high chair','luggage dropoff allowed',
]
PALETTE = ['#2ecc71','#3498db','#e67e22','#e74c3c','#9b59b6',
           '#1abc9c','#f39c12','#d35400']

def safe_key(s):
    return s.replace(' ','_').replace('/','_').replace('-','_')

# ════════════════════════════════════════════════════════════════
# LOAD & PREPROCESS DATA
# ════════════════════════════════════════════════════════════════

def parse_amenities(x):
    try:
        return [a.lower() for a in ast.literal_eval(x)] if isinstance(x,str) else []
    except:
        return []

@st.cache_data(show_spinner="🔄 Memuat & memproses data...")
def load_data():
    # File Excel dibaca langsung dari repo (Streamlit Cloud clone seluruh repo)
    df = pd.read_excel("petra - - - listings_austins.xlsx")

    # Tanggal
    for c in ['host_since','first_review','last_review','last_scraped']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')

    # Price
    df = df.dropna(subset=['price'])
    df = df[df['price'].between(10, 1500)].copy()

    # Bedrooms ×10 → bagi 10
    df['bedrooms_clean'] = (df['bedrooms'] / 10).clip(0, 20)

    # Bathrooms
    df['bathrooms_num'] = (
        df['bathrooms_text'].astype(str)
        .str.extract(r'(\d+\.?\d*)')[0].astype(float)
    )

    # Boolean
    for c in ['host_is_superhost','instant_bookable',
              'host_has_profile_pic','host_identity_verified','has_availability']:
        if c in df.columns:
            df[c] = df[c].map({'t':1,'f':0,True:1,False:0})

    # Host years
    df['host_years'] = (
        (pd.Timestamp.now() - df['host_since']).dt.days / 365
    ).clip(0,30).round(1)

    # Amenities
    df['amenities_list'] = df['amenities'].apply(parse_amenities)
    df['amenity_count']  = df['amenities_list'].apply(len)
    for amen in TOP_AMENITIES:
        df[f'amen_{safe_key(amen)}'] = df['amenities_list'].apply(
            lambda lst: 1 if amen in lst else 0)

    # Kategori harga
    df['price_category'] = pd.cut(df['price'],
        bins=[0,75,150,300,1500],
        labels=['Budget (<$75)','Mid ($75-150)','Premium ($150-300)','Luxury (>$300)'])

    df['neighbourhood_cleansed'] = df['neighbourhood_cleansed'].astype(str)
    return df


@st.cache_resource(show_spinner="⚙️ Melatih model clustering & prediksi harga...")
def train_models(df):
    amen_cols   = [c for c in df.columns if c.startswith('amen_')]
    struct_cols = ['bedrooms_clean','bathrooms_num','accommodates',
                   'amenity_count','instant_bookable']
    struct_cols = [c for c in struct_cols if c in df.columns]

    # ── CLUSTERING ────────────────────────────────────────────
    X_cl     = df[amen_cols + struct_cols].fillna(0)
    scaler   = StandardScaler()
    X_sc     = scaler.fit_transform(X_cl)

    # Cari K optimal (2–8)
    sil_scores = []
    for k in range(2, 9):
        km  = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        lbl = km.fit_predict(X_sc)
        sil_scores.append(silhouette_score(X_sc, lbl, sample_size=2000, random_state=SEED))
    best_k = range(2, 9)[sil_scores.index(max(sil_scores))]

    km_final = KMeans(n_clusters=best_k, random_state=SEED, n_init=20)
    df = df.copy()
    df['cluster'] = km_final.fit_predict(X_sc)

    rank_labels = {0:'🟢 Ekonomis',1:'🔵 Standar',2:'🟠 Nyaman',3:'🔴 Premium',4:'⭐ Mewah'}
    price_rank  = df.groupby('cluster')['price'].mean().rank().astype(int)
    df['cluster_rank']  = df['cluster'].map(price_rank - 1)
    df['cluster_label'] = df['cluster_rank'].map(
        {i: rank_labels.get(i, f'Cluster {i}') for i in range(best_k)})

    # PCA untuk visualisasi
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_sc)
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]
    explained = pca.explained_variance_ratio_ * 100

    # ── PREDIKSI HARGA ────────────────────────────────────────
    le_room = LabelEncoder()
    le_nb   = LabelEncoder()
    df['room_type_enc']     = le_room.fit_transform(df['room_type'])
    df['neighbourhood_enc'] = le_nb.fit_transform(df['neighbourhood_cleansed'])

    feat_cols = (
        ['bedrooms_clean','bathrooms_num','accommodates','beds',
         'amenity_count','minimum_nights','availability_365',
         'number_of_reviews','review_scores_rating','review_scores_cleanliness',
         'review_scores_value','review_scores_location',
         'host_is_superhost','instant_bookable','host_years',
         'room_type_enc','neighbourhood_enc','cluster_rank']
        + amen_cols
    )
    feat_cols = [c for c in feat_cols if c in df.columns]

    ml_df = df[feat_cols+['price']].dropna(subset=['price']).copy()
    ml_df[feat_cols] = ml_df[feat_cols].fillna(ml_df[feat_cols].median())

    X, y = ml_df[feat_cols], ml_df['price']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

    rf = RandomForestRegressor(n_estimators=150, max_depth=12,
                               min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    rf.fit(Xtr, ytr)
    yp_rf = rf.predict(Xte)

    gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08,
                                   max_depth=5, random_state=SEED)
    gb.fit(Xtr, ytr)
    yp_gb = gb.predict(Xte)

    r2_rf = r2_score(yte, yp_rf)
    r2_gb = r2_score(yte, yp_gb)
    best  = rf if r2_rf >= r2_gb else gb
    yp    = yp_rf if r2_rf >= r2_gb else yp_gb
    bname = 'Random Forest' if r2_rf >= r2_gb else 'Gradient Boosting'

    metrics = {
        'RF' : {'MAE':mean_absolute_error(yte,yp_rf),
                'RMSE':np.sqrt(mean_squared_error(yte,yp_rf)),
                'R2':r2_rf,
                'MAPE':np.mean(np.abs((yte-yp_rf)/yte.clip(lower=1)))*100},
        'GB' : {'MAE':mean_absolute_error(yte,yp_gb),
                'RMSE':np.sqrt(mean_squared_error(yte,yp_gb)),
                'R2':r2_gb,
                'MAPE':np.mean(np.abs((yte-yp_gb)/yte.clip(lower=1)))*100},
    }

    return {
        'df'        : df,
        'km'        : km_final,
        'scaler'    : scaler,
        'best_k'    : best_k,
        'sil_scores': sil_scores,
        'pca_exp'   : explained,
        'le_room'   : le_room,
        'le_nb'     : le_nb,
        'feat_cols' : feat_cols,
        'amen_cols' : amen_cols,
        'best_model': best,
        'best_name' : bname,
        'yp'        : yp,
        'yte'       : yte.values,
        'metrics'   : metrics,
    }

# ════════════════════════════════════════════════════════════════
# SIDEBAR FILTER
# ════════════════════════════════════════════════════════════════

def sidebar_filters(df):
    with st.sidebar:
        st.markdown("## 🔍 Filter Data")

        room_types = st.multiselect(
            "Tipe Room", df['room_type'].unique().tolist(),
            default=df['room_type'].unique().tolist())

        price_range = st.slider(
            "Rentang Harga ($)",
            int(df['price'].min()), int(df['price'].max()),
            (int(df['price'].quantile(0.05)), int(df['price'].quantile(0.95))))

        neighbourhoods = st.multiselect(
            "Area (ZIP Code)",
            sorted(df['neighbourhood_cleansed'].unique()),
            default=sorted(df['neighbourhood_cleansed'].unique())[:20])

        superhost = st.checkbox("Superhost saja")
        instant   = st.checkbox("Instant Bookable saja")

        st.markdown("---")
        st.markdown("📊 **Dataset:** Airbnb Austin TX  \n🏠 **10.533** listings  |  **79** kolom")

    return room_types, price_range, neighbourhoods, superhost, instant

def apply_filters(df, room_types, price_range, neighbourhoods, superhost, instant):
    mask = (df['room_type'].isin(room_types) &
            df['price'].between(*price_range) &
            df['neighbourhood_cleansed'].isin(neighbourhoods))
    if superhost: mask &= df['host_is_superhost'] == 1
    if instant:   mask &= df['instant_bookable'] == 1
    return df[mask].copy()

# ════════════════════════════════════════════════════════════════
# KPI CARDS
# ════════════════════════════════════════════════════════════════

def kpi_cards(df):
    cols = st.columns(5)
    cards = [
        (f"{len(df):,}",                            "Total Listings"),
        (f"${df['price'].median():.0f}",             "Median Harga/Malam"),
        (f"{df['review_scores_rating'].mean():.2f}⭐","Rating Rata-rata"),
        (f"{int(df['host_is_superhost'].sum()):,}",  "Superhost"),
        (f"{df['amenity_count'].mean():.0f}",        "Avg Fasilitas"),
    ]
    for col, (val, lbl) in zip(cols, cards):
        col.markdown(
            f'<div class="kpi-box"><div class="kpi-val">{val}</div>'
            f'<div class="kpi-lbl">{lbl}</div></div>',
            unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW EDA
# ════════════════════════════════════════════════════════════════

def tab_overview(df):
    st.markdown('<p class="section-title">Distribusi Harga & Tipe Room</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(df, x='price', nbins=60, color='room_type',
                           title='Distribusi Harga per Malam',
                           color_discrete_sequence=px.colors.qualitative.Set2)
        fig.add_vline(x=df['price'].median(), line_dash='dash', line_color='red',
                      annotation_text=f"Median: ${df['price'].median():.0f}")
        fig.update_layout(height=380, bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        rt = df['room_type'].value_counts().reset_index()
        rt.columns = ['room_type','count']
        fig2 = px.pie(rt, names='room_type', values='count', hole=0.42,
                      title='Distribusi Tipe Room',
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p class="section-title">Kategori Harga & Neighbourhood</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        pc = df['price_category'].value_counts().reset_index()
        pc.columns = ['cat','cnt']
        fig3 = px.bar(pc, x='cat', y='cnt', color='cat', text_auto=True,
                      title='Distribusi Kategori Harga',
                      color_discrete_sequence=['#4CAF50','#2196F3','#FF9800','#F44336'])
        fig3.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        nb = (df.groupby('neighbourhood_cleansed')['price']
              .agg(['mean','count']).reset_index()
              .query('count >= 15').nlargest(12,'mean'))
        fig4 = px.bar(nb, x='mean', y='neighbourhood_cleansed', orientation='h',
                      title='Top 12 Area – Rata-rata Harga', text_auto='$,.0f',
                      color='mean', color_continuous_scale='Blues')
        fig4.update_layout(yaxis={'categoryorder':'total ascending'}, height=350)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<p class="section-title">Dampak Fasilitas Terhadap Harga</p>', unsafe_allow_html=True)
    premium_list = ['pool','hot tub','gym','dedicated workspace','patio or balcony',
                    'bathtub','free parking on premises','pets allowed','washer',
                    'dryer','air conditioning','self check-in','long term stays allowed']
    avg_all = df['price'].mean()
    impact  = []
    for a in premium_list:
        col = f"amen_{safe_key(a)}"
        if col in df.columns:
            v = df[df[col]==1]['price'].mean()
            impact.append({'fasilitas':a,'selisih':v-avg_all})
    imp_df = pd.DataFrame(impact).sort_values('selisih')
    imp_df['warna'] = imp_df['selisih'].apply(lambda x: '+ di atas rata-rata' if x>=0 else '- di bawah rata-rata')
    fig5 = px.bar(imp_df, x='selisih', y='fasilitas', orientation='h',
                  color='warna', text_auto='$,.0f',
                  color_discrete_map={'+ di atas rata-rata':'#27ae60','- di bawah rata-rata':'#e74c3c'},
                  title=f'Selisih Harga vs Rata-rata Global (${avg_all:.0f})')
    fig5.update_layout(height=420)
    st.plotly_chart(fig5, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 2 – CLUSTERING FASILITAS
# ════════════════════════════════════════════════════════════════

def tab_clustering(df, model_data):
    st.info("ℹ️ **Clustering dilakukan MURNI berdasarkan fasilitas & struktur listing — tanpa mempertimbangkan harga.** Harga hanya ditampilkan sebagai informasi hasil.")

    c1, c2 = st.columns(2)
    with c1:
        k_list = list(range(2, 9))
        fig_sil = px.line(x=k_list, y=model_data['sil_scores'], markers=True,
                          title='Silhouette Score per K',
                          labels={'x':'Jumlah Cluster (K)','y':'Silhouette Score'})
        fig_sil.add_vline(x=model_data['best_k'], line_dash='dash', line_color='red',
                          annotation_text=f"K={model_data['best_k']} (optimal)")
        fig_sil.update_layout(height=350)
        st.plotly_chart(fig_sil, use_container_width=True)

    with c2:
        dist = df['cluster_label'].value_counts().reset_index()
        dist.columns = ['cluster','count']
        fig_dist = px.pie(dist, names='cluster', values='count', hole=0.4,
                          title=f'Distribusi Cluster (K={model_data["best_k"]})',
                          color_discrete_sequence=PALETTE)
        fig_dist.update_layout(height=350)
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown('<p class="section-title">Scatter PCA 2D – Sebaran Cluster</p>', unsafe_allow_html=True)
    exp = model_data['pca_exp']
    fig_pca = px.scatter(
        df.sample(min(4000,len(df)), random_state=SEED),
        x='pca1', y='pca2', color='cluster_label',
        opacity=0.55, size_max=6,
        title=f'PCA 2D  (PC1={exp[0]:.1f}%  PC2={exp[1]:.1f}%  — total {exp.sum():.1f}% variance)',
        color_discrete_sequence=PALETTE,
        labels={'pca1':f'PC1 ({exp[0]:.1f}%)','pca2':f'PC2 ({exp[1]:.1f}%)','cluster_label':'Cluster'},
        hover_data={'price':True,'amenity_count':True,'bedrooms_clean':True}
    )
    fig_pca.update_traces(marker=dict(size=4))
    fig_pca.update_layout(height=480)
    st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown('<p class="section-title">Profil Tiap Cluster</p>', unsafe_allow_html=True)
    profile = df.groupby('cluster_label').agg(
        Listings        = ('id',                       'count'),
        Avg_Harga       = ('price',                    'mean'),
        Median_Harga    = ('price',                    'median'),
        Avg_Fasilitas   = ('amenity_count',            'mean'),
        Avg_Kamar       = ('bedrooms_clean',           'mean'),
        Avg_Tamu        = ('accommodates',             'mean'),
        Avg_Rating      = ('review_scores_rating',     'mean'),
        Pool_pct        = ('amen_pool',                'mean'),
        HotTub_pct      = ('amen_hot_tub',             'mean'),
        Parking_pct     = ('amen_free_parking_on_premises','mean'),
        Workspace_pct   = ('amen_dedicated_workspace', 'mean'),
    ).round(2)
    for c in ['Pool_pct','HotTub_pct','Parking_pct','Workspace_pct']:
        profile[c] = (profile[c]*100).round(1).astype(str)+'%'
    profile['Avg_Harga']    = profile['Avg_Harga'].apply(lambda x: f"${x:.0f}")
    profile['Median_Harga'] = profile['Median_Harga'].apply(lambda x: f"${x:.0f}")
    st.dataframe(profile, use_container_width=True)

    st.markdown('<p class="section-title">Distribusi Harga per Cluster</p>', unsafe_allow_html=True)
    fig_box = px.box(df, x='cluster_label', y='price', color='cluster_label',
                     title='Box Plot Harga per Cluster (sebagai informasi, bukan input clustering)',
                     color_discrete_sequence=PALETTE,
                     labels={'cluster_label':'Cluster','price':'Harga ($)'})
    fig_box.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_box, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 3 – PREDIKSI HARGA
# ════════════════════════════════════════════════════════════════

def tab_prediksi(df, model_data):
    st.markdown('<p class="section-title">Performa Model</p>', unsafe_allow_html=True)

    m = model_data['metrics']
    c1, c2, c3, c4 = st.columns(4)
    bk = 'RF' if model_data['best_name']=='Random Forest' else 'GB'
    c1.metric("MAE",  f"${m[bk]['MAE']:.2f}", help="Mean Absolute Error – rata-rata selisih prediksi vs aktual")
    c2.metric("RMSE", f"${m[bk]['RMSE']:.2f}", help="Root Mean Square Error")
    c3.metric("R²",   f"{m[bk]['R2']*100:.1f}%", help="Persen variansi yang dijelaskan model")
    c4.metric("MAPE", f"{m[bk]['MAPE']:.1f}%", help="Mean Absolute Percentage Error")
    st.caption(f"🏆 Model terbaik: **{model_data['best_name']}**")

    c1, c2 = st.columns(2)
    with c1:
        comp_df = pd.DataFrame({
            'Model':['Random Forest','Gradient Boosting'],
            'MAE':  [m['RF']['MAE'], m['GB']['MAE']],
            'RMSE': [m['RF']['RMSE'],m['GB']['RMSE']],
            'R²':   [m['RF']['R2'],  m['GB']['R2']],
        })
        fig_cmp = px.bar(comp_df.melt(id_vars='Model',value_vars=['MAE','RMSE']),
                         x='variable', y='value', color='Model', barmode='group',
                         title='MAE & RMSE Perbandingan Model',
                         color_discrete_sequence=['#4C72B0','#DD8452'], text_auto='$.1f')
        fig_cmp.update_layout(height=370)
        st.plotly_chart(fig_cmp, use_container_width=True)

    with c2:
        yte = model_data['yte']
        yp  = model_data['yp']
        scatter_df = pd.DataFrame({'Aktual':yte,'Prediksi':yp})
        fig_sc = px.scatter(scatter_df.sample(min(1500,len(scatter_df))),
                            x='Aktual', y='Prediksi', opacity=0.45,
                            title=f'Actual vs Predicted – {model_data["best_name"]}')
        max_v = min(float(yte.max()), 800)
        fig_sc.add_shape(type='line',x0=0,y0=0,x1=max_v,y1=max_v,
                         line=dict(color='red',dash='dash',width=2))
        fig_sc.update_layout(height=370)
        st.plotly_chart(fig_sc, use_container_width=True)

    # Feature Importance
    st.markdown('<p class="section-title">Feature Importance (Top 20)</p>', unsafe_allow_html=True)
    fi = pd.DataFrame({
        'Fitur': model_data['feat_cols'],
        'Importance': model_data['best_model'].feature_importances_,
    }).sort_values('Importance', ascending=False).head(20)
    fig_fi = px.bar(fi, x='Importance', y='Fitur', orientation='h',
                    title='Top 20 Fitur Paling Berpengaruh',
                    color='Importance', color_continuous_scale='Blues', text_auto='.4f')
    fig_fi.update_layout(yaxis={'categoryorder':'total ascending'}, height=520)
    st.plotly_chart(fig_fi, use_container_width=True)

    # Simulasi prediksi
    st.markdown('<p class="section-title">🧮 Simulasi Prediksi Harga</p>', unsafe_allow_html=True)
    st.markdown("Isi spesifikasi listing untuk mendapatkan prediksi harga per malam:")

    with st.form("form_prediksi"):
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        bedrooms   = r1c1.number_input("Kamar Tidur",    0, 20, 2)
        bathrooms  = r1c2.number_input("Kamar Mandi",    0, 10, 1)
        accommodates = r1c3.number_input("Kapasitas Tamu", 1, 30, 4)
        beds       = r1c4.number_input("Tempat Tidur",   1, 30, 2)

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        room_type_sel  = r2c1.selectbox("Tipe Room", df['room_type'].unique().tolist())
        cluster_sel    = r2c2.selectbox("Cluster Fasilitas", sorted(df['cluster_label'].unique()))
        min_nights     = r2c3.number_input("Min Malam",      1, 365, 2)
        avail_365      = r2c4.number_input("Ketersediaan/Tahun", 0, 365, 200)

        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        rating    = r3c1.slider("Rating",    1.0, 5.0, 4.8, 0.1)
        n_reviews = r3c2.number_input("Jml Reviews",  0, 1000, 30)
        host_yrs  = r3c3.number_input("Pengalaman Host (Thn)", 0.0, 30.0, 3.0)
        amenity_n = r3c4.number_input("Jml Fasilitas", 0, 110, 35)

        r4c1, r4c2, r4c3 = st.columns(3)
        superhost_in = r4c1.checkbox("Superhost", value=True)
        instant_in   = r4c2.checkbox("Instant Bookable", value=True)
        pool_in      = r4c3.checkbox("Ada Pool")

        st.markdown("**Fasilitas tambahan:**")
        fa1, fa2, fa3, fa4, fa5 = st.columns(5)
        hottub_in  = fa1.checkbox("Hot Tub")
        ws_in      = fa2.checkbox("Workspace")
        parking_in = fa3.checkbox("Parking")
        washer_in  = fa4.checkbox("Washer+Dryer")
        pets_in    = fa5.checkbox("Pets Allowed")

        submitted = st.form_submit_button("💰 Prediksi Harga", use_container_width=True)

    if submitted:
        feat_cols = model_data['feat_cols']
        le_room   = model_data['le_room']
        le_nb     = model_data['le_nb']

        # Encode
        try:
            rt_enc = le_room.transform([room_type_sel])[0]
        except:
            rt_enc = 0

        cl_rank = df[df['cluster_label']==cluster_sel]['cluster_rank'].mode()
        cl_rank = int(cl_rank.iloc[0]) if len(cl_rank) > 0 else 0

        example = {c: 0 for c in feat_cols}
        example.update({
            'bedrooms_clean'              : bedrooms,
            'bathrooms_num'               : bathrooms,
            'accommodates'                : accommodates,
            'beds'                        : beds,
            'amenity_count'               : amenity_n,
            'minimum_nights'              : min_nights,
            'availability_365'            : avail_365,
            'number_of_reviews'           : n_reviews,
            'review_scores_rating'        : rating,
            'review_scores_cleanliness'   : rating,
            'review_scores_value'         : rating - 0.1,
            'review_scores_location'      : rating,
            'host_is_superhost'           : int(superhost_in),
            'instant_bookable'            : int(instant_in),
            'host_years'                  : host_yrs,
            'room_type_enc'               : rt_enc,
            'neighbourhood_enc'           : 5,
            'cluster_rank'                : cl_rank,
            'amen_wifi'                   : 1,
            'amen_kitchen'                : 1,
            'amen_pool'                   : int(pool_in),
            'amen_hot_tub'                : int(hottub_in),
            'amen_dedicated_workspace'    : int(ws_in),
            'amen_free_parking_on_premises': int(parking_in),
            'amen_washer'                 : int(washer_in),
            'amen_dryer'                  : int(washer_in),
            'amen_pets_allowed'           : int(pets_in),
            'amen_air_conditioning'       : 1,
            'amen_self_check_in'          : 1,
        })

        X_ex   = pd.DataFrame([example])[feat_cols].fillna(0)
        pred   = model_data['best_model'].predict(X_ex)[0]

        st.markdown("---")
        kp1, kp2, kp3, kp4 = st.columns(4)
        kp1.metric("💰 Harga per Malam", f"${pred:.2f}")
        kp2.metric("📅 per Minggu",      f"${pred*7:.0f}")
        kp3.metric("📅 per Bulan",       f"${pred*30:.0f}")
        kp4.metric("📅 per Tahun",       f"${pred*365:.0f}")

        # Bandingkan dengan rata-rata cluster
        avg_cluster = df[df['cluster_label']==cluster_sel]['price'].mean()
        delta = pred - avg_cluster
        st.info(f"📊 Rata-rata harga cluster **{cluster_sel}** = **${avg_cluster:.0f}** | "
                f"Prediksi Anda {'**${:.0f} di atas**'.format(delta) if delta>=0 else '**${:.0f} di bawah**'.format(abs(delta))} rata-rata cluster")


# ════════════════════════════════════════════════════════════════
# TAB 4 – PETA INTERAKTIF
# ════════════════════════════════════════════════════════════════

def tab_peta(df):
    import pydeck as pdk

    st.markdown('<p class="section-title">Peta Distribusi Listings</p>', unsafe_allow_html=True)

    color_by = st.selectbox("Warna berdasarkan:",
        ['cluster_label', 'price', 'room_type'])

    smp = df[df['price'] < 600].sample(min(4000, len(df)), random_state=SEED).copy()

    # Buat kolom warna RGB
    if color_by == 'price':
        max_p = smp['price'].max()
        smp['color'] = smp['price'].apply(
            lambda p: [int(255 * p/max_p), int(50 * (1-p/max_p)), 50, 180])

    elif color_by == 'cluster_label':
        cluster_color_map = {
            '🟢 Ekonomis' : [46, 204, 113, 180],
            '🔵 Standar'  : [52, 152, 219, 180],
            '🟠 Nyaman'   : [230, 126, 34, 180],
            '🔴 Premium'  : [231, 76, 60, 180],
            '⭐ Mewah'    : [155, 89, 182, 180],
        }
        smp['color'] = smp['cluster_label'].apply(
            lambda x: cluster_color_map.get(x, [100, 100, 100, 180]))

    elif color_by == 'room_type':
        rt_colors = {
            'Entire home/apt': [52, 152, 219, 180],
            'Private room'   : [46, 204, 113, 180],
            'Hotel room'     : [231, 76, 60, 180],
            'Shared room'    : [230, 126, 34, 180],
        }
        smp['color'] = smp['room_type'].apply(
            lambda x: rt_colors.get(x, [120, 120, 120, 180]))

    layer = pdk.Layer(
        'ScatterplotLayer',
        data=smp,
        get_position='[longitude, latitude]',
        get_color='color',
        get_radius=80,
        pickable=True,
    )

    view = pdk.ViewState(
        latitude=30.2672,
        longitude=-97.7431,
        zoom=10,
        pitch=0,
    )

    tooltip = {
        "html": "<b>{name}</b><br/>Harga: ${price}<br/>Tipe: {room_type}<br/>Cluster: {cluster_label}<br/>Rating: {review_scores_rating}<br/>Area: {neighbourhood_cleansed}",
        "style": {"backgroundColor": "steelblue", "color": "white", "fontSize": "12px"}
    }

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip=tooltip,
        map_style='mapbox://styles/mapbox/light-v10',
    ))

    # Legenda manual
    if color_by == 'cluster_label':
        st.markdown("**Legenda Cluster:**")
        cols = st.columns(5)
        legend = [('🟢','Ekonomis','#2ecc71'),('🔵','Standar','#3498db'),
                  ('🟠','Nyaman','#e67e22'),('🔴','Premium','#e74c3c'),('⭐','Mewah','#9b59b6')]
        for col, (icon, label, color) in zip(cols, legend[:df['cluster_label'].nunique()]):
            col.markdown(f"<span style='color:{color}'>⬤</span> {icon} {label}", unsafe_allow_html=True)

    st.caption(f"Menampilkan {len(smp):,} dari {len(df):,} listings · Hover titik untuk detail")

# ════════════════════════════════════════════════════════════════
# TAB 5 – DATA EXPLORER
# ════════════════════════════════════════════════════════════════

def tab_data(df):
    st.markdown('<p class="section-title">Eksplorasi Data</p>', unsafe_allow_html=True)

    disp_cols = st.multiselect("Pilih kolom:", df.columns.tolist(),
        default=['name','neighbourhood_cleansed','room_type','price',
                 'bedrooms_clean','accommodates','amenity_count',
                 'review_scores_rating','cluster_label','host_is_superhost'])

    sort_col = st.selectbox("Urutkan berdasarkan:", disp_cols)
    asc      = st.radio("Urutan:", ["Ascending","Descending"]) == "Ascending"

    shown = df[disp_cols].sort_values(sort_col, ascending=asc)
    st.dataframe(shown, use_container_width=True, height=480)

    st.download_button(
        "⬇️ Download Data Filtered (CSV)",
        shown.to_csv(index=False).encode('utf-8'),
        file_name='airbnb_austin_filtered.csv', mime='text/csv',
    )
    st.caption(f"Menampilkan {len(shown):,} baris")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    st.title("🏠 Airbnb Austin – Dashboard Analisis")
    st.markdown("**Analisis komprehensif** 10.533 listings Airbnb Austin Texas  ·  "
                "Clustering Fasilitas  ·  Prediksi Harga per Malam")
    st.markdown("---")

    df_raw = load_data()
    model_data = train_models(df_raw)
    df = model_data['df']

    rt, pr, nb, sh, ins = sidebar_filters(df)
    df_f = apply_filters(df, rt, pr, nb, sh, ins)

    st.markdown(f"📌 **{len(df_f):,}** listings ditampilkan dari **{len(df):,}** total")
    kpi_cards(df_f)
    st.markdown("---")

    tabs = st.tabs([
        "📊 Overview EDA",
        "🔵 Clustering Fasilitas",
        "💰 Prediksi Harga",
        "🗺️ Peta Interaktif",
        "🔎 Data Explorer",
    ])
    with tabs[0]: tab_overview(df_f)
    with tabs[1]: tab_clustering(df_f, model_data)
    with tabs[2]: tab_prediksi(df_f, model_data)
    with tabs[3]: tab_peta(df_f)
    with tabs[4]: tab_data(df_f)

    st.markdown("---")
    st.markdown("🔧 Dibuat dengan **Streamlit** | "
                "Model: **Random Forest & Gradient Boosting** | "
                "Clustering: **K-Means**")

if __name__ == "__main__":
    main()
